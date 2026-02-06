"""GPU Memory Monitor and Optimizer

Provides real-time GPU memory monitoring and automatic optimization.
"""

import torch
import psutil
import asyncio
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from loguru import logger
import time
from collections import deque


@dataclass
class MemoryStats:
    """GPU memory statistics."""

    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float
    timestamp: float


@dataclass
class GPUMemoryThresholds:
    """Memory thresholds for optimization."""

    warning_mb: float = 5120  # 5GB
    critical_mb: float = 6144  # 6GB (on 8GB GPU)
    emergency_mb: float = 7168  # 7GB


class GPUMemoryMonitor:
    """
    Monitor GPU memory usage and trigger optimizations.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        check_interval: float = 1.0,
        thresholds: Optional[GPUMemoryThresholds] = None,
    ):
        self.device = device
        self.check_interval = check_interval
        self.thresholds = thresholds or GPUMemoryThresholds()

        self.is_running = False
        self._monitor_task = None
        self._callbacks = []

        # Statistics
        self.history = deque(maxlen=1000)  # Last 1000 readings
        self.peak_memory = 0
        self.leak_detected = False

        logger.info(f"GPUMemoryMonitor initialized (device: {device})")

    def start(self):
        """Start monitoring."""
        if not self.is_running:
            self.is_running = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("GPU memory monitoring started")

    def stop(self):
        """Stop monitoring."""
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        logger.info("GPU memory monitoring stopped")

    def register_callback(self, callback: Callable[[MemoryStats, str], None]):
        """Register callback for memory alerts.

        Args:
            callback: Function(stats, level) where level is 'info', 'warning', 'critical'
        """
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable):
        """Unregister callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                stats = self._get_memory_stats()
                self.history.append(stats)

                # Update peak
                if stats.allocated_mb > self.peak_memory:
                    self.peak_memory = stats.allocated_mb

                # Check thresholds
                level = self._check_thresholds(stats)

                if level != "normal":
                    await self._trigger_callbacks(stats, level)

                # Leak detection
                if len(self.history) >= 100:
                    await self._detect_leak()

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    def _get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return MemoryStats(0, 0, 0, 0, 0, time.time())

        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2

        # Get free memory from cache
        free = total - reserved

        utilization = (allocated / total) * 100 if total > 0 else 0

        return MemoryStats(
            allocated_mb=allocated,
            reserved_mb=reserved,
            free_mb=free,
            total_mb=total,
            utilization_percent=utilization,
            timestamp=time.time(),
        )

    def _check_thresholds(self, stats: MemoryStats) -> str:
        """Check memory against thresholds."""
        if stats.allocated_mb > self.thresholds.emergency_mb:
            return "emergency"
        elif stats.allocated_mb > self.thresholds.critical_mb:
            return "critical"
        elif stats.allocated_mb > self.thresholds.warning_mb:
            return "warning"
        return "normal"

    async def _trigger_callbacks(self, stats: MemoryStats, level: str):
        """Trigger registered callbacks."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(stats, level)
                else:
                    callback(stats, level)
            except Exception as e:
                logger.error(f"Memory callback error: {e}")

    async def _detect_leak(self):
        """Detect memory leaks by analyzing trend."""
        if len(self.history) < 100:
            return

        # Get last 100 readings
        recent = list(self.history)[-100:]

        # Check for steady increase
        start_mem = recent[0].allocated_mb
        end_mem = recent[-1].allocated_mb

        # If memory increased by more than 100MB over 100 readings
        if end_mem - start_mem > 100:
            # Check if it's a steady increase (not just spikes)
            increasing_count = 0
            for i in range(1, len(recent)):
                if recent[i].allocated_mb > recent[i - 1].allocated_mb:
                    increasing_count += 1

            # If 70% of readings show increase, it's a leak
            if increasing_count > 70:
                if not self.leak_detected:
                    self.leak_detected = True
                    logger.warning(
                        f"⚠ Memory leak detected! Growth: {end_mem - start_mem:.1f}MB"
                    )
                    await self._trigger_callbacks(recent[-1], "leak_detected")
        else:
            self.leak_detected = False

    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        return self._get_memory_stats()

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.history:
            return {}

        allocated = [s.allocated_mb for s in self.history]

        return {
            "current_mb": allocated[-1] if allocated else 0,
            "peak_mb": self.peak_memory,
            "average_mb": sum(allocated) / len(allocated) if allocated else 0,
            "min_mb": min(allocated) if allocated else 0,
            "max_mb": max(allocated) if allocated else 0,
            "readings_count": len(self.history),
            "leak_detected": self.leak_detected,
        }


class GPUMemoryOptimizer:
    """
    Automatic GPU memory optimization strategies.
    """

    def __init__(self, monitor: GPUMemoryMonitor):
        self.monitor = monitor
        self.emergency_actions = []
        self.critical_actions = []
        self.warning_actions = []

        # Register as callback
        self.monitor.register_callback(self._on_memory_alert)

        logger.info("GPUMemoryOptimizer initialized")

    def register_emergency_action(self, action: Callable):
        """Register action for emergency memory situations."""
        self.emergency_actions.append(action)

    def register_critical_action(self, action: Callable):
        """Register action for critical memory situations."""
        self.critical_actions.append(action)

    def register_warning_action(self, action: Callable):
        """Register action for warning memory situations."""
        self.warning_actions.append(action)

    async def _on_memory_alert(self, stats: MemoryStats, level: str):
        """Handle memory alerts."""
        logger.warning(
            f"Memory alert [{level}]: {stats.allocated_mb:.1f}MB / {stats.total_mb:.1f}MB"
        )

        if level == "emergency":
            await self._execute_emergency_actions()
        elif level == "critical":
            await self._execute_critical_actions()
        elif level == "warning":
            await self._execute_warning_actions()

    async def _execute_emergency_actions(self):
        """Execute emergency memory recovery."""
        logger.error("Executing EMERGENCY memory recovery")

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Execute registered actions
        for action in self.emergency_actions:
            try:
                if asyncio.iscoroutinefunction(action):
                    await action()
                else:
                    action()
            except Exception as e:
                logger.error(f"Emergency action failed: {e}")

    async def _execute_critical_actions(self):
        """Execute critical memory recovery."""
        logger.warning("Executing CRITICAL memory recovery")

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Execute registered actions
        for action in self.critical_actions:
            try:
                if asyncio.iscoroutinefunction(action):
                    await action()
                else:
                    action()
            except Exception as e:
                logger.error(f"Critical action failed: {e}")

    async def _execute_warning_actions(self):
        """Execute warning actions."""
        logger.info("Executing WARNING memory optimization")

        # Execute registered actions
        for action in self.warning_actions:
            try:
                if asyncio.iscoroutinefunction(action):
                    await action()
                else:
                    action()
            except Exception as e:
                logger.error(f"Warning action failed: {e}")

    @staticmethod
    def optimize_tensor_memory():
        """Static optimization: Use efficient tensor storage."""
        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        logger.info("Tensor memory optimizations enabled")

    @staticmethod
    def clear_unused_memory():
        """Clear unused memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

    @staticmethod
    def get_memory_report() -> str:
        """Get formatted memory report."""
        if not torch.cuda.is_available():
            return "CUDA not available"

        stats = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3

            stats.append(
                f"GPU {i} ({props.name}): "
                f"{allocated:.2f}GB / {total:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved"
            )

        return "\n".join(stats)
