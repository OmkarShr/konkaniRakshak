"""Graceful Shutdown Handler

Handles SIGINT, SIGTERM for clean shutdown of pipeline.
"""

import signal
import asyncio
from typing import Callable, List
from loguru import logger


class GracefulShutdown:
    """Manages graceful shutdown of async services."""

    def __init__(self):
        """Initialize shutdown handler."""
        self._shutdown_event = asyncio.Event()
        self._cleanup_tasks: List[Callable] = []
        self._is_shutting_down = False

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("GracefulShutdown initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        if self._is_shutting_down:
            logger.warning("Forced shutdown requested")
            return

        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self._is_shutting_down = True
        self._shutdown_event.set()

    def add_cleanup_task(self, task: Callable) -> None:
        """Add cleanup task to run on shutdown.

        Args:
            task: Async or sync callable to run during shutdown
        """
        self._cleanup_tasks.append(task)
        logger.debug(f"Added cleanup task: {task.__name__}")

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    async def shutdown(self) -> None:
        """Execute graceful shutdown."""
        if not self._is_shutting_down:
            logger.info("Initiating manual shutdown...")
            self._is_shutting_down = True

        logger.info(f"Running {len(self._cleanup_tasks)} cleanup tasks...")

        for i, task in enumerate(self._cleanup_tasks, 1):
            try:
                logger.info(
                    f"[{i}/{len(self._cleanup_tasks)}] Cleaning up: {task.__name__}..."
                )

                if asyncio.iscoroutinefunction(task):
                    await task()
                else:
                    task()

                logger.info(f"✓ Cleanup completed: {task.__name__}")
            except Exception as e:
                logger.error(f"✗ Cleanup failed for {task.__name__}: {e}")

        logger.info("Graceful shutdown complete")

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._is_shutting_down


# Global shutdown handler
shutdown_handler = GracefulShutdown()
