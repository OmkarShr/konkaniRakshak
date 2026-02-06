"""Monitoring Dashboard

Real-time web dashboard for monitoring the Konkani Voice Agent.
Shows GPU usage, latency metrics, error rates, and conversation stats.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from loguru import logger

# Web framework
from aiohttp import web
import aiohttp_cors


@dataclass
class DashboardMetrics:
    """Real-time metrics for dashboard."""

    timestamp: float

    # GPU
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_utilization: float

    # Latency
    avg_latency_ms: float
    p95_latency_ms: float
    last_latency_ms: float

    # Errors
    error_count: int
    error_rate: float

    # Conversations
    conversation_count: int
    active_conversations: int

    # System
    uptime_seconds: float


class MonitoringDashboard:
    """
    Web dashboard for real-time monitoring.
    """

    def __init__(
        self,
        port: int = 8080,
        update_interval: float = 1.0,
        metrics_callback: Optional[callable] = None,
    ):
        self.port = port
        self.update_interval = update_interval
        self.metrics_callback = metrics_callback

        # Current metrics
        self.current_metrics: Optional[DashboardMetrics] = None
        self.metrics_history: list = []
        self.max_history = 300  # 5 minutes at 1s intervals

        # Server
        self.app = web.Application()
        self.runner = None
        self.site = None

        # Stats
        self.start_time = time.time()
        self.setup_routes()

        logger.info(f"Dashboard initialized on port {port}")

    def setup_routes(self):
        """Setup HTTP routes."""
        # CORS setup
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*",
                )
            },
        )

        # Static files
        self.app.router.add_get("/", self.index_handler)
        self.app.router.add_get("/api/metrics", self.metrics_handler)
        self.app.router.add_get("/api/history", self.history_handler)
        self.app.router.add_get("/api/status", self.status_handler)
        self.app.router.add_get("/ws", self.websocket_handler)

        # WebSocket clients
        self.websocket_clients = []

    async def index_handler(self, request):
        """Serve main dashboard page."""
        html = self._generate_dashboard_html()
        return web.Response(text=html, content_type="text/html")

    async def metrics_handler(self, request):
        """API endpoint for current metrics."""
        if self.current_metrics:
            return web.json_response(asdict(self.current_metrics))
        return web.json_response({"error": "No metrics available"}, status=503)

    async def history_handler(self, request):
        """API endpoint for metrics history."""
        history = [asdict(m) for m in self.metrics_history]
        return web.json_response({"history": history})

    async def status_handler(self, request):
        """API endpoint for system status."""
        uptime = time.time() - self.start_time
        return web.json_response(
            {
                "status": "running",
                "uptime_seconds": uptime,
                "metrics_available": self.current_metrics is not None,
                "websocket_clients": len(self.websocket_clients),
            }
        )

    async def websocket_handler(self, request):
        """WebSocket endpoint for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.websocket_clients.append(ws)
        logger.info(f"Dashboard client connected. Total: {len(self.websocket_clients)}")

        try:
            # Send initial metrics
            if self.current_metrics:
                await ws.send_json(asdict(self.current_metrics))

            # Keep connection alive
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    if msg.data == "ping":
                        await ws.send_str("pong")
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break

        finally:
            self.websocket_clients.remove(ws)
            logger.info(
                f"Dashboard client disconnected. Total: {len(self.websocket_clients)}"
            )

        return ws

    def _generate_dashboard_html(self) -> str:
        """Generate HTML for dashboard."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Konkani Voice Agent - Monitoring Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }

        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #00d4ff;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }

        .card {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }

        .card h2 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid #0f3460;
            padding-bottom: 10px;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: #0f3460;
            border-radius: 5px;
        }

        .metric-label {
            color: #aaa;
        }

        .metric-value {
            font-weight: bold;
            font-size: 1.1em;
        }

        .good {
            color: #4ade80;
        }

        .warning {
            color: #fbbf24;
        }

        .critical {
            color: #f87171;
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-online {
            background: #4ade80;
            box-shadow: 0 0 10px #4ade80;
        }

        .status-offline {
            background: #f87171;
        }

        #latency-chart, #gpu-chart {
            width: 100%;
            height: 200px;
            background: #0a0a0a;
            border-radius: 5px;
            margin-top: 10px;
        }

        .last-updated {
            text-align: center;
            margin-top: 20px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>
        <span class="status-indicator status-online"></span>
        Konkani Voice Agent - Monitoring Dashboard
    </h1>

    <div class="grid">
        <div class="card">
            <h2>System Status</h2>
            <div class="metric">
                <span class="metric-label">Status</span>
                <span class="metric-value good" id="status">Running</span>
            </div>
            <div class="metric">
                <span class="metric-label">Uptime</span>
                <span class="metric-value" id="uptime">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Active Conversations</span>
                <span class="metric-value" id="active-conv">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Conversations</span>
                <span class="metric-value" id="total-conv">0</span>
            </div>
        </div>

        <div class="card">
            <h2>GPU Memory</h2>
            <div class="metric">
                <span class="metric-label">Allocated</span>
                <span class="metric-value" id="gpu-allocated">-- MB</span>
            </div>
            <div class="metric">
                <span class="metric-label">Reserved</span>
                <span class="metric-value" id="gpu-reserved">-- MB</span>
            </div>
            <div class="metric">
                <span class="metric-label">Utilization</span>
                <span class="metric-value" id="gpu-util">--%</span>
            </div>
            <canvas id="gpu-chart"></canvas>
        </div>

        <div class="card">
            <h2>Latency Metrics</h2>
            <div class="metric">
                <span class="metric-label">Current</span>
                <span class="metric-value" id="latency-current">-- ms</span>
            </div>
            <div class="metric">
                <span class="metric-label">Average</span>
                <span class="metric-value" id="latency-avg">-- ms</span>
            </div>
            <div class="metric">
                <span class="metric-label">P95</span>
                <span class="metric-value" id="latency-p95">-- ms</span>
            </div>
            <canvas id="latency-chart"></canvas>
        </div>

        <div class="card">
            <h2>Error Tracking</h2>
            <div class="metric">
                <span class="metric-label">Total Errors</span>
                <span class="metric-value" id="error-count">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Error Rate</span>
                <span class="metric-value" id="error-rate">0%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Health</span>
                <span class="metric-value good" id="health">Good</span>
            </div>
        </div>
    </div>

    <div class="last-updated">
        Last updated: <span id="last-updated">--</span>
    </div>

    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        let metricsHistory = [];

        ws.onopen = function() {
            console.log('Connected to dashboard');
        };

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };

        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };

        ws.onclose = function() {
            console.log('Disconnected from dashboard');
            // Try to reconnect after 5 seconds
            setTimeout(() => {
                location.reload();
            }, 5000);
        };

        function updateDashboard(data) {
            // Update system status
            document.getElementById('uptime').textContent = formatTime(data.uptime_seconds);
            document.getElementById('active-conv').textContent = data.active_conversations;
            document.getElementById('total-conv').textContent = data.conversation_count;

            // Update GPU
            document.getElementById('gpu-allocated').textContent = data.gpu_allocated_mb.toFixed(1) + ' MB';
            document.getElementById('gpu-reserved').textContent = data.gpu_reserved_mb.toFixed(1) + ' MB';
            document.getElementById('gpu-util').textContent = data.gpu_utilization.toFixed(1) + '%';

            // Update latency
            document.getElementById('latency-current').textContent = data.last_latency_ms.toFixed(1) + ' ms';
            document.getElementById('latency-avg').textContent = data.avg_latency_ms.toFixed(1) + ' ms';
            document.getElementById('latency-p95').textContent = data.p95_latency_ms.toFixed(1) + ' ms';

            // Update errors
            document.getElementById('error-count').textContent = data.error_count;
            document.getElementById('error-rate').textContent = (data.error_rate * 100).toFixed(2) + '%';

            // Update health status
            const healthEl = document.getElementById('health');
            if (data.error_rate > 0.1 || data.gpu_utilization > 90) {
                healthEl.textContent = 'Warning';
                healthEl.className = 'metric-value warning';
            } else if (data.error_rate > 0.25 || data.gpu_utilization > 95) {
                healthEl.textContent = 'Critical';
                healthEl.className = 'metric-value critical';
            } else {
                healthEl.textContent = 'Good';
                healthEl.className = 'metric-value good';
            }

            // Update timestamp
            document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();

            // Keep history
            metricsHistory.push(data);
            if (metricsHistory.length > 300) {
                metricsHistory.shift();
            }

            // Update charts
            updateCharts();
        }

        function formatTime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            return `${hours}h ${minutes}m ${secs}s`;
        }

        function updateCharts() {
            // Simple canvas drawing for latency
            const latencyCanvas = document.getElementById('latency-chart');
            const ctx = latencyCanvas.getContext('2d');
            ctx.clearRect(0, 0, latencyCanvas.width, latencyCanvas.height);

            if (metricsHistory.length < 2) return;

            // Draw latency history
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 2;
            ctx.beginPath();

            const maxLatency = Math.max(...metricsHistory.map(m => m.last_latency_ms), 2000);
            const stepX = latencyCanvas.width / metricsHistory.length;

            metricsHistory.forEach((m, i) => {
                const x = i * stepX;
                const y = latencyCanvas.height - (m.last_latency_ms / maxLatency * latencyCanvas.height);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });

            ctx.stroke();

            // Draw target line
            ctx.strokeStyle = '#4ade80';
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            const targetY = latencyCanvas.height - (1000 / maxLatency * latencyCanvas.height);
            ctx.moveTo(0, targetY);
            ctx.lineTo(latencyCanvas.width, targetY);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    </script>
</body>
</html>
        """

    def update_metrics(self, metrics: DashboardMetrics):
        """Update current metrics and notify clients."""
        self.current_metrics = metrics

        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)

        # Broadcast to WebSocket clients
        asyncio.create_task(self._broadcast_metrics(metrics))

    async def _broadcast_metrics(self, metrics: DashboardMetrics):
        """Broadcast metrics to all connected clients."""
        if not self.websocket_clients:
            return

        message = json.dumps(asdict(metrics))
        disconnected = []

        for ws in self.websocket_clients:
            try:
                await ws.send_str(message)
            except:
                disconnected.append(ws)

        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_clients.remove(ws)

    async def start(self):
        """Start the dashboard server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await self.site.start()

        logger.info(f"✓ Dashboard running at http://0.0.0.0:{self.port}")

    async def stop(self):
        """Stop the dashboard server."""
        if self.runner:
            await self.runner.cleanup()
            logger.info("✓ Dashboard stopped")

    async def run_metric_collection(self, get_metrics_fn: callable):
        """Run metric collection loop."""
        while True:
            try:
                metrics = await get_metrics_fn()
                if metrics:
                    self.update_metrics(metrics)
            except Exception as e:
                logger.error(f"Metric collection error: {e}")

            await asyncio.sleep(self.update_interval)
