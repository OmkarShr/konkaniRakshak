"""Simple Web Dashboard for Pipeline Monitoring

Provides HTTP endpoint to view pipeline status and metrics.
"""

from flask import Flask, jsonify, render_template_string
from loguru import logger
import time

app = Flask(__name__)

# HTML Dashboard Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Konkani Agent Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
        .metric-label { font-size: 12px; color: #7f8c8d; }
        .status-online { color: #27ae60; }
        .status-offline { color: #e74c3c; }
        .latency-bar { background: #ecf0f1; height: 20px; border-radius: 10px; overflow: hidden; margin: 5px 0; }
        .latency-fill { background: #3498db; height: 100%; border-radius: 10px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #34495e; color: white; }
        .voice-selector { margin: 10px 0; }
        .voice-btn { padding: 8px 16px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
        .voice-btn.active { background: #3498db; color: white; }
        .voice-btn:hover { background: #2980b9; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ Konkani Conversational AI Agent</h1>
            <p>Real-time Voice Pipeline for FIR Filing Assistance</p>
        </div>
        
        <div class="card">
            <h2>System Status</h2>
            <div class="metric">
                <div class="metric-value {{ 'status-online' if stt_status else 'status-offline' }}">
                    {{ '● Online' if stt_status else '● Offline' }}
                </div>
                <div class="metric-label">STT Service</div>
            </div>
            <div class="metric">
                <div class="metric-value {{ 'status-online' if pipeline_status else 'status-offline' }}">
                    {{ '● Online' if pipeline_status else '● Offline' }}
                </div>
                <div class="metric-label">Pipeline</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ session_duration }}</div>
                <div class="metric-label">Session Duration</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Performance Metrics</h2>
            <p>Average Latencies (last 100 requests):</p>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Average Latency</th>
                    <th>Visual</th>
                </tr>
                <tr>
                    <td>VAD Detection</td>
                    <td>{{ metrics.get('avg_vad_latency_ms', 0)|round(1) }} ms</td>
                    <td>
                        <div class="latency-bar">
                            <div class="latency-fill" style="width: {{ min(metrics.get('avg_vad_latency_ms', 0) / 2, 100) }}%"></div>
                        </div>
                    </td>
                </tr>
                <tr>
                    <td>STT Transcription</td>
                    <td>{{ metrics.get('avg_stt_latency_ms', 0)|round(1) }} ms</td>
                    <td>
                        <div class="latency-bar">
                            <div class="latency-fill" style="width: {{ min(metrics.get('avg_stt_latency_ms', 0) / 5, 100) }}%"></div>
                        </div>
                    </td>
                </tr>
                <tr>
                    <td>LLM Response</td>
                    <td>{{ metrics.get('avg_llm_latency_ms', 0)|round(1) }} ms</td>
                    <td>
                        <div class="latency-bar">
                            <div class="latency-fill" style="width: {{ min(metrics.get('avg_llm_latency_ms', 0) / 5, 100) }}%"></div>
                        </div>
                    </td>
                </tr>
                <tr>
                    <td>TTS Synthesis</td>
                    <td>{{ metrics.get('avg_tts_latency_ms', 0)|round(1) }} ms</td>
                    <td>
                        <div class="latency-bar">
                            <div class="latency-fill" style="width: {{ min(metrics.get('avg_tts_latency_ms', 0) / 7, 100) }}%"></div>
                        </div>
                    </td>
                </tr>
                <tr style="background: #ecf0f1; font-weight: bold;">
                    <td>Total Latency</td>
                    <td>{{ metrics.get('avg_total_latency_ms', 0)|round(1) }} ms</td>
                    <td>
                        <div class="latency-bar">
                            <div class="latency-fill" style="width: {{ min(metrics.get('avg_total_latency_ms', 0) / 15, 100) }}%; background: #27ae60;"></div>
                        </div>
                    </td>
                </tr>
            </table>
        </div>
        
        <div class="card">
            <h2>Request Statistics</h2>
            <div class="metric">
                <div class="metric-value">{{ metrics.get('total_requests', 0) }}</div>
                <div class="metric-label">Total Requests</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: #27ae60;">{{ metrics.get('successful', 0) }}</div>
                <div class="metric-label">Successful</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: #e74c3c;">{{ metrics.get('failed', 0) }}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ metrics.get('success_rate', 0)|round(1) }}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Voice Configuration</h2>
            <p>Current Voice: <strong>{{ current_voice }}</strong></p>
            <div class="voice-selector">
                {% for voice, desc in voices.items() %}
                <button class="voice-btn {{ 'active' if voice == current_voice else '' }}" 
                        onclick="changeVoice('{{ voice }}')">
                    {{ voice.replace('_', ' ').title() }}
                </button>
                {% endfor %}
            </div>
            <p><small>Click to change voice profile (requires restart)</small></p>
        </div>
        
        <div class="card">
            <h2>System Info</h2>
            <p><strong>Hardware:</strong> {{ gpu_info }}</p>
            <p><strong>TTS Model:</strong> AI4Bharat Indic-Parler-TTS (~600MB)</p>
            <p><strong>STT Model:</strong> IndicConformer Kok (499MB)</p>
            <p><strong>LLM:</strong> Gemini 2.5 Flash (Cloud)</p>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 5 seconds
        setInterval(function() {
            window.location.reload();
        }, 5000);
        
        function changeVoice(voice) {
            fetch('/api/voice/' + voice, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Voice changed to: ' + voice);
                        window.location.reload();
                    }
                });
        }
    </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    """Render monitoring dashboard."""
    try:
        from ..utils.monitor import monitor
        from ..utils.voice_manager import voice_manager

        metrics = monitor.get_statistics()

        return render_template_string(
            DASHBOARD_HTML,
            stt_status=True,  # TODO: Check actual status
            pipeline_status=True,
            session_duration="Active",
            metrics=metrics,
            current_voice=voice_manager.current_voice,
            voices=voice_manager.list_voices(),
            gpu_info="2x RTX Ada 4000 (20GB)",  # TODO: Detect actual GPUs
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics")
def api_metrics():
    """API endpoint for metrics."""
    try:
        from ..utils.monitor import monitor

        return jsonify(monitor.get_statistics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/voice/<voice_name>", methods=["POST"])
def change_voice(voice_name):
    """Change voice profile."""
    try:
        from ..utils.voice_manager import voice_manager

        voice_manager.set_voice(voice_name)
        return jsonify({"success": True, "voice": voice_name})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


def start_dashboard(port: int = 8080):
    """Start the monitoring dashboard.

    Args:
        port: Port to run dashboard on
    """
    logger.info(f"Starting monitoring dashboard on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    start_dashboard()
