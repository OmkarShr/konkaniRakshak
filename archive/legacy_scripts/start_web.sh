#!/bin/bash
# Quick start script for Konkani Voice Agent Web UI

cd /home/btech/Music/NagarRakshakKonkani/konkaniRakshak

echo "🎤 Konkani Voice Agent - Web Interface"
echo "========================================"
echo ""

# Check if backend is running
if ! curl -s http://localhost:8080/health > /dev/null; then
    echo "🚀 Starting backend server..."
    python3 web_backend.py > /tmp/web_backend.log 2>&1 &
    sleep 3
    
    if curl -s http://localhost:8080/health > /dev/null; then
        echo "✅ Backend server started on http://localhost:8080"
    else
        echo "❌ Failed to start backend server"
        echo "Check logs: tail -f /tmp/web_backend.log"
        exit 1
    fi
else
    echo "✅ Backend server already running"
fi

echo ""
echo "🌐 Web Interface:"
echo "   http://localhost:8080/web_ui/index.html"
echo ""
echo "📱 Or open the file directly:"
echo "   file:///home/btech/Music/NagarRakshakKonkani/konkaniRakshak/web_ui/index.html"
echo ""
echo "✨ Instructions:"
echo "   1. Open the web interface in your browser"
echo "   2. Click the microphone button"
echo "   3. Allow microphone access"
echo "   4. Speak in Konkani!"
echo "   5. The AI will respond with text (TTS coming soon)"
echo ""
echo "🛑 To stop the server:"
echo "   pkill -f web_backend.py"
echo ""

# Try to open browser
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:8080/web_ui/index.html 2>/dev/null &
    echo "🌐 Opening browser..."
fi

echo "✅ Ready! Open your browser to start chatting."
