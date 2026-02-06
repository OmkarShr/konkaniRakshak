#!/bin/bash
# View logs from Docker services

if [ "$1" == "stt" ] || [ "$1" == "stt-service" ]; then
    docker compose logs -f stt-service
elif [ "$1" == "pipeline" ]; then
    docker compose logs -f pipeline
else
    echo "Viewing logs for all services..."
    echo "Usage: ./docker-logs.sh [stt|pipeline]"
    echo ""
    docker compose logs -f
fi
