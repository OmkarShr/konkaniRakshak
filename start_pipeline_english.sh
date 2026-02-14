#!/bin/bash
#
# Start script for English-only pipeline
#

export WS_PORT=8767
exec python /app/ws_pipeline_english.py
