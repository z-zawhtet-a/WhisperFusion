#!/bin/bash -e

cd /workspace

echo "Running RealtimeWhisper..."
exec python3 run_server.py -p 8080 -trt /root/scratch-space/models/whisper_model -m -omp 8

