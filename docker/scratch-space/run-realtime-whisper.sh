#!/bin/bash -e

test -f /etc/shinit_v2 && source /etc/shinit_v2

cd /root/RealtimeWhisper

echo "Running RealtimeWhisper..."
exec python3 run_server.py -p 8080 -trt /root/scratch-space/models/whisper_model

