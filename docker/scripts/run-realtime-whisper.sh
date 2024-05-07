#!/bin/bash -e

test -f /etc/shinit_v2 && source /etc/shinit_v2

# echo "Running build-models.sh..."
# cd /root/scratch-space/
# ./build-models.sh

cd /root/RealtimeWhisper

echo "Running RealtimeWhisper..."
exec python3 main.py --whisper_tensorrt_path /root/scratch-space/models/whisper_model

