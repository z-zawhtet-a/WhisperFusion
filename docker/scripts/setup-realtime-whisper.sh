#!/bin/bash -e

## Clone this repo and install requirements
[ -d "RealtimeWhisper" ] || git clone -b asr-ws https://github.com/z-zawhtet-a/WhisperFusion.git RealtimeWhisper

cd RealtimeWhisper
apt update
apt install ffmpeg portaudio19-dev -y

## Install torchaudio matching the PyTorch from the base image
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torchaudio==2.1.2

## Install all the other dependencies normally
pip install -r requirements.txt

