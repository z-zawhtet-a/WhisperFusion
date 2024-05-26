# RTWhisper Server

Whisper is optimized to run efficiently as TensorRT engine, maximizing
performance and real-time processing capabilities.

## Features

- **Real-Time Speech-to-Text**: Utilizes OpenAI WhisperLive to convert
  spoken language into text in real-time.
- **TensorRT Optimization**: Whisper models are optimized to
  run as TensorRT engines, ensuring high-performance and low-latency
  processing.

## Hardware Requirements

- A GPU with at least 12GB of RAM
- For optimal latency, the GPU should have a similar FP16 (half) TFLOPS as the RTX 3090.

## Getting Started
We provide a Docker Compose setup to streamline the deployment of the pre-built TensorRT-LLM docker container.

- Build and Run with docker compose for RTX 3090 and RTX 4090
```bash
mkdir -p docker/scratch-space/models
mv <path-to-whisper-tensorrt engine> docker/scratch-space/models/whisper_model

cd docker && docker build . -t realtime-whisper:latest && cd ..
```
- Create a .env file
```
VALID_API_KEYS=your_api_key_1,your_api_key_2
```
- Run the server
```bash
docker compose --env-file .env up
```

- Connect frontend to `ws://localhost:8080`


## References
[WhisperLive](https://github.com/collabora/WhisperLive) and
[WhisperSpeech](https://github.com/collabora/WhisperFusion)
