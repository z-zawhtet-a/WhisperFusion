# RTWhisper Server

Whisper is optimized to run efficiently as a TensorRT engine, maximizing performance and real-time processing capabilities.

## Features

- **Real-Time Speech-to-Text**: Utilizes OpenAI WhisperLive to convert spoken language into text in real-time.
- **TensorRT Optimization**: Whisper models are optimized to run as TensorRT engines, ensuring high-performance and low-latency processing.

## Hardware Requirements

- A GPU with at least 12GB of RAM
- For optimal latency, the GPU should have a similar FP16 (half) TFLOPS as the RTX 3090.

## Getting Started

We provide a Docker Compose setup to streamline the deployment of the pre-built TensorRT-LLM docker container.

### 1. Build the Docker image

```bash
docker build . -t whisper-realtime-trt:0.11.0
```

### 2. Build the TensorRT engine

a. Create the model directory:
```bash
mkdir -p docker/scratch-space/models/whisper_model
```

b. Add your Hugging Face Whisper model to `./docker/scratch-space/models/whisper_model`
   Example: Use `biodatlab/whisper-th-medium-combined`
   
   Folder structure should be:
   ```
   ./docker/scratch-space/models/whisper_model/whisper-th-medium-combined/
   ```

c. Build the TensorRT engine:
```bash
mount="./docker/scratch-space/models/whisper_model:/workspace/TensorRT-LLM/examples/whisper/whisper_model"
docker run -it --name "whisper-build" --gpus all --net host -v $mount --shm-size=2g whisper-realtime-trt:0.11.0

# Once you are inside container under /workspace 
cd TensorRT-LLM/examples/whisper

# Convert the huggingface model into openai model
python3 distil_whisper/convert_from_distil_whisper.py \
  --model_name ./whisper_model/whisper-th-medium-combined \
  --output_name medium

# Set the TRTLLM parameters
INFERENCE_PRECISION=float16
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8
checkpoint_dir=tllm_checkpoint
output_dir=whisper_model
```

- Convert the openai model into trtllm compatible checkpoint.
```bash
python3 convert_checkpoint.py \
                --model_name medium \
                --output_dir $checkpoint_dir
```

- Build the trtllm engines
```bash
trtllm-build --checkpoint_dir ${checkpoint_dir}/encoder \
                --output_dir ${output_dir}/encoder \
                --paged_kv_cache disable \
                --moe_plugin disable \
                --enable_xqa disable \
                --use_custom_all_reduce disable \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --gemm_plugin disable \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding disable

trtllm-build --checkpoint_dir ${checkpoint_dir}/decoder \
                --output_dir ${output_dir}/decoder \
                --paged_kv_cache disable \
                --moe_plugin disable \
                --enable_xqa disable \
                --use_custom_all_reduce disable \
                --max_beam_width ${MAX_BEAM_WIDTH} \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --max_output_len 100 \
                --max_input_len 14 \
                --max_encoder_input_len 1500 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding disable
```
The tensorrt engine will be saved in `./docker/scratch-space/models/whisper_model/`

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