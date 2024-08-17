FROM nvcr.io/nvidia/tritonserver:24.05-py3
LABEL maintainer="NVIDIA"
LABEL repository="tritonserver"

RUN apt update && apt-get install -y ffmpeg
# COPY tensorrt_llm-0.11.0.dev2024052800-cp310-cp310-linux_x86_64.whl /workspace/tensorrt_llm-0.11.0.dev2024052800-cp310-cp310-linux_x86_64.whl
# RUN python3 -m pip install /workspace/tensorrt_llm-0.11.0.dev2024052800-cp310-cp310-linux_x86_64.whl
# RUN rm /workspace/tensorrt_llm-0.11.0.dev2024052800-cp310-cp310-linux_x86_64.whl
RUN python3 -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-llm==0.11.0.dev2024052800
RUN python3 -m pip install mpmath==1.3.0 tritonclient[all]

COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace
RUN python3 -m pip install -r requirements.txt

RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git && \
    cd TensorRT-LLM && \
    git checkout v0.11.0

COPY whisper_live /workspace/whisper_live
COPY run_server.py /workspace/run_server.py
COPY assets /workspace/assets
