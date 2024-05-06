#!/bin/bash -e

## Change working dir to the [whisper example dir](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper) in TensorRT-LLM.
cd /root/TensorRT-LLM-examples/whisper

## Currently, by default TensorRT-LLM only supports `large-v2` and `large-v3`. In this repo, we use `small.en`.
## Download the required assets

if [ ! -f assets/mel_filters.npz ]; then
    echo "Downloading mel filter definitions"
    wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz > /dev/null 2>&1
else
    echo "Mel filter definitions already exist, skipping download."
fi

# the small.en model weights
# if [ ! -f assets/small.en.pt ]; then
#     echo "Downloading PyTorch weights for small.en model"
#     wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt > /dev/null 2>&1
# else
#     echo "PyTorch weights for small.en model already exist, skipping download."
# fi

echo "Building Whisper TensorRT Engine..."
pip install -r requirements.txt > /dev/null 2>&1

python3 build.py --output_dir distil_whisper_medium_th --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin  --use_bert_attention_plugin --model_name distil-whisper-medium-th > /dev/null 2>&1

mkdir -p /root/scratch-space/models
cp -r distil_whisper_medium_th /root/scratch-space/models
