#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

draft_model="../Llama-160M-Chat-v1.gguf"
model="../prosparse-llama-2-7b.gguf"
model_split="../prosparse-llama-2-7b-sparkinfer-model-split.gguf"
prompt="Bubble sort algorithm in python:"

inference_opts=(
    -md $draft_model
    -m $model
    -spif-ms $model_split
    -cffn --no-mmap -ngld 999 -ngl 999 -t 4 -co
    --samplers "temperature;top_p" --temp 0.8 --top-p 0.95 -s 512
    --draft-min 3 --draft-max 5 --draft-p-min 0.5
    -p "$prompt" -n 128
)

release_bin="./build_rel/bin/llama-speculative"
debug_bin="./build/bin/llama-speculative"

if [[ ${1-} == "release" ]]; then
    bin="$release_bin"
elif [[ ${1-} == "debug" ]]; then
    bin="$debug_bin"
else
    echo "usage: $0 [release|debug] [nvtx]"
    echo "example:"
    echo "  $0 debug"
    echo "  $0 release nvtx"
    exit 1
fi

if [[ ${2-} == "nvtx" ]]; then
    nsys profile --trace=cuda,nvtx "$bin" "${inference_opts[@]}"
else
    "$bin" "${inference_opts[@]}"
fi
