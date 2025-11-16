#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model="../prosparse-llama-2-7b.gguf"
model_split="../prosparse-llama-2-7b-sparkinfer-model-split.gguf"
prompt="Bubble sort algorithm in python:"

inference_opts=(
    -m $model
    -spif-ms $model_split
    -cffn --no-mmap -ngl 999 -t 4
    --samplers "temperature;top_p" --temp 0.8 --top-p 0.95 -s 1234
    -p "$prompt" -n 128 -no-cnv
    --no-warmup
)

release_bin="./build_rel/bin/llama-cli"
debug_bin="./build/bin/llama-cli"

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
