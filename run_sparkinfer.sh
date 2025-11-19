#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export GPU_ONLY=8
export DFR_DECAY=69

export SPIF_REORDER=ON
export SPIF_PARALLEL=ON
export SPIF_RELOAD=ON
# export FAST_AXPY=ON

draft_model="../Llama-160M-Chat-v1.gguf"
model="../prosparse-llama-2-7b.gguf"
model_split="../prosparse-llama-2-7b-sparkinfer-model-split.gguf"
prompt="Bubble sort algorithm in python:"

threads=4
seed=1234
max_tokens=128

common_opts=(
    -spif-ms "$model_split"
    -cffn --no-mmap
    -t "$threads"
    -s "$seed"
    -p "$prompt"
    -n "$max_tokens"
)

cli_opts=(
    -m "$model" -ngl 999
)

speculative_opts=(
    -md "$draft_model" -m "$model"
    -ngld 999 -ngl 999
    -co --draft-min 3 --draft-max 5
)

usage() {
    echo "usage: $0 [release|debug] [cli|speculative] [nvtx]"
    exit 1
}

mode=${1-}
kind=${2-}
nvtx_flag=${3-}

case "$mode" in
release)
    bin_dir="./build_rel/bin"
    ;;
debug)
    bin_dir="./build/bin"
    ;;
*)
    usage
    ;;
esac

case "$kind" in
cli)
    bin="$bin_dir/llama-cli"
    inference_opts=("${cli_opts[@]}" "${common_opts[@]}")
    ;;
speculative)
    bin="$bin_dir/llama-speculative"
    inference_opts=("${speculative_opts[@]}" "${common_opts[@]}")
    ;;
*)
    usage
    ;;
esac

if [[ -z ${nvtx_flag-} ]]; then
    "$bin" "${inference_opts[@]}"
elif [[ $nvtx_flag == "nvtx" ]]; then
    nsys profile --trace=cuda,nvtx "$bin" "${inference_opts[@]}"
else
    usage
fi
