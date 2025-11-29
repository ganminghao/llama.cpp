#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# export SPIF_SPLIT_DEBUG=ON
# export SPIF_DFR_EMA=ON
export SPIF_DX_DFR_DECAY=20
export SPIF_RELOAD_WINDOW_SIZE=4

export SPIF_REORDER=ON
export SPIF_PARALLEL=ON
export SPIF_RELOAD=ON
export SPIF_FAST_AXPY=ON

draft_model="/share/models/sparkinfer-sharing/opt-125m.gguf"
model="/share/models/sparkinfer-sharing/opt-6.7b.gguf"
# model="/share/models/sparkinfer-sharing/opt-6.7b-q8_0.gguf"
model_split="/share/models/sparkinfer-sharing/opt-6.7b-sparkinfer-model-split-1024.gguf"
# draft_model="/share/models/sparkinfer-sharing/llama-160m-chat.gguf"
# model="/share/models/sparkinfer-sharing/prosparse-llama-2-7b.gguf"
# model_split="/share/models/sparkinfer-sharing/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf"
# draft_model="/share/models/sparkinfer-sharing/lite-mistral-150m.gguf"
# model="/share/models/sparkinfer-sharing/bamboo-7b.gguf"
# model_split="/share/models/sparkinfer-sharing/bamboo-7b-sparkinfer-model-split-896.gguf"
# draft_model="/share/models/sparkinfer-sharing/Qwen2-0.5B.gguf"
# model="/share/models/sparkinfer-sharing/SparseQwen2-7B.gguf"
# model_split="/share/models/sparkinfer-sharing/SparseQwen2-7B-sparkinfer-model-split-592.gguf"
prompt="Bubble sort algorithm in python:"

vram_budget=10
threads=4
seed=1234
ctx_size=1024
max_tokens=128

common_opts=(
    -spif-ms "$model_split"
    -cffn --no-mmap
    -vb "$vram_budget"
    -t "$threads"
    -s "$seed"
    -p "$prompt"
    -c "$ctx_size"
    -n "$max_tokens"
)

cli_opts=(
    -m "$model" -ngl 999 -no-cnv
)

speculative_opts=(
    -md "$draft_model" -m "$model"
    -ngld 999 -ngl 999 -kvu
    -co --draft-min 3 --draft-max 5
    # --repeat-penalty 1.1
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
