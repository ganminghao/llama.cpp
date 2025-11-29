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
# draft_model="/share/models/sparkinfer-sharing/Llama-160M-Chat-v1.gguf"
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
    echo "usage: $0 [cli|speculative] [cuda]"
    exit 1
}

kind=${1-}
cuda_flag=${2-}

case "$kind" in
cli)
    bin="./build/bin/llama-cli"
    inference_opts=("${cli_opts[@]}" "${common_opts[@]}")
    ;;
speculative)
    bin="./build/bin/llama-speculative"
    inference_opts=("${speculative_opts[@]}" "${common_opts[@]}")
    ;;
*)
    usage
    ;;
esac

if [[ -z ${cuda_flag-} ]]; then
    dbg=gdb
elif [[ $cuda_flag == "cuda" ]]; then
    dbg=cuda-gdb
else
    usage
fi

$dbg --args "$bin" "${inference_opts[@]}"
