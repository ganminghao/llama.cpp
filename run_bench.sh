#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# ---------------- SPIF Options ----------------
export SPIF_DFR_EMA=ON
export SPIF_DX_DFR_DECAY=51
export SPIF_RELOAD_WINDOW_SIZE=4

export SPIF_REORDER=ON
export SPIF_PARALLEL=ON
export SPIF_RELOAD=ON
export SPIF_FAST_AXPY=ON

# ---------------- Model Paths -----------------
model="/root/autodl-tmp/models/sparkinfer/Bamboo-7b.gguf"
model_split="/root/autodl-tmp/models/sparkinfer/bamboo-7b-sparkinfer-model-split-896.gguf"

model="/root/autodl-tmp/models/sparkinfer/opt-6.7b.gguf"
model_split="/root/autodl-tmp/models/sparkinfer/opt-6.7b-sparkinfer-model-split-1024.gguf"

model="/root/autodl-tmp/models/sparkinfer/SparseQwen2-7b.gguf"
model_split="/root/autodl-tmp/models/sparkinfer/SparseQwen2-7B-sparkinfer-model-split-592.gguf"

model="/root/autodl-tmp/models/sparkinfer/prosparse-llama-7b.gguf"
model_split="/root/autodl-tmp/models/sparkinfer/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf"


# ---------------- Inference Settings -----------------
vram_budget=7
threads=12
seed=1234
ctx_size=1024
max_tokens=128
file="/root/autodl-tmp/dataset/ShareGPT/processed_prompts.txt"
n_prompts=20

common_opts=(
    -m "$model" -ngl 999 -no-cnv
    -spif-ms "$model_split"
    -cffn --no-mmap
    --repeat-penalty 1.1

    -vb "$vram_budget"
    -t "$threads"
    -s "$seed"
    -c "$ctx_size"
    -n "$max_tokens"
    --file "$file"
    --n-test-prompts "$n_prompts"
)

usage() {
    echo "usage: $0 [release|debug] [cli] [nvtx]"
    exit 1
}

mode=${1-}
kind=${2-}
nvtx_flag=${3-}

case "$mode" in
    release) bin_dir="./build_rel/bin" ;;
    debug)   bin_dir="./build/bin" ;;
    *) usage ;;
esac

case "$kind" in
    cli)
        bin="$bin_dir/spif-bench"
        inference_opts=("${common_opts[@]}")
        ;;
    *)
        usage
        ;;
esac

# ---------------- Run -----------------
if [[ -z ${nvtx_flag-} ]]; then
    "$bin" "${inference_opts[@]}"
elif [[ $nvtx_flag == "nvtx" ]]; then
    nsys profile --trace=cuda,nvtx "$bin" "${inference_opts[@]}"
else
    usage
fi
