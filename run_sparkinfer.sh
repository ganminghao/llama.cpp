#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# export SPIF_SPLIT_DEBUG=ON
# export SPIF_DFR_DEBUG=1
export SPIF_REORDER=ON
export SPIF_PARALLEL=ON
export SPIF_RELOAD=ON
export SPIF_DFR_EMA=ON
export SPIF_INIT_DFR_DECAY=67
export SPIF_DX_DFR_DECAY=50
export SPIF_RELOAD_WINDOW_SIZE=4

draft_model="/share/models/sparkinfer-models/llama-160m-chat.gguf"
model="/share/models/sparkinfer-models/prosparse-llama-2-7b.gguf"
model_split="/share/models/sparkinfer-models/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf"
prompt="Bubble sort algorithm in python:"

vram_budget=12
threads=8
seed=1234
ctx_size=1024
max_tokens=128

common_opts=(
    -spif-ms "$model_split"
    -cffn -fit off -ngl all
    --no-mmap --no-direct-io
    -vb "$vram_budget"
    -t "$threads"
    -s "$seed"
    -p "$prompt"
    -c "$ctx_size"
    -n "$max_tokens"
)

completion_opts=(
    -m "$model" -no-cnv
    --repeat-penalty 1.05
)

speculative_opts=(
    -md "$draft_model" -ngld all -m "$model"
    --draft-min 3 --draft-max 5
    --repeat-penalty 1.1
)

usage() {
    echo "usage: $0 <release|debug> <completion|speculative> [nvtx|gdb|cugdb]"
    exit 1
}

mode=${1-}
kind=${2-}
nvtx_flag=0
gdb_flag=0
cugdb_flag=0

extra_args=("${@:3}")
for arg in "${extra_args[@]}"; do
    case "$arg" in
    nvtx)
        nvtx_flag=1
        ;;
    gdb)
        gdb_flag=1
        ;;
    cugdb)
        cugdb_flag=1
        ;;
    "") ;;
    *)
        usage
        ;;
    esac
done

if ((nvtx_flag && (gdb_flag || cugdb_flag))); then
    usage
fi

if ((gdb_flag && cugdb_flag)); then
    usage
fi

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
completion)
    bin="$bin_dir/llama-completion"
    inference_opts=("${completion_opts[@]}" "${common_opts[@]}")
    ;;
speculative)
    bin="$bin_dir/llama-speculative"
    inference_opts=("${speculative_opts[@]}" "${common_opts[@]}")
    ;;
*)
    usage
    ;;
esac

if ((gdb_flag || cugdb_flag)); then
    if ((!cugdb_flag)); then
        dbg=gdb
    else
        dbg=cuda-gdb
    fi
    $dbg --args "$bin" "${inference_opts[@]}"
elif ((!nvtx_flag)); then
    "$bin" "${inference_opts[@]}"
else
    nsys profile --trace=cuda,nvtx "$bin" "${inference_opts[@]}"
fi
