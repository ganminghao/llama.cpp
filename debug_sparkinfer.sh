#!/bin/bash

env CUDA_VISIBLE_DEVICES=0 SPIF_PARALLEL=ON SPIF_RELOAD=ON \
    gdb --args ./build/bin/llama-cli -m ../prosparse-llama-2-7b.gguf \
    -spif-ms ../prosparse-llama-2-7b-sparkinfer-model-split.gguf \
    -ngl 999 -cffn --no-mmap -p "I believe the meaning of life is" \
    --samplers "temperature;top_p" --temp 0.8 --top-p 0.95 \
    -n 64 -no-cnv -t 1 -s 42 --no-warmup
