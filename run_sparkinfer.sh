# !bin bash

CUDA_VISIBLE_DEVICE=1 ./build/bin/llama-speculative \
        -m /share/models/prosparse-7b-gguf-w-our-predictor/prosparse-7b.gguf \
        -md /share/models/Llama-160M-Chat-v1-gguf/Llama-160M-Chat-v1.F16.gguf \
        -c 0 -co -ngl 99 -ngld 99 -fa \
        --draft-max 7 --draft-min 3 --draft-p-min 0.0 \
        -p "# Dijkstra's shortest path algorithm in Python (4 spaces indentation) + complexity analysis:\n\n" \
        --n-predict 512 --seed 42 \
        --sampling-seq k \
        --top-k 4 --temp 0.0 -np 4