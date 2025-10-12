GGML_SCHED_DEBUG=true CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-simple -m /share/models/prosparse-7b-gguf-w-our-predictor/prosparse-7b.gguf
# --vram-budget 12


# CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-simple -m /share/models/PowerInfer-prosparse-llama-2-7b-gguf/rename_prosparse-llama-2-7b.gguf "hello,world!" 

