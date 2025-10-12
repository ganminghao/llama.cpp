# !bin bash
# run this one time 
# pip install -e sparkinfer-py
# pip install -e gguf-py
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_DEBUG=ON -DGGML_CUDA_FORCE_CUBLAS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -DSPARKINFER_SCHEDULE_DEBUG=ON
cmake --build build --config Release  --target llama-simple llama-speculative -j12
