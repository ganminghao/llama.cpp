# !bin bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j4
