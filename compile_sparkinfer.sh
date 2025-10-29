#!/bin/bash

release_dir=build_rel
debug_dir=build

cmake_opts=(
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_CUDA=ON
    -DGGML_CUDA_GRAPHS=OFF
)

if [[ ${2-} == "nvtx" ]]; then
    cmake_opts+=(
        -DCMAKE_C_FLAGS="-DUSE_NVTX -I/usr/local/cuda/include"
        -DCMAKE_CXX_FLAGS="-DUSE_NVTX -I/usr/local/cuda/include"
        -DCMAKE_CUDA_FLAGS="-DUSE_NVTX -I/usr/local/cuda/include"
    )
fi

if [[ ${1-} == "release" ]]; then
    cmake -B "$release_dir" -DCMAKE_BUILD_TYPE=Release "${cmake_opts[@]}"
    cmake --build "$release_dir" --config Release -j$(nproc) --target llama-cli

elif [[ ${1-} == "debug" ]]; then
    cmake_opts+=(-DGGML_CUDA_DEBUG=ON)
    cmake -B "$debug_dir" -DCMAKE_BUILD_TYPE=Debug "${cmake_opts[@]}"
    cmake --build "$debug_dir" --config Debug -j$(nproc) --target llama-cli

else
    echo "usage: $0 [release|debug] [nvtx]"
    echo "example:"
    echo "  $0 debug"
    echo "  $0 release nvtx"
    exit 1
fi
