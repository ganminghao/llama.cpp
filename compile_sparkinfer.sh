#!/bin/bash

release_dir=build_rel
debug_dir=build

cmake_opts=(
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_CUDA=ON
    -DGGML_CUDA_GRAPHS=OFF
    -DCMAKE_CUDA_ARCHITECTURES="89-real"
)

usage() {
    echo "usage: $0 [release|debug] [nvtx]"
    exit 1
}

mode=${1-}
nvtx_flag=${2-}

case "$mode" in
release)
    build_dir="$release_dir"
    build_type=Release
    ;;
debug)
    build_dir="$debug_dir"
    build_type=Debug
    cmake_opts+=(-DGGML_CUDA_DEBUG=ON)
    ;;
*)
    usage
    ;;
esac

if [[ -z ${nvtx_flag-} ]]; then
    :
elif [[ $nvtx_flag == "nvtx" ]]; then
    cmake_opts+=(
        -DCMAKE_C_FLAGS="-DUSE_NVTX -I/usr/local/cuda/include"
        -DCMAKE_CXX_FLAGS="-DUSE_NVTX -I/usr/local/cuda/include"
        -DCMAKE_CUDA_FLAGS="-DUSE_NVTX -I/usr/local/cuda/include"
    )
else
    usage
fi

cmake -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type" "${cmake_opts[@]}"
cmake --build "$build_dir" --config "$build_type" -j"$(nproc)" --target llama-cli llama-speculative spif-bench
