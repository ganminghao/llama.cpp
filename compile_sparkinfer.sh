#!/bin/bash

if ! dpkg -s libssl-dev >/dev/null 2>&1; then
    apt update >/dev/null 2>&1
    apt install -y libssl-dev >/dev/null 2>&1
fi

release_dir=build_rel
debug_dir=build

cmake_opts=(
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_CUDA=ON
    -DGGML_CUDA_GRAPHS=OFF
    -DCMAKE_CUDA_ARCHITECTURES=native
)

usage() {
    echo "usage: $0 <release|debug> [nvtx] [clean]"
    exit 1
}

mode=${1-}
nvtx_flag=0
clean_flag=0

extra_args=("${@:2}")
for arg in "${extra_args[@]}"; do
    case "$arg" in
    nvtx)
        nvtx_flag=1
        ;;
    clean)
        clean_flag=1
        ;;
    "") ;;
    *)
        usage
        ;;
    esac
done

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

if ((clean_flag)); then
    rm -rf "$build_dir"
fi

extra_flags=()
((nvtx_flag)) && extra_flags+=(-DUSE_NVTX -I/usr/local/cuda/include)

if ((${#extra_flags[@]} > 0)); then
    joined="${extra_flags[*]}"
    cmake_opts+=(
        -DCMAKE_C_FLAGS="$joined"
        -DCMAKE_CXX_FLAGS="$joined"
        -DCMAKE_CUDA_FLAGS="$joined"
    )
fi

cmake -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type" "${cmake_opts[@]}"
cmake --build "$build_dir" --config "$build_type" -j"$(nproc)" \
    --target llama-completion llama-speculative llama-quantize
