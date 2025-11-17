#include "axpy-sparse.cuh"
#include "common.cuh"
#include "ggml.h"

// template <typename T, typename type_acc>
// static __global__ void mul_mat_axpy_sparse(const void * __restrict__ vx,
//                                            const float * __restrict__ y,
//                                            float * __restrict__ dst,
//                                            const int       ncols,
//                                            const int       nrows,
//                                            const int32_t * gpu_neu_idx,
//                                            const float *   sparse_idx) {
//     const int blk_idx      = blockIdx.x;  // block index, range from [0,nrows]
//     const int thds_per_blk = blockDim.x;  // number of threads per block

//     const int neu = gpu_neu_idx ? gpu_neu_idx[blk_idx] : blk_idx;
//     const int tid = threadIdx.x;  // range from [0,31]

//     float alpha_fp32 = y[neu];

//     if (sparse_idx[neu] < 0.5f) {
//         return;
//     }

//     extern __shared__ float shared_dst[];  // TODO:dynamic

//     const int VALS_PER_ITER = 2;           // each iter compute 2 vals consequently, we should not modify this
//     const int iter_stride   = VALS_PER_ITER * thds_per_blk;

//     // partial sum for each thread
//     float tmp = 0.0f;
//     for (int i = 0; i < ncols; i += thds_per_blk) {
//         shared_dst[i + tid] = 0;
//     }
//     __syncthreads();

//     for (int i = 0; i < ncols; i += iter_stride) {
//         const int col  = i + VALS_PER_ITER * tid;
//         const int vx_i = blk_idx * ncols + col;  // vx index, vx was store in "blk_idx way", so indice it with blk_idx

//         float2 v;
//         if constexpr (std::is_same<T, half>::value) {
//             const half * x = reinterpret_cast<const half *>(vx);
//             v.x            = __half2float(x[vx_i + 0]);
//             v.y            = __half2float(x[vx_i + 1]);
//         } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
//             const __nv_bfloat16 * x = reinterpret_cast<const __nv_bfloat16 *>(vx);
//             v.x                     = __bfloat162float(x[vx_i + 0]);
//             v.y                     = __bfloat162float(x[vx_i + 1]);
//         } else if constexpr (std::is_same<T, float>::value) {
//             const float * x = reinterpret_cast<const float *>(vx);
//             v.x             = x[vx_i + 0];
//             v.y             = x[vx_i + 1];
//         } else {
//             static_assert(std::is_same<T, void>::value, "unsupported type for axpy_sparse");
//         }

//         // matrix multiplication, process 2 vals per j iter
//         tmp                 = v.x * alpha_fp32;
//         shared_dst[col]     = tmp;  // share_dst[col] = tmp
//         tmp                 = v.y * alpha_fp32;
//         shared_dst[col + 1] = tmp;  // share_dst[col+1] = tmp
//     }
//     __syncthreads();

//     for (int i = 0; i < ncols; i += thds_per_blk) {
//         atomicAdd(&dst[i + tid], shared_dst[i + tid]);
//     }
// }

// template <typename T, typename type_acc>
// static __global__ void mul_mat_axpy_sparse_batch(const void * __restrict__ vx,
//                                                  const float * __restrict__ y,
//                                                  float * __restrict__ dst,
//                                                  const int       ncols,
//                                                  const int       nrows,
//                                                  const int32_t * gpu_neu_idx,
//                                                  const float *   sparse_idx) {
//     const int blk_idx      = blockIdx.x;  // block index, range from [0, nrows]
//     const int token_idx    = blockIdx.y;  // parallel input index, range from [0, src1_ncols]
//     const int thds_per_blk = blockDim.x;  // number of threads per block

//     y += token_idx * nrows;
//     dst += token_idx * ncols;
//     sparse_idx += token_idx * nrows;

//     const int neu = gpu_neu_idx ? gpu_neu_idx[blk_idx] : blk_idx;
//     const int tid = threadIdx.x;  // range from [0,31]

//     float alpha_fp32 = y[neu];

//     if (sparse_idx[neu] < 0.5f) {
//         return;
//     }

//     extern __shared__ float shared_dst[];  // TODO:dynamic

//     const int VALS_PER_ITER = 2;           // each iter compute 2 vals consequently, we should not modify this
//     const int iter_stride   = VALS_PER_ITER * thds_per_blk;

//     // partial sum for each thread
//     float tmp = 0.0f;
//     for (int i = 0; i < ncols; i += thds_per_blk) {
//         shared_dst[i + tid] = 0;
//     }
//     __syncthreads();

//     for (int i = 0; i < ncols; i += iter_stride) {
//         const int col  = i + VALS_PER_ITER * tid;
//         const int vx_i = blk_idx * ncols + col;  // vx index, vx was store in "blk_idx way", so indice it with blk_idx

//         float2 v;
//         if constexpr (std::is_same<T, half>::value) {
//             const half * x = reinterpret_cast<const half *>(vx);
//             v.x            = __half2float(x[vx_i + 0]);
//             v.y            = __half2float(x[vx_i + 1]);
//         } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
//             const __nv_bfloat16 * x = reinterpret_cast<const __nv_bfloat16 *>(vx);
//             v.x                     = __bfloat162float(x[vx_i + 0]);
//             v.y                     = __bfloat162float(x[vx_i + 1]);
//         } else if constexpr (std::is_same<T, float>::value) {
//             const float * x = reinterpret_cast<const float *>(vx);
//             v.x             = x[vx_i + 0];
//             v.y             = x[vx_i + 1];
//         } else {
//             static_assert(std::is_same<T, void>::value, "unsupported type for axpy_sparse");
//         }

//         // matrix multiplication, process 2 vals per j iter
//         tmp                 = v.x * alpha_fp32;
//         shared_dst[col]     = tmp;  // share_dst[col] = tmp
//         tmp                 = v.y * alpha_fp32;
//         shared_dst[col + 1] = tmp;  // share_dst[col+1] = tmp
//     }
//     __syncthreads();

//     for (int i = 0; i < ncols; i += thds_per_blk) {
//         atomicAdd(&dst[i + tid], shared_dst[i + tid]);
//     }
// }

// template <typename T, typename type_acc>
// static void launch_mul_mat_axpy_cuda_sparse(const T *       x,
//                                             const float *   y,
//                                             const float *   sparse_idx,
//                                             const int32_t * gpu_neu_idx,
//                                             float *         dst,
//                                             const int64_t   ncols,
//                                             const int64_t   nrows,
//                                             const int64_t   src_ncols,
//                                             const int64_t   num_gpu_neurons,
//                                             cudaStream_t    stream) {
//     // vec_axpy
//     if (src_ncols == 1) {
//         // the lanucher for powerinfer kernel:
//         const dim3 block_nums(num_gpu_neurons, 1, 1);
//         const dim3 block_dims(WARP_SIZE, 1, 1);

//         mul_mat_axpy_sparse<T, type_acc><<<block_nums, block_dims, ncols * sizeof(float), stream>>>(
//             x, y, dst, ncols, nrows, gpu_neu_idx, sparse_idx);
//     } else {  // batch_axpy
//         const dim3 block_nums(num_gpu_neurons, src_ncols, 1);
//         const dim3 block_dims(WARP_SIZE, 1, 1);
//         mul_mat_axpy_sparse_batch<T, type_acc><<<block_nums, block_dims, ncols * sizeof(float), stream>>>(
//             x, y, dst, ncols, nrows, gpu_neu_idx, sparse_idx);
//     }
// }

// kernels: colwise versions
template <typename T, typename type_acc>
__global__ void mul_mat_axpy_sparse_colwise(
    const void * __restrict__ vx,  // [nrows, ncols] row-major: row * ncols + col
    const float * __restrict__ y,  // [nrows]
    float * __restrict__ dst,      // [ncols]
    const int64_t ncols,
    const int64_t nrows,
    const int64_t num_gpu_neurons,
    const int32_t * __restrict__ gpu_neu_idx,  // optional row index remap (int64_t)
    const float * __restrict__ sparse_idx)     // [nrows], mask per row
{
    constexpr int VALS_PER_THREAD = 2;

    const int     t    = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t col0 = (int64_t) t * VALS_PER_THREAD;
    const int64_t col1 = col0 + 1;

    if (col0 >= ncols) {
        return;
    }

    type_acc acc0 = (type_acc) 0;
    type_acc acc1 = (type_acc) 0;

    // iterate over rows you actually have on GPU (num_gpu_neurons)
    for (int64_t r = 0; r < num_gpu_neurons; ++r) {
        const int64_t neu = gpu_neu_idx ? gpu_neu_idx[r] : r;

        // mask: if this neu not active, skip
        if (sparse_idx[neu] < 0.5f) {
            continue;
        }

        const float alpha = y[neu];

        // compute base index for row r
        const int64_t vx_i = r * ncols + col0;

        float v0 = 0.0f;
        float v1 = 0.0f;

        if constexpr (std::is_same<T, half>::value) {
            const half * x = reinterpret_cast<const half *>(vx);
            if (col0 < ncols) {
                v0 = __half2float(x[vx_i]);
            }
            if (col1 < ncols) {
                v1 = __half2float(x[vx_i + 1]);
            }
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            const __nv_bfloat16 * x = reinterpret_cast<const __nv_bfloat16 *>(vx);
            if (col0 < ncols) {
                v0 = __bfloat162float(x[vx_i]);
            }
            if (col1 < ncols) {
                v1 = __bfloat162float(x[vx_i + 1]);
            }
        } else {  // float
            const float * x = reinterpret_cast<const float *>(vx);
            if (col0 < ncols) {
                v0 = x[vx_i];
            }
            if (col1 < ncols) {
                v1 = x[vx_i + 1];
            }
        }

        if (col0 < ncols) {
            acc0 += (type_acc) v0 * (type_acc) alpha;
        }
        if (col1 < ncols) {
            acc1 += (type_acc) v1 * (type_acc) alpha;
        }
    }

    // write back: only this thread writes dst[col*]
    dst[col0] = (float) acc0;
    if (col1 < ncols) {
        dst[col1] = (float) acc1;
    }
}

template <typename T, typename type_acc>
__global__ void mul_mat_axpy_sparse_batch_colwise(const void * __restrict__ vx,  // [nrows, ncols]，shared among batch
                                                  const float * __restrict__ y,  // [batch, nrows]
                                                  float * __restrict__ dst,      // [batch, ncols]
                                                  const int64_t ncols,
                                                  const int64_t nrows,
                                                  const int64_t num_gpu_neurons,
                                                  const int32_t * __restrict__ gpu_neu_idx,  // [nrows] or nullptr
                                                  const float * __restrict__ sparse_idx)     // [batch, nrows]
{
    constexpr int VALS_PER_THREAD = 2;

    const int     token_idx = blockIdx.y;  // batch index
    const int     t         = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t col0      = (int64_t) t * VALS_PER_THREAD;
    const int64_t col1      = col0 + 1;

    if (col0 >= ncols) {
        return;
    }

    const float * y_tok      = y + token_idx * nrows;
    const float * sparse_tok = sparse_idx + token_idx * nrows;
    float *       dst_tok    = dst + token_idx * ncols;

    type_acc acc0 = (type_acc) 0;
    type_acc acc1 = (type_acc) 0;

    for (int64_t r = 0; r < num_gpu_neurons; ++r) {
        const int64_t neu = gpu_neu_idx ? gpu_neu_idx[r] : r;

        if (sparse_tok[neu] < 0.5f) {
            continue;
        }

        const float   alpha = y_tok[neu];
        const int64_t vx_i  = r * ncols + col0;

        float v0 = 0.0f;
        float v1 = 0.0f;

        if constexpr (std::is_same<T, half>::value) {
            const half * x = reinterpret_cast<const half *>(vx);
            if (col0 < ncols) {
                v0 = __half2float(x[vx_i]);
            }
            if (col1 < ncols) {
                v1 = __half2float(x[vx_i + 1]);
            }
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            const __nv_bfloat16 * x = reinterpret_cast<const __nv_bfloat16 *>(vx);
            if (col0 < ncols) {
                v0 = __bfloat162float(x[vx_i]);
            }
            if (col1 < ncols) {
                v1 = __bfloat162float(x[vx_i + 1]);
            }
        } else {
            const float * x = reinterpret_cast<const float *>(vx);
            if (col0 < ncols) {
                v0 = x[vx_i];
            }
            if (col1 < ncols) {
                v1 = x[vx_i + 1];
            }
        }

        if (col0 < ncols) {
            acc0 += (type_acc) v0 * (type_acc) alpha;
        }
        if (col1 < ncols) {
            acc1 += (type_acc) v1 * (type_acc) alpha;
        }
    }

    dst_tok[col0] = (float) acc0;
    if (col1 < ncols) {
        dst_tok[col1] = (float) acc1;
    }
}

// ==============================
// host launcher (fixed names/types)
// ==============================
template <typename T, typename type_acc>
static void launch_mul_mat_axpy_cuda_sparse_new(const T *       vx,
                                                const float *   y,
                                                const float *   sparse_idx,
                                                const int32_t * gpu_neu_idx,
                                                float *         dst,
                                                const int64_t   ncols,
                                                const int64_t   nrows,
                                                const int64_t   src_ncols,  // batch dim of y/dst
                                                const int64_t   num_gpu_neurons,
                                                cudaStream_t    stream) {
    // vector case (src_ncols == 1)
    const int vals_per_thread = 2;
    const int threads         = 128;
    const int cols_per_block  = threads * vals_per_thread;
    const int grid_x          = (int) ((ncols + cols_per_block - 1) / cols_per_block);

    if (src_ncols == 1) {
        // launch single-kernel (1D grid)
        dim3 g(grid_x);
        mul_mat_axpy_sparse_colwise<T, type_acc>
            <<<g, threads, 0, stream>>>(vx, y, dst, ncols, nrows, num_gpu_neurons, gpu_neu_idx, sparse_idx);
    } else {
        // batch case: grid.x over columns, grid.y over tokens
        const int batch = (int) src_ncols;
        dim3      g(grid_x, batch);
        mul_mat_axpy_sparse_batch_colwise<T, type_acc>
            <<<g, threads, 0, stream>>>(vx, y, dst, ncols, nrows, num_gpu_neurons, gpu_neu_idx, sparse_idx);
    }
}

// template <typename T>
// static void mul_mat_axpy_cuda_sparse(const T *       x,
//                                      const float *   y,
//                                      const float *   sparse_idx,
//                                      const int32_t * gpu_neu_idx,
//                                      float *         dst,
//                                      const int64_t   ncols,
//                                      const int64_t   nrows,
//                                      const int64_t   src_ncols,
//                                      const int64_t   num_gpu_neurons,
//                                      enum ggml_prec  prec,
//                                      cudaStream_t    stream) {
//     if constexpr (std::is_same<T, half>::value) {
//         if (prec == GGML_PREC_DEFAULT) {
//             launch_mul_mat_axpy_cuda_sparse<T, half>(x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src_ncols,
//                                                      num_gpu_neurons, stream);
//             return;
//         }
//     }
//     launch_mul_mat_axpy_cuda_sparse<T, float>(x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src_ncols,
//                                               num_gpu_neurons, stream);
// }

// higher-level wrapper (choose type_acc = float)
template <typename T>
static void mul_mat_axpy_cuda_sparse(const T *       vx,
                                     const float *   y,
                                     const float *   sparse_idx,
                                     const int32_t * gpu_neu_idx,
                                     float *         dst,
                                     const int64_t   ncols,
                                     const int64_t   nrows,
                                     const int64_t   src_ncols,
                                     const int64_t   num_gpu_neurons,
                                     enum ggml_prec  prec,
                                     cudaStream_t    stream) {
    GGML_UNUSED(prec);
    launch_mul_mat_axpy_cuda_sparse_new<T, float>(vx, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src_ncols,
                                                  num_gpu_neurons, stream);
}

void ggml_cuda_op_axpy_sparse(ggml_backend_cuda_context & ctx,
                              const ggml_tensor *         src0,
                              const ggml_tensor *         src1,
                              ggml_tensor *               dst,
                              const char *                src0_dd_i,
                              const float *               src1_ddf_i,
                              const char *                src1_ddq_i,
                              float *                     dst_dd_i,
                              const int64_t               row_low,
                              const int64_t               row_high,
                              const int64_t               src1_ncols,
                              const int64_t               src1_padded_row_size,
                              cudaStream_t                stream) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_UNUSED(row_low);
    GGML_UNUSED(row_high);

    GGML_ASSERT(dst->src[2]->data != nullptr && "missing sparse_idx");

    const int64_t ncols = src0->ne[0];  // feature dimension
    const int64_t nrows = src1->ne[0];  // total number of neurons

    float *   sparse_idx      = static_cast<float *>(dst->src[2]->data);
    int32_t * gpu_neu_idx     = dst->src[3] != NULL ? static_cast<int32_t *>(dst->src[3]->data) : NULL;
    int64_t   num_gpu_neurons = dst->src[3] ? dst->src[3]->ne[0] : nrows;

    const int            cc   = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // set dst_dd_i as zero
    CUDA_CHECK(cudaMemsetAsync(dst_dd_i, 0, sizeof(float) * dst->ne[0] * dst->ne[1], stream));

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                const float * src0_d = (const float *) src0_dd_i;
                mul_mat_axpy_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows,
                                         src1_ncols, num_gpu_neurons, prec, stream);
            }
            break;
        case GGML_TYPE_F16:
            {
                const half * src0_d = (const half *) src0_dd_i;
                mul_mat_axpy_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows,
                                         src1_ncols, num_gpu_neurons, prec, stream);
            }
            break;
        case GGML_TYPE_BF16:
            {
                const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
                mul_mat_axpy_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows,
                                         src1_ncols, num_gpu_neurons, prec, stream);
            }
            break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED(ctx);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}
