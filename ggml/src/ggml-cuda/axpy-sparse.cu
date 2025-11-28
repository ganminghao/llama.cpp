#include "axpy-sparse.cuh"
#include "common.cuh"
#include "ggml.h"

// ======================================================================================
// Version1: fast version for release, but unreproducible results due to atomicAdd order
// ======================================================================================
#define TILE_TOKENS 4  // we process 4 tokens per block
#define TILE_COLS   4  // we divide ncols into 4 tiles

template <typename T, typename type_acc>
static __global__ void mul_mat_axpy_sparse_rowwise(const void * __restrict__ vx,
                                                   const float * __restrict__ y,
                                                   float * __restrict__ dst,
                                                   const int       ncols,
                                                   const int       nrows,
                                                   const int       src_ncols,
                                                   const int32_t * gpu_neu_idx,
                                                   const float *   sparse_idx) {
    const int blk_idx      = blockIdx.x;   // block index, range from [0,num_gpu_neurons)]
    const int token_ty     = threadIdx.y;
    const int thds_per_blk = blockDim.x;   // number of threads per block
    const int tid          = threadIdx.x;  // range from [0,31]
    const int col_len      = (ncols + TILE_COLS - 1) / TILE_COLS;
    const int col_start    = blockIdx.z * col_len;
    const int col_end      = min(col_start + col_len, ncols);

    const int token_idx = blockIdx.y * blockDim.y + token_ty;

    if (token_idx >= src_ncols) {
        return;
    }

    const float * y_tok      = y + token_idx * nrows;
    float *       dst_tok    = dst + token_idx * ncols;
    const float * sparse_tok = sparse_idx + token_idx * nrows;

    const int neu = gpu_neu_idx ? gpu_neu_idx[blk_idx] : blk_idx;

    float alpha_fp32 = y_tok[neu];

    // init sharemem
    extern __shared__ float shmem[];
    for (int i = tid; i < col_len; i += thds_per_blk) {
        shmem[token_ty * col_len + i] = 0.0f;
    }
    __syncthreads();

    if (sparse_tok[neu] < 0.5f || alpha_fp32 == 0.0f) {
        return;
    }

    const int VALS_PER_ITER = 2;
    const int iter_stride   = VALS_PER_ITER * thds_per_blk;

    for (int i = col_start; i < col_end; i += iter_stride) {
        const int col  = i + VALS_PER_ITER * tid;
        const int vx_i = blk_idx * ncols + col;

        float2 v;
        if constexpr (std::is_same<T, half>::value) {
            const half * x = reinterpret_cast<const half *>(vx);
            v.x            = __half2float(x[vx_i + 0]);
            v.y            = __half2float(x[vx_i + 1]);
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            const __nv_bfloat16 * x = reinterpret_cast<const __nv_bfloat16 *>(vx);
            v.x                     = __bfloat162float(x[vx_i + 0]);
            v.y                     = __bfloat162float(x[vx_i + 1]);
        } else {
            const float * x = reinterpret_cast<const float *>(vx);
            v.x             = x[vx_i + 0];
            v.y             = x[vx_i + 1];
        }
        int local = col - col_start;
        shmem[token_ty * col_len + local + 0] += v.x * alpha_fp32;
        shmem[token_ty * col_len + local + 1] += v.y * alpha_fp32;
    }
    // write back to global memory
    for (int i = tid; i < col_len; i += thds_per_blk) {
        atomicAdd(&dst_tok[col_start + i], shmem[token_ty * col_len + i]);
    }
}

template <typename T, typename type_acc>
static void launch_mul_mat_axpy_cuda_sparse_rowwise(const T *       x,
                                                    const float *   y,
                                                    const float *   sparse_idx,
                                                    const int32_t * gpu_neu_idx,
                                                    float *         dst,
                                                    const int64_t   ncols,
                                                    const int64_t   nrows,
                                                    const int64_t   src_ncols,
                                                    const int64_t   num_gpu_neurons,
                                                    cudaStream_t    stream) {
    dim3      block_nums;
    dim3      block_dims;
    size_t    share_mem_size = 0;
    const int col_len        = (ncols + TILE_COLS - 1) / TILE_COLS;
    if (src_ncols == 1) {
        block_nums     = dim3(num_gpu_neurons, 1, TILE_COLS);
        block_dims     = dim3(WARP_SIZE, 1, 1);
        share_mem_size = col_len * sizeof(float);
    } else {
        block_nums     = dim3(num_gpu_neurons, (src_ncols + TILE_TOKENS - 1) / TILE_TOKENS, TILE_COLS);
        block_dims     = dim3(WARP_SIZE, TILE_TOKENS, 1);
        share_mem_size = col_len * TILE_TOKENS * sizeof(float);
    }
    mul_mat_axpy_sparse_rowwise<T, type_acc><<<block_nums, block_dims, share_mem_size, stream>>>(
        x, y, dst, ncols, nrows, src_ncols, gpu_neu_idx, sparse_idx);
}

// ======================================================================================
// Version2: reproducible version for debugging, no atomicAdd, but much slower
// ======================================================================================
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

template <typename T, typename type_acc>
static void launch_mul_mat_axpy_cuda_sparse_colwise(const T *       vx,
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
    const int threads         = 64;
    const int cols_per_block  = threads * vals_per_thread;
    const int grid_x          = (int) ((ncols + cols_per_block - 1) / cols_per_block);

    const int batch = (int) src_ncols;
    dim3      g(grid_x, batch);
    mul_mat_axpy_sparse_colwise<T, type_acc>
        <<<g, threads, 0, stream>>>(vx, y, dst, ncols, nrows, num_gpu_neurons, gpu_neu_idx, sparse_idx);
}

template <typename T>
static void mul_mat_axpy_cuda_sparse(const T *       x,
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
    static const bool k_spif_fast_axpy = (getenv("SPIF_FAST_AXPY") != nullptr);
    if (!k_spif_fast_axpy) {
        launch_mul_mat_axpy_cuda_sparse_colwise<T, float>(x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src_ncols,
                                                          num_gpu_neurons, stream);
        return;
    } else {
        if constexpr (std::is_same<T, half>::value) {
            if (prec == GGML_PREC_DEFAULT) {
                launch_mul_mat_axpy_cuda_sparse_rowwise<T, half>(x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows,
                                                                 src_ncols, num_gpu_neurons, stream);
                return;
            }
        }
        launch_mul_mat_axpy_cuda_sparse_rowwise<T, float>(x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src_ncols,
                                                          num_gpu_neurons, stream);
    }
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
