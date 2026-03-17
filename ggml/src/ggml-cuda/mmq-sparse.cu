#include "common.cuh"
#include "ggml.h"
#include "mmq-sparse.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <stdint.h>

static __device__ __forceinline__ float dot_block_q8_0_q8_1_dp4a(const block_q8_0 & bx, const block_q8_1 & by) {
    const int8_t * x_qs8 = bx.qs;
    const int8_t * y_qs8 = by.qs;

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < QK8_0; i += 4) {
        int ax = (int) (uint8_t) x_qs8[i + 0] | ((int) (uint8_t) x_qs8[i + 1] << 8) |
                 ((int) (uint8_t) x_qs8[i + 2] << 16) | ((int) (uint8_t) x_qs8[i + 3] << 24);

        int ay = (int) (uint8_t) y_qs8[i + 0] | ((int) (uint8_t) y_qs8[i + 1] << 8) |
                 ((int) (uint8_t) y_qs8[i + 2] << 16) | ((int) (uint8_t) y_qs8[i + 3] << 24);

        sumi = ggml_cuda_dp4a(ax, ay, sumi);
    }

    const float d0 = __half2float(bx.d);
    const float d1 = __half2float(__low2half(by.ds));

    return d0 * d1 * (float) sumi;
}

static __device__ __forceinline__ float dot_block_q4_k_q8_1(const block_q4_K & bx, const block_q8_1 * by) {
    float sumf = 0.0f;
    int   kbx  = 0;

#pragma unroll
    for (int iqs = 0; iqs < QI4_K; iqs += VDR_Q4_K_Q8_1_MMVQ) {
        sumf += vec_dot_q4_K_q8_1((const void *) &bx, by, kbx, iqs);
    }

    return sumf;
}

template <int block_size>
static __global__ void mul_mat_vec_sparse_q8_0_q8_1(const block_q8_0 * __restrict__ w_q8,      // [nrows * num_blocks]
                                                    const block_q8_1 * __restrict__ x_q8,      // [num_blocks]
                                                    const float * __restrict__ sparse_col,     // [nrows]
                                                    const int32_t * __restrict__ gpu_neu_idx,  // [num_gpu_neurons]
                                                    float * __restrict__ dst_col,              // [nrows]
                                                    const int64_t nrows,
                                                    const int64_t num_gpu_neurons,
                                                    const int64_t num_blocks) {
    const int row_block = blockIdx.x;  // 0 .. num_gpu_neurons-1
    const int tid       = threadIdx.x;

    if (row_block >= num_gpu_neurons) {
        return;
    }

    const int32_t neu = gpu_neu_idx ? gpu_neu_idx[row_block] : row_block;

    if (neu >= nrows) {
        return;
    }

    if (sparse_col[neu] < 0.5f) {
        return;
    }

    const block_q8_0 * row_w = w_q8 + (int64_t) row_block * num_blocks;

    extern __shared__ float smem[];
    float *                 buf = smem;

    float sumf = 0.0f;

    for (int64_t ib = tid; ib < num_blocks; ib += block_size) {
        const block_q8_0 & bx = row_w[ib];
        const block_q8_1 & by = x_q8[ib];

        sumf += dot_block_q8_0_q8_1_dp4a(bx, by);
    }

    sumf = warp_reduce_sum<WARP_SIZE>(sumf);

    if (block_size > WARP_SIZE) {
        if (tid % WARP_SIZE == 0) {
            buf[tid / WARP_SIZE] = sumf;
        }

        __syncthreads();

        if (tid >= WARP_SIZE) {
            return;
        }

        sumf = (tid < block_size / WARP_SIZE) ? buf[tid] : 0.0f;
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }

    if (tid == 0) {
        dst_col[neu] = sumf;
    }
}

template <int block_size>
static __global__ void mul_mat_batch_sparse_q8_0_q8_1(
    const block_q8_0 * __restrict__ w_q8,      // [nrows * num_blocks]
    const block_q8_1 * __restrict__ x_q8,      // [src1_ncols * num_blocks]
    const float * __restrict__ sparse_idx,     // [src1_ncols * nrows]
    const int32_t * __restrict__ gpu_neu_idx,  // [num_gpu_neurons]
    float * __restrict__ dst,                  // [src1_ncols * nrows]
    const int64_t ncols,                       // K
    const int64_t nrows,                       // M
    const int64_t src1_ncols,                  // N
    const int64_t num_gpu_neurons) {
    const int64_t num_blocks = ncols / QK8_0;

    const int64_t row_block = blockIdx.x;  // 0 .. num_gpu_neurons-1
    const int64_t col       = blockIdx.y;  // 0 .. src1_ncols-1
    const int     tid       = threadIdx.x;

    if (row_block >= num_gpu_neurons || col >= src1_ncols) {
        return;
    }

    const int32_t neu = gpu_neu_idx ? gpu_neu_idx[row_block] : row_block;
    if (neu >= nrows) {
        return;
    }

    const block_q8_0 * row_w      = w_q8 + (int64_t) row_block * num_blocks;
    const block_q8_1 * x_col      = x_q8 + (int64_t) col * num_blocks;
    float *            dst_col    = dst + (int64_t) col * nrows;
    const float *      sparse_col = sparse_idx + (int64_t) col * nrows;

    if (sparse_col[neu] < 0.5f) {
        return;
    }

    extern __shared__ float smem[];
    float *                 buf = smem;

    float sumf = 0.0f;

    for (int64_t ib = tid; ib < num_blocks; ib += block_size) {
        const block_q8_0 & bx = row_w[ib];
        const block_q8_1 & by = x_col[ib];

        sumf += dot_block_q8_0_q8_1_dp4a(bx, by);
    }

    sumf = warp_reduce_sum<WARP_SIZE>(sumf);

    if (block_size > WARP_SIZE) {
        if (tid % WARP_SIZE == 0) {
            buf[tid / WARP_SIZE] = sumf;
        }
        __syncthreads();

        if (tid >= WARP_SIZE) {
            return;
        }

        sumf = (tid < block_size / WARP_SIZE) ? buf[tid] : 0.0f;
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }

    if (tid == 0) {
        dst_col[neu] = sumf;
    }
}

template <int block_size>
static __global__ void mul_mat_vec_sparse_q4_K_q8_1(const block_q4_K * __restrict__ w_q4_k,    // [nrows * num_blocks]
                                                    const block_q8_1 * __restrict__ x_q8,       // [num_blocks * q8_per_block]
                                                    const float * __restrict__ sparse_col,       // [nrows]
                                                    const int32_t * __restrict__ gpu_neu_idx,    // [num_gpu_neurons]
                                                    float * __restrict__ dst_col,                // [nrows]
                                                    const int64_t nrows,
                                                    const int64_t num_gpu_neurons,
                                                    const int64_t num_blocks,
                                                    const int64_t q8_per_block) {
    const int row_block = blockIdx.x;
    const int tid       = threadIdx.x;

    if (row_block >= num_gpu_neurons) {
        return;
    }

    const int32_t neu = gpu_neu_idx ? gpu_neu_idx[row_block] : row_block;
    if (neu >= nrows) {
        return;
    }

    if (sparse_col[neu] < 0.5f) {
        return;
    }

    const block_q4_K * row_w = w_q4_k + (int64_t) row_block * num_blocks;

    extern __shared__ float smem[];
    float *                 buf = smem;

    float sumf = 0.0f;

    for (int64_t ib = tid; ib < num_blocks; ib += block_size) {
        sumf += dot_block_q4_k_q8_1(row_w[ib], x_q8 + ib * q8_per_block);
    }

    sumf = warp_reduce_sum<WARP_SIZE>(sumf);

    if (block_size > WARP_SIZE) {
        if (tid % WARP_SIZE == 0) {
            buf[tid / WARP_SIZE] = sumf;
        }
        __syncthreads();

        if (tid >= WARP_SIZE) {
            return;
        }

        sumf = (tid < block_size / WARP_SIZE) ? buf[tid] : 0.0f;
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }

    if (tid == 0) {
        dst_col[neu] = sumf;
    }
}

template <int block_size>
static __global__ void mul_mat_batch_sparse_q4_K_q8_1(
    const block_q4_K * __restrict__ w_q4_k,    // [nrows * num_blocks]
    const block_q8_1 * __restrict__ x_q8,      // [src1_ncols * num_blocks * q8_per_block]
    const float * __restrict__ sparse_idx,     // [src1_ncols * nrows]
    const int32_t * __restrict__ gpu_neu_idx,  // [num_gpu_neurons]
    float * __restrict__ dst,                  // [src1_ncols * nrows]
    const int64_t ncols,                       // K
    const int64_t nrows,                       // M
    const int64_t src1_ncols,                  // N
    const int64_t num_gpu_neurons,
    const int64_t q8_per_block) {
    const int64_t num_blocks = ncols / QK_K;

    const int64_t row_block = blockIdx.x;
    const int64_t col       = blockIdx.y;
    const int     tid       = threadIdx.x;

    if (row_block >= num_gpu_neurons || col >= src1_ncols) {
        return;
    }

    const int32_t neu = gpu_neu_idx ? gpu_neu_idx[row_block] : row_block;
    if (neu >= nrows) {
        return;
    }

    const block_q4_K * row_w      = w_q4_k + (int64_t) row_block * num_blocks;
    const block_q8_1 * x_col      = x_q8 + (int64_t) col * num_blocks * q8_per_block;
    float *            dst_col    = dst + (int64_t) col * nrows;
    const float *      sparse_col = sparse_idx + (int64_t) col * nrows;

    if (sparse_col[neu] < 0.5f) {
        return;
    }

    extern __shared__ float smem[];
    float *                 buf = smem;

    float sumf = 0.0f;

    for (int64_t ib = tid; ib < num_blocks; ib += block_size) {
        sumf += dot_block_q4_k_q8_1(row_w[ib], x_col + ib * q8_per_block);
    }

    sumf = warp_reduce_sum<WARP_SIZE>(sumf);

    if (block_size > WARP_SIZE) {
        if (tid % WARP_SIZE == 0) {
            buf[tid / WARP_SIZE] = sumf;
        }
        __syncthreads();

        if (tid >= WARP_SIZE) {
            return;
        }

        sumf = (tid < block_size / WARP_SIZE) ? buf[tid] : 0.0f;
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }

    if (tid == 0) {
        dst_col[neu] = sumf;
    }
}

static void launch_mul_mat_cuda_sparse_q(ggml_backend_cuda_context & ctx,
                                         const mmq_sparse_args &     args,
                                         cudaStream_t                stream) {
    GGML_UNUSED(ctx);

    const int64_t nrows           = args.nrows_dst;  // M
    const int64_t ncols           = args.ncols_x;    // K
    const int64_t num_gpu_neurons = args.num_gpu_neurons;
    const int64_t src1_ncols      = args.ncols_y;    // N

    const bool is_q8_0 = args.type_x == GGML_TYPE_Q8_0;
    const bool is_q4_k = args.type_x == GGML_TYPE_Q4_K;
    GGML_ASSERT(is_q8_0 || is_q4_k);

    const int64_t q8_per_block = is_q4_k ? (QK_K / QK8_1) : 1;
    if (is_q8_0) {
        GGML_ASSERT(ncols % QK8_0 == 0);
    } else {
        GGML_ASSERT(ncols % QK_K == 0);
    }
    const int64_t num_blocks = is_q8_0 ? (ncols / QK8_0) : (ncols / QK_K);

    const block_q8_0 * w_q8   = (const block_q8_0 *) args.x;
    const block_q4_K * w_q4_k = (const block_q4_K *) args.x;
    const block_q8_1 * x_q8   = (const block_q8_1 *) args.y;
    float *            dst    = args.dst;

    const float *   sparse_idx  = args.sparse_idx;
    const int32_t * gpu_neu_idx = args.gpu_neu_idx;

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int64_t block_size_best = WARP_SIZE;
    int64_t niter_best      = (num_blocks + block_size_best - 1) / block_size_best;
    int64_t max_block_size  = 256;

    if (ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD &&
        ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }

    for (int64_t bs = 2 * WARP_SIZE; bs <= max_block_size; bs += WARP_SIZE) {
        const int64_t niter = (num_blocks + bs - 1) / bs;
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = bs;
        }
    }

    const int smem = WARP_SIZE * sizeof(float);

    if (src1_ncols == 1) {
        dim3 grid(num_gpu_neurons, 1, 1);

        switch (block_size_best) {
            case 32:
                if (is_q8_0) {
                    mul_mat_vec_sparse_q8_0_q8_1<32><<<grid, 32, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 nrows, num_gpu_neurons, num_blocks);
                } else {
                    mul_mat_vec_sparse_q4_K_q8_1<32><<<grid, 32, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 nrows, num_gpu_neurons, num_blocks, q8_per_block);
                }
                break;
            case 64:
                if (is_q8_0) {
                    mul_mat_vec_sparse_q8_0_q8_1<64><<<grid, 64, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 nrows, num_gpu_neurons, num_blocks);
                } else {
                    mul_mat_vec_sparse_q4_K_q8_1<64><<<grid, 64, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 nrows, num_gpu_neurons, num_blocks, q8_per_block);
                }
                break;
            case 96:
                if (is_q8_0) {
                    mul_mat_vec_sparse_q8_0_q8_1<96><<<grid, 96, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 nrows, num_gpu_neurons, num_blocks);
                } else {
                    mul_mat_vec_sparse_q4_K_q8_1<96><<<grid, 96, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 nrows, num_gpu_neurons, num_blocks, q8_per_block);
                }
                break;
            case 128:
                if (is_q8_0) {
                    mul_mat_vec_sparse_q8_0_q8_1<128><<<grid, 128, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                   nrows, num_gpu_neurons, num_blocks);
                } else {
                    mul_mat_vec_sparse_q4_K_q8_1<128><<<grid, 128, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                   nrows, num_gpu_neurons, num_blocks, q8_per_block);
                }
                break;
            case 160:
                if (is_q8_0) {
                    mul_mat_vec_sparse_q8_0_q8_1<160><<<grid, 160, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                   nrows, num_gpu_neurons, num_blocks);
                } else {
                    mul_mat_vec_sparse_q4_K_q8_1<160><<<grid, 160, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                   nrows, num_gpu_neurons, num_blocks, q8_per_block);
                }
                break;
            case 192:
                if (is_q8_0) {
                    mul_mat_vec_sparse_q8_0_q8_1<192><<<grid, 192, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                   nrows, num_gpu_neurons, num_blocks);
                } else {
                    mul_mat_vec_sparse_q4_K_q8_1<192><<<grid, 192, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                   nrows, num_gpu_neurons, num_blocks, q8_per_block);
                }
                break;
            case 224:
                if (is_q8_0) {
                    mul_mat_vec_sparse_q8_0_q8_1<224><<<grid, 224, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                   nrows, num_gpu_neurons, num_blocks);
                } else {
                    mul_mat_vec_sparse_q4_K_q8_1<224><<<grid, 224, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                   nrows, num_gpu_neurons, num_blocks, q8_per_block);
                }
                break;
            case 256:
            default:
                if (is_q8_0) {
                    mul_mat_vec_sparse_q8_0_q8_1<256><<<grid, 256, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                   nrows, num_gpu_neurons, num_blocks);
                } else {
                    mul_mat_vec_sparse_q4_K_q8_1<256><<<grid, 256, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                   nrows, num_gpu_neurons, num_blocks, q8_per_block);
                }
                break;
        }

        CUDA_CHECK(cudaGetLastError());
        return;
    }

    dim3 grid(num_gpu_neurons, src1_ncols, 1);

    switch (block_size_best) {
        case 32:
            if (is_q8_0) {
                mul_mat_batch_sparse_q8_0_q8_1<32><<<grid, 32, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                               ncols, nrows, src1_ncols, num_gpu_neurons);
            } else {
                mul_mat_batch_sparse_q4_K_q8_1<32><<<grid, 32, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                               ncols, nrows, src1_ncols, num_gpu_neurons, q8_per_block);
            }
            break;
        case 64:
            if (is_q8_0) {
                mul_mat_batch_sparse_q8_0_q8_1<64><<<grid, 64, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                               ncols, nrows, src1_ncols, num_gpu_neurons);
            } else {
                mul_mat_batch_sparse_q4_K_q8_1<64><<<grid, 64, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                               ncols, nrows, src1_ncols, num_gpu_neurons, q8_per_block);
            }
            break;
        case 96:
            if (is_q8_0) {
                mul_mat_batch_sparse_q8_0_q8_1<96><<<grid, 96, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                               ncols, nrows, src1_ncols, num_gpu_neurons);
            } else {
                mul_mat_batch_sparse_q4_K_q8_1<96><<<grid, 96, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                               ncols, nrows, src1_ncols, num_gpu_neurons, q8_per_block);
            }
            break;
        case 128:
            if (is_q8_0) {
                mul_mat_batch_sparse_q8_0_q8_1<128><<<grid, 128, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 ncols, nrows, src1_ncols, num_gpu_neurons);
            } else {
                mul_mat_batch_sparse_q4_K_q8_1<128><<<grid, 128, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 ncols, nrows, src1_ncols, num_gpu_neurons, q8_per_block);
            }
            break;
        case 160:
            if (is_q8_0) {
                mul_mat_batch_sparse_q8_0_q8_1<160><<<grid, 160, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 ncols, nrows, src1_ncols, num_gpu_neurons);
            } else {
                mul_mat_batch_sparse_q4_K_q8_1<160><<<grid, 160, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 ncols, nrows, src1_ncols, num_gpu_neurons, q8_per_block);
            }
            break;
        case 192:
            if (is_q8_0) {
                mul_mat_batch_sparse_q8_0_q8_1<192><<<grid, 192, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 ncols, nrows, src1_ncols, num_gpu_neurons);
            } else {
                mul_mat_batch_sparse_q4_K_q8_1<192><<<grid, 192, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 ncols, nrows, src1_ncols, num_gpu_neurons, q8_per_block);
            }
            break;
        case 224:
            if (is_q8_0) {
                mul_mat_batch_sparse_q8_0_q8_1<224><<<grid, 224, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 ncols, nrows, src1_ncols, num_gpu_neurons);
            } else {
                mul_mat_batch_sparse_q4_K_q8_1<224><<<grid, 224, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 ncols, nrows, src1_ncols, num_gpu_neurons, q8_per_block);
            }
            break;
        case 256:
        default:
            if (is_q8_0) {
                mul_mat_batch_sparse_q8_0_q8_1<256><<<grid, 256, smem, stream>>>(w_q8, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 ncols, nrows, src1_ncols, num_gpu_neurons);
            } else {
                mul_mat_batch_sparse_q4_K_q8_1<256><<<grid, 256, smem, stream>>>(w_q4_k, x_q8, sparse_idx, gpu_neu_idx, dst,
                                                                                 ncols, nrows, src1_ncols, num_gpu_neurons, q8_per_block);
            }
            break;
    }

    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_sparse_q(ggml_backend_cuda_context & ctx,
                                const ggml_tensor *         src0,
                                const ggml_tensor *         src1,  // F32
                                ggml_tensor *               dst)  // F32
{
    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(dst->src[2]->data != nullptr && "missing sparse_idx");

    float *   sparse_idx      = static_cast<float *>(dst->src[2]->data);
    int32_t * gpu_neu_idx     = dst->src[3] ? static_cast<int32_t *>(dst->src[3]->data) : nullptr;
    int64_t   num_gpu_neurons = dst->src[3] ? dst->src[3]->ne[0] : dst->src[2]->ne[0];

    const int            cc   = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;
    GGML_UNUSED(prec);

    cudaStream_t stream = ctx.stream();

    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    const char *  src0_d = (const char *) src0->data;   // Q8_0
    const float * src1_d = (const float *) src1->data;  // F32
    float *       dst_d  = (float *) dst->data;         // F32

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  = dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  = dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  = dst->nb[3] / ts_dst;

    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t nrows_src1       = ne11 * ne12 * ne13;
    const int64_t nblocks_per_row  = ne10 / QK8_1;
    const size_t  nbytes_src1_q8_1 = nrows_src1 * nblocks_per_row * sizeof(block_q8_1);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.ptr, src1->type, ne10, s11, s12, s13,  // ne00, s01, s02, s03
                               ne10, ne11, ne12, ne13,                                           // ne0, ne1, ne2, ne3
                               stream);

        CUDA_CHECK(cudaGetLastError());
    }
    if (src0->type == GGML_TYPE_Q8_0) {
        GGML_ASSERT(ne00 % QK8_0 == 0);
    } else {
        GGML_ASSERT(ne00 % QK_K == 0);
    }

    const int64_t src1_ncols      = ne11;             // N
    const int64_t nblocks_per_col = nblocks_per_row;  // = ne10 / QK8_1

    const mmq_sparse_args args = {
        src0_d,
        src0->type,
        (const int *) src1_q8_1.ptr,
        nullptr,
        nullptr,
        dst_d,
        ne00,
        ne01,
        ne1,
        s01,
        src1_ncols,
        s1,
        ne02,
        ne12,
        s02,
        nblocks_per_col,
        s2,
        ne03,
        ne13,
        s03,
        0,
        s3,
        false,
        ne1,
        sparse_idx,
        gpu_neu_idx,
        num_gpu_neurons,
    };

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
            launch_mul_mat_cuda_sparse_q(ctx, args, stream);
            break;
        default:
            GGML_ABORT("ggml_cuda_mul_mat_sparse_q: unsupported src0 type: %s", ggml_type_name(src0->type));
    }
}
