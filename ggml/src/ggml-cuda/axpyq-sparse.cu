#include "axpyq-sparse.cuh"
#include "common.cuh"
#include "ggml.h"
#include <stdint.h>

#define Q8_BLOCK_SIZE 34
#define Q8_NUM_B 32
#define TILE_TOKENS 4  // we process 4 tokens per block
#define TILE_COLS   4  // we divide ncols into 8 tiles

static __device__ inline float fp16_to_fp32_device(uint16_t h) {
    union {
        uint16_t u;
        __half   h;
    } tmp;
    tmp.u = h;
    return __half2float(tmp.h);
}

static __global__ void mul_mat_axpy_sparse_rowwise_q(const void * __restrict__ vx,
                                                     const float * __restrict__ y,
                                                     float * __restrict__ dst,
                                                     const int       ncols,
                                                     const int       nrows,
                                                     const int       qblock_num,
                                                     const int       block_len,
                                                     const int       src_ncols,
                                                     const int32_t * gpu_neu_idx,
                                                     const float *   sparse_idx) {
    const int blk_idx      = blockIdx.x;   // block index, range from [0,num_gpu_neurons)
    const int token_ty     = threadIdx.y;  // [0, TILE_TOKENS)
    const int thds_per_blk = blockDim.x;   // number of threads per block (WARP_SIZE)
    const int tid          = threadIdx.x;  // range from [0,31]

    const int token_idx = blockIdx.y * blockDim.y + token_ty;
    if (token_idx >= src_ncols) {
        return;
    }

    const float * y_tok      = y + token_idx * nrows;
    float *       dst_tok    = dst + token_idx * ncols;
    const float * sparse_tok = sparse_idx + token_idx * nrows;

    const int neu = gpu_neu_idx ? gpu_neu_idx[blk_idx] : blk_idx;

    const float alpha_fp32 = y_tok[neu];

    const int tile_id      = blockIdx.z;
    const int block_start  = tile_id * block_len;                               // block index start in this tile
    const int block_end    = min(block_start + block_len, qblock_num);          // block index end (exclusive)
    const int tile_blocks  = max(block_end - block_start, 0);
    const int col_start    = block_start * Q8_NUM_B;                            // column start for this tile
    const int col_end      = min(col_start + tile_blocks * Q8_NUM_B, ncols);    // column end for this tile
    const int col_len      = max(col_end - col_start, 0);                       // number of valid columns in this tile

    if (col_len <= 0) {
        return;
    }

    extern __shared__ float shmem[];
    const int tile_cols = block_len * Q8_NUM_B;

    // init shmem
    for (int i = tid; i < col_len; i += thds_per_blk) {
        shmem[token_ty * tile_cols + i] = 0.0f;
    }
    __syncthreads();

    if (sparse_tok[neu] < 0.5f || alpha_fp32 == 0.0f) {
        return;
    }

    const char * wdata = reinterpret_cast<const char *>(vx);

    const int row_block_offset = blk_idx * qblock_num;

    for (int col = col_start + tid; col < col_end; col += thds_per_blk) {
        const int local_col = col - col_start;

        const int block_idx_col = col / Q8_NUM_B;
        const int idx_in_block  = col - block_idx_col * Q8_NUM_B;

        const int flat_block_idx = row_block_offset + block_idx_col;
        const int byte_offset    = flat_block_idx * Q8_BLOCK_SIZE;

        const uint16_t * d_ptr = reinterpret_cast<const uint16_t *>(wdata + byte_offset);
        const uint16_t   d_h   = *d_ptr;
        const float      d     = fp16_to_fp32_device(d_h);

        const int8_t * qs = reinterpret_cast<const int8_t *>(wdata + byte_offset + sizeof(uint16_t));
        const float    w  = d * static_cast<float>(qs[idx_in_block]);

        shmem[token_ty * tile_cols + local_col] += w * alpha_fp32;
    }

    for (int i = tid; i < col_len; i += thds_per_blk) {
        atomicAdd(&dst_tok[col_start + i], shmem[token_ty * tile_cols + i]);
    }
}

static void launch_mul_mat_axpy_cuda_sparse_rowwise(const char  *   x,
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
    const int qblock_num = ncols / Q8_NUM_B + (ncols % Q8_NUM_B != 0 ? 1 : 0);  // number of blocks per neurons
    const int block_len  = (qblock_num + TILE_COLS - 1) / TILE_COLS;            // number of blocks per tile
    if (src_ncols == 1) {
        block_nums     = dim3(num_gpu_neurons, 1, TILE_COLS);
        block_dims     = dim3(WARP_SIZE, 1, 1);
        share_mem_size = block_len * Q8_NUM_B * sizeof(float);
    } else {
        block_nums     = dim3(num_gpu_neurons, (src_ncols + TILE_TOKENS - 1) / TILE_TOKENS, TILE_COLS);
        block_dims     = dim3(WARP_SIZE, TILE_TOKENS, 1);
        share_mem_size = block_len * Q8_NUM_B * TILE_TOKENS * sizeof(float);
    }
    mul_mat_axpy_sparse_rowwise_q<<<block_nums, block_dims, share_mem_size, stream>>>(
        x, y, dst, ncols, nrows, qblock_num, block_len, src_ncols, gpu_neu_idx, sparse_idx);
}


void ggml_cuda_op_axpy_sparse_q(ggml_backend_cuda_context & ctx,
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
        case GGML_TYPE_Q8_0:
            {
                launch_mul_mat_axpy_cuda_sparse_rowwise(src0_dd_i, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows, dst->ne[1],
                                                                  num_gpu_neurons, stream);
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
