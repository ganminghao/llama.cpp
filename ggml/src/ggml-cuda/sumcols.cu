#include "sumcols.cuh"

template <int BLOCK_SIZE>
static __global__ void sum_cols_f32(const float * src,
                                    float *       dst,
                                    int64_t       ne0,
                                    int64_t       ne1,
                                    int64_t       ne2,
                                    int64_t       ne3,
                                    int64_t       src_nb0,
                                    int64_t       src_nb1,
                                    int64_t       src_nb2,
                                    int64_t       src_nb3,
                                    int64_t       dst_nb0,
                                    int64_t       dst_nb1,
                                    int64_t       dst_nb2,
                                    int64_t       dst_nb3) {
    GGML_UNUSED(dst_nb1);

    const int64_t i0 = blockIdx.x;
    const int64_t i2 = blockIdx.y;
    const int64_t i3 = blockIdx.z;

    if (i0 >= ne0 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int tid = threadIdx.x;

    const char * src_row = (const char *) src + i0 * src_nb0 + i2 * src_nb2 + i3 * src_nb3;
    char *       dst_out = (char *) dst + i0 * dst_nb0 + i2 * dst_nb2 + i3 * dst_nb3;

    float partial_sum = 0.0f;
    for (int64_t i1 = tid; i1 < ne1; i1 += BLOCK_SIZE) {
        partial_sum += *(const float *) (src_row + i1 * src_nb1);
    }

    float v = warp_reduce_sum<WARP_SIZE>(partial_sum);

    if constexpr (BLOCK_SIZE == WARP_SIZE) {
        if (tid == 0) {
            *(float *) dst_out = v;
        }
        return;
    }

    __shared__ float shmem[WARP_SIZE];

    if (tid < WARP_SIZE) {
        shmem[tid] = 0.0f;
    }
    __syncthreads();

    if ((tid & (WARP_SIZE - 1)) == 0) {
        shmem[tid >> 5] = v;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        float sum = shmem[tid];
        sum       = warp_reduce_sum<WARP_SIZE>(sum);
        if (tid == 0) {
            *(float *) dst_out = sum;
        }
    }
}

void ggml_cuda_op_sum_cols(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    const dim3 grid_dims(ne0, ne2, ne3);

    const int block_size = (ne1 <= 32) ? 32 : (ne1 <= 64) ? 64 : (ne1 <= 128) ? 128 : 256;

    switch (block_size) {
        case 32:
            sum_cols_f32<32><<<grid_dims, dim3(32, 1, 1), 0, stream>>>(
                (const float *) src0->data, (float *) dst->data, ne0, ne1, ne2, ne3, src0->nb[0], src0->nb[1],
                src0->nb[2], src0->nb[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        case 64:
            sum_cols_f32<64><<<grid_dims, dim3(64, 1, 1), 0, stream>>>(
                (const float *) src0->data, (float *) dst->data, ne0, ne1, ne2, ne3, src0->nb[0], src0->nb[1],
                src0->nb[2], src0->nb[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        case 128:
            sum_cols_f32<128><<<grid_dims, dim3(128, 1, 1), 0, stream>>>(
                (const float *) src0->data, (float *) dst->data, ne0, ne1, ne2, ne3, src0->nb[0], src0->nb[1],
                src0->nb[2], src0->nb[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        default:
            sum_cols_f32<256><<<grid_dims, dim3(256, 1, 1), 0, stream>>>(
                (const float *) src0->data, (float *) dst->data, ne0, ne1, ne2, ne3, src0->nb[0], src0->nb[1],
                src0->nb[2], src0->nb[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
    }
}
