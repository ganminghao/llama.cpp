#include "sumcols.cuh"

// Kernel: sum over dimension ne[1] for tensors with shape {m, n, 1, 1}
// Each block handles one "row" indexed by i0 (blockIdx.x)
// Threads accumulate partial sums across ne[1] with stride blockDim.x,
// then reduce inside shared memory to produce a single float output.

static __global__ void sum_cols_f32(const float * src,
                                    float *       dst,
                                    const int64_t ne00,
                                    const int64_t ne01,
                                    const int64_t ne02,
                                    const int64_t ne03,
                                    const int64_t nb00,
                                    const int64_t nb01,
                                    const int64_t nb02,
                                    const int64_t nb03,
                                    const int64_t nb0,
                                    const int64_t nb1,
                                    const int64_t nb2,
                                    const int64_t nb3) {
    // Each block processes one (i0, i2, i3)
    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i0 = blockIdx.x;

    if (i0 >= ne00 || i2 >= ne02 || i3 >= ne03) {
        return;
    }

    const int tid = threadIdx.x;

    // Base address for src row: src[i0, :, i2, i3]
    const char * src_base = (const char *) src + i0 * nb00 + i2 * nb02 + i3 * nb03;

    // Destination pointer for dst[i0, 0, i2, i3]
    char * dst_ptr = (char *) dst + i0 * nb0 + i2 * nb2 + i3 * nb3;

    // Each thread accumulates partial sum along ne[1]
    float partial_sum = 0.0f;
    for (int64_t i1 = tid; i1 < ne01; i1 += blockDim.x) {
        const float * vptr = (const float *) (src_base + i1 * nb01);
        partial_sum += *vptr;
    }

    // Shared memory for block reduction
    extern __shared__ float shmem[];
    shmem[tid] = partial_sum;
    __syncthreads();

    // Tree-reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmem[tid] += shmem[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the output
    if (tid == 0) {
        float * out = (float *) dst_ptr;
        *out        = shmem[0];
    }
    GGML_UNUSED(nb1);
}

// ggml entry point
void ggml_cuda_op_sum_cols(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dim3 block_dims(256, 1, 1);
    dim3 grid_dims(src0->ne[0], src0->ne[2], src0->ne[3]);

    size_t shmem_size = block_dims.x * sizeof(float);

    sum_cols_f32<<<grid_dims, block_dims, shmem_size, stream>>>(
        (const float *) src0->data, (float *) dst->data, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
}
