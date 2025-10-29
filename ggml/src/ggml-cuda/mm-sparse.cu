#include "common.cuh"
#include "ggml.h"
#include "mm-sparse.cuh"

// vec
template <typename T, typename type_acc, int block_size>
static __global__ void mul_mat_vec_sparse(const T * __restrict__ x,
                                          const float * __restrict__ y,
                                          const float * __restrict__ sparse_idx,
                                          const int32_t * __restrict__ gpu_neu_idx,
                                          float * __restrict__ dst,
                                          const int64_t ncols2) {
    const int64_t row = blockIdx.x;                      // (0, num_gpu_neurons)
    const int     tid = threadIdx.x;                     // (0, 256)

    int32_t neu = gpu_neu_idx ? gpu_neu_idx[row] : row;  // (one of the neurons(on gpu) original index)

    if (sparse_idx[neu] < 0.5f) {                        // GTODO: do we need sparse_threshold?
        return;
    }

    x += row * ncols2 * 2;

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float *                buf_iw = (float *) data_mmv;

    if (block_size > WARP_SIZE) {
        if (tid < WARP_SIZE) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf = 0.0f;

    if constexpr (std::is_same<T, float>::value) {
        const float2 * x2 = (const float2 *) x;

        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
            const float2 tmpy = y2[col2];
            sumf += tmpx.x * tmpy.x;
            sumf += tmpx.y * tmpy.y;
        }
    } else if constexpr (std::is_same<T, half>::value) {
        const half2 * x2 = (const half2 *) x;
        if (std::is_same<type_acc, float>::value) {
            for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                const float2 tmpy = y2[col2];
                sumf += tmpx.x * tmpy.x;
                sumf += tmpx.y * tmpy.y;
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2 = make_half2(0.0f, 0.0f);
            for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmp = y2[col2];
                sumh2 += x2[col2] * make_half2(tmp.x, tmp.y);
            }

            sumf = __low2float(sumh2) + __high2float(sumh2);
#else
            NO_DEVICE_CODE;
#endif  // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        const int * x2 = (const int *) x;
        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const int    tmpx = x2[col2];
            const float2 tmpy = y2[col2];
            sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]) * tmpy.x;
            sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]) * tmpy.y;
        }
    } else {
        static_assert(std::is_same<T, void>::value, "unsupported type");
    }

    sumf = warp_reduce_sum<WARP_SIZE>(sumf);

    if (block_size > WARP_SIZE) {
        buf_iw[tid / WARP_SIZE] = sumf;
        __syncthreads();
        if (tid >= WARP_SIZE) {
            return;
        }
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }

    if (tid != 0) {
        return;
    }

    dst[neu] = sumf;
}

//

// batch
template <typename T, typename type_acc, int block_size>
static __global__ void mul_mat_batch_sparse(const T * __restrict__ x,
                                            const float * __restrict__ y,
                                            const float * __restrict__ sparse_idx,
                                            const int32_t * __restrict__ gpu_neu_idx,
                                            float * __restrict__ dst,
                                            const int64_t ncols,
                                            const int64_t nrows,
                                            const int64_t src1_ncols  // token batch number
) {
    const int64_t ncols2 = ncols / 2;

    const int64_t row     = blockIdx.x;   // (0, num_gpu_neurons)
    const int64_t s1col_b = blockIdx.y;   // (0, scr1_ncols) the block that responsible for the specific token in batch
    const int     tid     = threadIdx.x;  // (0, 256)

    int32_t neu = gpu_neu_idx ? gpu_neu_idx[row] : row;  // (one of the gpu_neurons index)

    x += ncols * row;
    y += ncols * s1col_b;
    dst += nrows * s1col_b;
    sparse_idx += nrows * s1col_b;

    if (sparse_idx[neu] < 0.5f) {  // GTODO: do we need sparse_threshold?
        return;
    }

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float *                buf_iw = (float *) data_mmv;

    if (block_size > WARP_SIZE) {
        if (tid < WARP_SIZE) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf = 0.0f;

    if constexpr (std::is_same<T, float>::value) {
        const float2 * x2 = (const float2 *) x;

        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
            const float2 tmpy = y2[col2];
            sumf += tmpx.x * tmpy.x;
            sumf += tmpx.y * tmpy.y;
        }
    } else if constexpr (std::is_same<T, half>::value) {
        const half2 * x2 = (const half2 *) x;

        if (std::is_same<type_acc, float>::value) {
            for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                const float2 tmpy = y2[col2];
                sumf += tmpx.x * tmpy.x;
                sumf += tmpx.y * tmpy.y;
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2 = make_half2(0.0f, 0.0f);

            for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmp = y2[col2];
                sumh2 += x2[col2] * make_half2(tmp.x, tmp.y);
            }

            sumf = __low2float(sumh2) + __high2float(sumh2);
#else
            NO_DEVICE_CODE;
#endif  // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        const int * x2 = (const int *) x;
        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const int    tmpx = x2[col2];
            const float2 tmpy = y2[col2];
            sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]) * tmpy.x;
            sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]) * tmpy.y;
        }
    } else {
        static_assert(std::is_same<T, void>::value, "unsupported type");
    }

    sumf = warp_reduce_sum<WARP_SIZE>(sumf);

    if (block_size > WARP_SIZE) {
        buf_iw[tid / WARP_SIZE] = sumf;
        __syncthreads();
        if (tid >= WARP_SIZE) {
            return;
        }
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }

    if (tid != 0) {
        return;
    }

    dst[neu] = sumf;
}

// // powerinfer kernels:
// static __device__ void convert_f16(const void * vx, const int ib, const int iqs, dfloat2 & v){
//     const half * x = (const half *) vx;

//     // automatic half -> float type cast if dfloat == float
//     v.x = x[ib + iqs + 0];
//     v.y = x[ib + iqs + 1];
// }

// static __global__ void dequantize_mul_mat_batch_sparse(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows, int src1_cols, int dst_ne0,int64_t * lst, float * idx) {
//     // qk = quantized weights per x block
//     // qr = number of quantized weights per data value in x block
//     const int gpu_row = blockIdx.y*blockDim.y + threadIdx.y;
//     int qk = 1;
//     int qr = 1;
//     int GGML_CUDA_DMMV_X = 32;

//     if (gpu_row >= nrows) {
//         return;
//     }
//     int row = lst ? lst[gpu_row] : gpu_row;

//     const int tid = threadIdx.x;

//     const int iter_stride = 2*GGML_CUDA_DMMV_X;
//     const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
//     const int y_offset = qr == 1 ? 1 : qk/2;
//     float * loop_idx = idx;;
//     dfloat * loop_y = (dfloat *)y;
//     float * loop_dst = dst;

//     float tmp = 0.0f;

//     for (int col_id = 0; col_id < src1_cols; col_id++)
//     {
//         __syncthreads();
//         tmp = 0.0f;
//         // if (loop_idx[row] < 0.5f)
//         // {
//         //     loop_dst += dst_ne0;
//         //     loop_idx += dst_ne0;
//         //     loop_y += ncols;
//         //     continue;
//         // }

//         for (int i = 0; i < ncols; i += iter_stride)
//         {
//             const int col = i + vals_per_iter * tid;
//             const int ib = (gpu_row * ncols + col) / qk; // x block index
//             const int iqs = (col % qk) / qr;         // x quant index
//             const int iybs = col - col % qk;         // y block start index

// // processing >2 values per i iter is faster for fast GPUs
// #pragma unroll
//             for (int j = 0; j < vals_per_iter; j += 2)
//             {
//                 // process 2 vals per j iter

//                 // dequantize
//                 // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
//                 dfloat2 v;
//                 convert_f16(vx, ib, iqs + j / qr, v);

//                 // matrix multiplication

//                 tmp += v.x * loop_y[iybs + iqs + j / qr + 0];
//                 tmp += v.y * loop_y[iybs + iqs + j / qr + y_offset];
//                 // #endif
//             }
//         }
//         atomicAdd(&loop_dst[row], tmp);
//         loop_dst += dst_ne0;
//         loop_idx += dst_ne0;
//         loop_y += ncols;
//     }
// }

// static __global__ void print(
//     const float *       __restrict__ sparse_idx,
//     const int32_t *     __restrict__ gpu_neu_idx,
//     const int64_t ncols,
//     const int64_t nrows,
//     const int64_t src1_ncols)
// {
//     if (blockIdx.x == 0 && threadIdx.x == 0) {
//         printf("\n=== sparse_idx ===\n");
//         for (int t = 0; t < src1_ncols; t++) {
//             int cnt = 0;
//             for (int i = t * nrows; i < (t + 1) * nrows; i++) {
//                 if (sparse_idx[i] > 0.5f) {
//                     cnt++;
//                 }
//             }
//             printf("sparse_idx cnt in batch %d: %d, activation_ratio: %.3f\n", t, cnt, float(cnt) / float(nrows));
//         }

//         if(gpu_neu_idx){
//             printf("=== gpu_neu_idx ===\n");
//             for (int i = 0; i < nrows; i++) {
//                 printf("gpu_neu_idx[%d] = %lld\n", i, gpu_neu_idx[i]);
//             }
//         }

//     }
// }

template <typename T, typename type_acc>
static void launch_mul_mat_cuda_sparse(const T *       x,
                                       const float *   y,
                                       const float *   sparse_idx,
                                       const int32_t * gpu_neu_idx,
                                       float *         dst,
                                       const int64_t   ncols,
                                       const int64_t   nrows,
                                       const int64_t   src1_ncols,
                                       int64_t         num_gpu_neurons,
                                       cudaStream_t    stream) {
    // print base  address
    // printf("\n x base address: %p, y base address: %p, dst base address: %p, sparse_idx base address: %p\n", x, y, dst, sparse_idx);
    // printf("launch_mul_mat_cuda_sparse: ncols=%ld, nrows=%ld, src1_ncols=%ld, num_gpu_neurons=%ld\n", ncols, nrows, src1_ncols, num_gpu_neurons);
    // print<<<1, 32, 0, stream>>>(sparse_idx, gpu_neu_idx, ncols, nrows, src1_ncols);

    GGML_ASSERT(ncols % 2 == 0);
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int64_t block_size_best = WARP_SIZE;
    int64_t niter_best      = (ncols + 2 * WARP_SIZE - 1) / (2 * WARP_SIZE);
    int64_t max_block_size  = 256;
    if (ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD &&
        ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }

    // GTODO: understand why we choose block_size like this, do we need to change this in sparse inference?
    for (int64_t block_size = 2 * WARP_SIZE; block_size <= max_block_size; block_size += WARP_SIZE) {
        const int64_t niter = (ncols + 2 * block_size - 1) / (2 * block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    // Shared memory size
    const int smem = WARP_SIZE * sizeof(float);
    // printf("src1_ncols=%ld, num_gpu_neurons=%ld, block_size_best=%ld, niter_best=%ld\n", src1_ncols, num_gpu_neurons, block_size_best, niter_best);

    if (src1_ncols == 1) {
        // vector case
        dim3 grid(num_gpu_neurons, 1, 1);
        switch (block_size_best) {
            case 32:
                mul_mat_vec_sparse<T, type_acc, 32>
                    <<<grid, dim3(32, 1, 1), smem, stream>>>(x, y, sparse_idx, gpu_neu_idx, dst, ncols / 2);
                break;
            case 64:
                mul_mat_vec_sparse<T, type_acc, 64>
                    <<<grid, dim3(64, 1, 1), smem, stream>>>(x, y, sparse_idx, gpu_neu_idx, dst, ncols / 2);
                break;
            case 96:
                mul_mat_vec_sparse<T, type_acc, 96>
                    <<<grid, dim3(96, 1, 1), smem, stream>>>(x, y, sparse_idx, gpu_neu_idx, dst, ncols / 2);
                break;
            case 128:
                mul_mat_vec_sparse<T, type_acc, 128>
                    <<<grid, dim3(128, 1, 1), smem, stream>>>(x, y, sparse_idx, gpu_neu_idx, dst, ncols / 2);
                break;
            case 160:
                mul_mat_vec_sparse<T, type_acc, 160>
                    <<<grid, dim3(160, 1, 1), smem, stream>>>(x, y, sparse_idx, gpu_neu_idx, dst, ncols / 2);
                break;
            case 192:
                mul_mat_vec_sparse<T, type_acc, 192>
                    <<<grid, dim3(192, 1, 1), smem, stream>>>(x, y, sparse_idx, gpu_neu_idx, dst, ncols / 2);
                break;
            case 224:
                mul_mat_vec_sparse<T, type_acc, 224>
                    <<<grid, dim3(224, 1, 1), smem, stream>>>(x, y, sparse_idx, gpu_neu_idx, dst, ncols / 2);
                break;
            case 256:
            default:
                mul_mat_vec_sparse<T, type_acc, 256>
                    <<<grid, dim3(256, 1, 1), smem, stream>>>(x, y, sparse_idx, gpu_neu_idx, dst, ncols / 2);
                break;
        }
    } else {
        // Batch case
        // dequantize_mul_mat_batch_sparse<<<dim3(1, nrows, 1), dim3(32, 1, 1), 0, stream>>>(
        //     x, (const dfloat *)y, dst, ncols, nrows, src1_ncols, nrows, (int64_t *)gpu_neu_idx, (float *)sparse_idx);
        dim3 grid(num_gpu_neurons, src1_ncols, 1);
        // float * sparse_idx_squeezed;
        // CUDA_CHECK(cudaMalloc((void **)&sparse_idx_squeezed, sizeof(float)*nrows));
        // squeezed_idx<<<dim3(num_gpu_neurons,1,1), dim3(32,1,1), 0, stream>>>(sparse_idx, nrows, src1_ncols, sparse_idx_squeezed);
        // sparse_idx = sparse_idx_squeezed;

        // check powerinfer type batch kernels
        // dequantize_mul_mat_batch_sparse<<<dim3(1, num_gpu_neurons, 1), dim3(32, 1, 1), 0, stream>>>(
        //     x, (const dfloat *)y, dst, ncols, nrows, src1_ncols, nrows, (int64_t *)gpu_neu_idx, (float *)sparse_idx);

        switch (block_size_best) {
            case 32:
                mul_mat_batch_sparse<T, type_acc, 32><<<grid, dim3(32, 1, 1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 64:
                mul_mat_batch_sparse<T, type_acc, 64><<<grid, dim3(64, 1, 1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 96:
                mul_mat_batch_sparse<T, type_acc, 96><<<grid, dim3(96, 1, 1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 128:
                mul_mat_batch_sparse<T, type_acc, 128><<<grid, dim3(128, 1, 1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 160:
                mul_mat_batch_sparse<T, type_acc, 160><<<grid, dim3(160, 1, 1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 192:
                mul_mat_batch_sparse<T, type_acc, 192><<<grid, dim3(192, 1, 1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 224:
                mul_mat_batch_sparse<T, type_acc, 224><<<grid, dim3(224, 1, 1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 256:
            default:
                mul_mat_batch_sparse<T, type_acc, 256><<<grid, dim3(256, 1, 1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
        }
    }
}

template <typename T>
static void mul_mat_cuda_sparse(const T *       x,
                                const float *   y,
                                const float *   sparse_idx,
                                const int32_t * gpu_neu_idx,
                                float *         dst,
                                const int64_t   ncols,
                                const int64_t   nrows,
                                const int64_t   src1_ncols,
                                const int64_t   num_gpu_neurons,
                                enum ggml_prec  prec,
                                cudaStream_t    stream) {
    if constexpr (std::is_same<T, half>::value) {
        if (prec == GGML_PREC_DEFAULT) {
            launch_mul_mat_cuda_sparse<T, half>(x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols,
                                                num_gpu_neurons, stream);
            return;
        }
    }
    launch_mul_mat_cuda_sparse<T, float>(x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols, num_gpu_neurons,
                                         stream);
}

void ggml_cuda_op_mul_mat_sparse(ggml_backend_cuda_context & ctx,
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

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = dst->ne[0];

    GGML_ASSERT((dst->src[2]->data) != nullptr && "missing sparse_idx");

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
                mul_mat_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows, src1_ncols,
                                    num_gpu_neurons, prec, stream);
            }
            break;
        case GGML_TYPE_F16:
            {
                const half * src0_d = (const half *) src0_dd_i;
                mul_mat_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows, src1_ncols,
                                    num_gpu_neurons, prec, stream);
            }
            break;
        case GGML_TYPE_BF16:
            {
                const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
                mul_mat_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows, src1_ncols,
                                    num_gpu_neurons, prec, stream);
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
