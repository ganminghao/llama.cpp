#include "ggml.h"
#include "common.cuh"
#include "axpy_sparse.cuh"

template <typename T, typename type_acc>
static __global__ void mul_mat_axpy_sparse(
    const void * __restrict__ vx, 
    const dfloat * __restrict__ y, 
    float * __restrict__ dst, 
    
    const int ncols, 
    const int nrows, 
    
    const int64_t   * gpu_neu_idx, 
    const float     * sparse_idx
    ) {

    const int blk_idx = blockIdx.x;          // block index, range from [0,nrows]
    const int thds_per_blk = blockDim.x;     // number of threads per block

    const int neu = gpu_neu_idx ? gpu_neu_idx[blk_idx] : blk_idx;
    const int tid = threadIdx.x; // range from [0,31]

    float alpha_fp32 = y[neu];

    if (fabsf(alpha_fp32) < 1e-9f || sparse_idx[neu] < 0.5f) {
        return;
    }

    extern __shared__ float shared_dst[]; // TODO:dynamic
    
    const int VALS_PER_ITER = 2;   // each iter compute 2 vals consequently, we should not modify this
    const int   iter_stride = VALS_PER_ITER * thds_per_blk;

// partial sum for each thread
    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += thds_per_blk) {
        shared_dst[i+tid] = 0;
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col  = i + VALS_PER_ITER*tid;
        const int vx_i = blk_idx*ncols + col; // vx index, vx was store in "blk_idx way", so indice it with blk_idx

        dfloat2 v;
        if constexpr (std::is_same<T, half>::value) {
            const half * x = reinterpret_cast<const half *>(vx);
            v.x = __half2float(x[vx_i + 0]);
            v.y = __half2float(x[vx_i + 1]);
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            const __nv_bfloat16 * x = reinterpret_cast<const __nv_bfloat16 *>(vx);
            v.x = __bfloat162float(x[vx_i + 0]);
            v.y = __bfloat162float(x[vx_i + 1]);
        } else if constexpr (std::is_same<T, float>::value) {
            const float * x = reinterpret_cast<const float *>(vx);
            v.x = x[vx_i + 0];
            v.y = x[vx_i + 1];
        }else {
            static_assert(std::is_same<T, void>::value, "unsupported type for axpy_sparse");
        }

        // matrix multiplication, process 2 vals per j iter
        tmp = v.x * alpha_fp32;
        shared_dst[col] = tmp;  // share_dst[col] = tmp
        tmp = v.y * alpha_fp32;
        shared_dst[col+1] = tmp; // share_dst[col+1] = tmp       
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += thds_per_blk) {
        atomicAdd(&dst[i+tid], shared_dst[i+tid]);
    }
}

template <typename T, typename type_acc>
static __global__ void mul_mat_axpy_sparse_batch(
    const void * __restrict__ vx, 
    const dfloat * __restrict__ y, 
    float * __restrict__ dst, 
    
    const int ncols, 
    const int nrows, 
    
    const int64_t   * gpu_neu_idx, 
    const float     * sparse_idx
    ) {

    const int blk_idx   = blockIdx.x;          // block index, range from [0, nrows]
    const int token_idx = blockIdx.y;         // parallel input index, range from [0, src1_ncols]
    const int thds_per_blk = blockDim.x;     // number of threads per block

    y           += token_idx * nrows;
    dst         += token_idx * ncols;
    sparse_idx  += token_idx * nrows;

    const int neu = gpu_neu_idx ? gpu_neu_idx[blk_idx] : blk_idx;
    const int tid = threadIdx.x; // range from [0,31]

    float alpha_fp32 = y[neu];

    if (fabsf(alpha_fp32) < 1e-9f || sparse_idx[neu] < 0.5f) {
        // if (tid == 0) dst[gpu_neu] = 0.0f;
        return;
    }

    extern __shared__ float shared_dst[]; // TODO:dynamic
    
    const int VALS_PER_ITER = 2;   // each iter compute 2 vals consequently, we should not modify this
    const int   iter_stride = VALS_PER_ITER * thds_per_blk;

// partial sum for each thread
    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += thds_per_blk) {
        shared_dst[i+tid] = 0;
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col  = i + VALS_PER_ITER*tid;
        const int vx_i = blk_idx*ncols + col; // vx index, vx was store in "blk_idx way", so indice it with blk_idx

        dfloat2 v;
        if constexpr (std::is_same<T, half>::value) {
            const half * x = reinterpret_cast<const half *>(vx);
            v.x = __half2float(x[vx_i + 0]);
            v.y = __half2float(x[vx_i + 1]);
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            const __nv_bfloat16 * x = reinterpret_cast<const __nv_bfloat16 *>(vx);
            v.x = __bfloat162float(x[vx_i + 0]);
            v.y = __bfloat162float(x[vx_i + 1]);
        } else if constexpr (std::is_same<T, float>::value) {
            const float * x = reinterpret_cast<const float *>(vx);
            v.x = x[vx_i + 0];
            v.y = x[vx_i + 1];
        } else {
            static_assert(std::is_same<T, void>::value, "unsupported type for axpy_sparse");
        }

        // matrix multiplication, process 2 vals per j iter
        tmp = v.x * alpha_fp32;
        shared_dst[col] = tmp;  // share_dst[col] = tmp
        tmp = v.y * alpha_fp32;
        shared_dst[col+1] = tmp; // share_dst[col+1] = tmp       
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += thds_per_blk) {
        atomicAdd(&dst[i+tid], shared_dst[i+tid]);
    }
}


template <typename T, typename type_acc>
static void launch_mul_mat_axpy_cuda_sparse(
        const T * x, const float * y, const float * sparse_idx, const int64_t * gpu_neu_idx, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t src_ncols, const int64_t num_gpu_neurons, cudaStream_t stream) {
    
    int64_t num_blocks = num_gpu_neurons == 0 ? nrows : num_gpu_neurons;
    // vec_axpy
    if(src_ncols == 1){
        // the lanucher for powerinfer kernel: 
        const dim3 block_nums(num_blocks, 1, 1);
        const dim3 block_dims(WARP_SIZE, 1, 1);

        mul_mat_axpy_sparse<T,type_acc><<<block_nums, block_dims, ncols*sizeof(float), stream>>>(x, y, dst, ncols, nrows, gpu_neu_idx, sparse_idx);
    }
    else{ // batch_axpy
        const dim3 block_nums(num_blocks, src_ncols, 1);
        const dim3 block_dims(WARP_SIZE, 1, 1);
        mul_mat_axpy_sparse_batch<T,type_acc><<<block_nums, block_dims, ncols*sizeof(float), stream>>>(x, y, dst, ncols, nrows, gpu_neu_idx, sparse_idx);
    }

}

template<typename T>
static void mul_mat_axpy_cuda_sparse(
        const T * x, const float * y, const float * sparse_idx, const int64_t * gpu_neu_idx, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t src_ncols, const int64_t num_gpu_neurons,
        enum ggml_prec prec, cudaStream_t stream) {
    if constexpr(std::is_same<T, half>::value) {
        if (prec == GGML_PREC_DEFAULT) {
            launch_mul_mat_axpy_cuda_sparse<T, half>
                (x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src_ncols, num_gpu_neurons, stream);
            return;
        }
    }
    launch_mul_mat_axpy_cuda_sparse<T, float>
        (x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src_ncols, num_gpu_neurons, stream);
}

void ggml_cuda_op_axpy_sparse(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, 
    const ggml_tensor * src1, 
    ggml_tensor *       dst, 

    const char *        src0_dd_i, 
    const float *       src1_ddf_i,
    const char *        src1_ddq_i, 
    float *             dst_dd_i, 

    const int64_t       row_low, 
    const int64_t       row_high, 
    const int64_t       src1_ncols,
    const int64_t       src1_padded_row_size, 
    
    cudaStream_t        stream) 
    {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = row_high - row_low;

    GGML_ASSERT(dst->src[2]->data!=nullptr  && "missing sparse_idx");

    float * sparse_idx = static_cast<float *>(dst->src[2]->data);
    int64_t * gpu_neu_idx = dst->src[3] != NULL ? static_cast<int64_t *>(dst->src[3]->data) : NULL;
    
    int64_t num_gpu_neurons = 0;
    if (dst->src[3]){
        num_gpu_neurons = dst->src[3]->ne[0];
    } 

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // set dst_dd_i as zero
    CUDA_CHECK(cudaMemsetAsync(dst_dd_i, 0, sizeof(float)*dst->ne[0]*dst->ne[1], stream));  

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_axpy_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows, src1_ncols, num_gpu_neurons, prec, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_axpy_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows, src1_ncols, num_gpu_neurons, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_axpy_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows, src1_ncols, num_gpu_neurons, prec, stream);
        } break;
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
