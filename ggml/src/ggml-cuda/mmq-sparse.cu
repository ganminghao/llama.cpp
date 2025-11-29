#include "common.cuh"
#include "quantize.cuh"
#include "ggml.h"
#include "mmq-sparse.cuh"
#include "vecdotq.cuh"
#include "mma.cuh"

struct mmq_sparse_args {
    const char * x; ggml_type type_x; const int * y; const int32_t * ids_dst; const int32_t * expert_bounds; float * dst;
    int64_t ncols_x; int64_t nrows_x; int64_t ncols_dst; int64_t stride_row_x; int64_t ncols_y; int64_t nrows_dst;
    int64_t nchannels_x; int64_t nchannels_y; int64_t stride_channel_x; int64_t stride_channel_y; int64_t stride_channel_dst;
    int64_t nsamples_x; int64_t nsamples_y; int64_t stride_sample_x; int64_t stride_sample_y; int64_t stride_sample_dst;
    bool use_stream_k; int64_t ncols_max; const float * sparse_idx; const int32_t * gpu_neu_idx; int64_t num_gpu_neurons;
};

template <typename type_acc, int block_size>  
static __global__ void mul_mat_vec_sparse_q8_0(  
    const block_q8_0 * __restrict__ x,  
    const block_q8_1 * __restrict__ y,  
    const float * __restrict__ sparse_idx,  
    const int32_t * __restrict__ gpu_neu_idx,  
    float * __restrict__ dst,  
    const int64_t ncols2) {  
      
    const int64_t row = blockIdx.x;  
    const int tid = threadIdx.x;  
      
    int32_t neu = gpu_neu_idx ? gpu_neu_idx[row] : row;  
      
    if (sparse_idx[neu] < 0.5f) {  
        return;  
    }  
      
    // 计算当前行的 q8_0 和 q8_1 块数量  
    const int64_t num_blocks = (ncols2 * 2) / QK8_0;  
      
    extern __shared__ char data_mmv[];  
    float * buf_iw = (float *) data_mmv;  
      
    if (block_size > WARP_SIZE) {  
        if (tid < WARP_SIZE) {  
            buf_iw[tid] = 0.0f;  
        }  
        __syncthreads();  
    }  
      
    float sumf = 0.0f;  
      
    // 每个 block 处理一个 q8_0/q8_1 块对  
    for (int64_t block_idx = tid; block_idx < num_blocks; block_idx += block_size) {  
        const block_q8_0 & bx = x[row * num_blocks + block_idx];  
        const block_q8_1 & by = y[block_idx];  
          
        // 加载量化数据到 int 数组以便 dp4a 处理  
        const int * x_qs = (const int *)bx.qs;  
        const int * y_qs = (const int *)by.qs;  
          
        int sumi = 0;  
          
        // 使用 dp4a 计算点积  
#pragma unroll  
        for (int i = 0; i < QK8_0/sizeof(int); ++i) {  
            sumi = ggml_cuda_dp4a(x_qs[i], y_qs[i], sumi);  
        }  
          
        // 应用缩放因子  
        const float d8_0 = __half2float(bx.d);  
        const float2 d8_1 = __half22float2(by.ds);  
          
        sumf += d8_0 * d8_1.x * sumi + d8_0 * d8_1.y * (QK8_0/QI8_1);  
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
  
// batch 版本  
template <typename type_acc, int block_size>  
static __global__ void mul_mat_batch_sparse_q8_0(  
    const block_q8_0 * __restrict__ x,  
    const block_q8_1 * __restrict__ y,  
    const float * __restrict__ sparse_idx,  
    const int32_t * __restrict__ gpu_neu_idx,  
    float * __restrict__ dst,  
    const int64_t ncols,  
    const int64_t nrows,  
    const int64_t src1_ncols) {  
      
    const int64_t num_blocks = ncols / QK8_0;  
      
    const int64_t row = blockIdx.x;  
    const int64_t s1col_b = blockIdx.y;  
    const int tid = threadIdx.x;  
      
    int32_t neu = gpu_neu_idx ? gpu_neu_idx[row] : row;  
      
    x += num_blocks * row;  
    y += num_blocks * s1col_b;  
    dst += nrows * s1col_b;  
    sparse_idx += nrows * s1col_b;  
      
    if (sparse_idx[neu] < 0.5f) {  
        return;  
    }  
      
    extern __shared__ char data_mmv[];  
    float * buf_iw = (float *) data_mmv;  
      
    if (block_size > WARP_SIZE) {  
        if (tid < WARP_SIZE) {  
            buf_iw[tid] = 0.0f;  
        }  
        __syncthreads();  
    }  
      
    float sumf = 0.0f;  
      
    for (int64_t block_idx = tid; block_idx < num_blocks; block_idx += block_size) {  
        const block_q8_0 & bx = x[block_idx];  
        const block_q8_1 & by = y[block_idx];  
          
        const int * x_qs = (const int *)bx.qs;  
        const int * y_qs = (const int *)by.qs;  
          
        int sumi = 0;  
          
#pragma unroll  
        for (int i = 0; i < QK8_0/sizeof(int); ++i) {  
            sumi = ggml_cuda_dp4a(x_qs[i], y_qs[i], sumi);  
        }  
          
        const float d8_0 = __half2float(bx.d);  
        const float2 d8_1 = __half22float2(by.ds);  
          
        sumf += d8_0 * d8_1.x * sumi + d8_0 * d8_1.y * (QK8_0/QI8_1);  
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

static void launch_mul_mat_cuda_sparse_q(  
        ggml_backend_cuda_context & ctx,  
        const mmq_sparse_args & args,  
        cudaStream_t stream) {  
    GGML_UNUSED(ctx);
    int nrows = args.nrows_dst;  
    int ncols = args.ncols_x;  
    int num_gpu_neurons = args.num_gpu_neurons;  
    int src1_ncols = args.ncols_y;  
  
    GGML_ASSERT(ncols % QK8_0 == 0);
    int device;  
    CUDA_CHECK(cudaGetDevice(&device));  
  
    int64_t block_size_best = WARP_SIZE;  
    int64_t niter_best = (ncols + QK8_0 * WARP_SIZE - 1) / (QK8_0 * WARP_SIZE);  
    int64_t max_block_size = 256;  
      
    if (ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD &&  
        ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {  
        max_block_size = 128;  
    }  
  
    for (int64_t block_size = 2 * WARP_SIZE; block_size <= max_block_size; block_size += WARP_SIZE) {  
        const int64_t niter = (ncols + QK8_0 * block_size - 1) / (QK8_0 * block_size);  
        if (niter < niter_best) {  
            niter_best = niter;  
            block_size_best = block_size;  
        }  
    }  
  
    const int smem = WARP_SIZE * sizeof(float);  
  
    if (src1_ncols == 1) {  
        // vector case  
        dim3 grid(num_gpu_neurons, 1, 1);  
        const int64_t num_blocks = ncols / QK8_0;  
          
        switch (block_size_best) {
            case 32:
                mul_mat_vec_sparse_q8_0<float, 32><<<grid, dim3(32, 1, 1), smem, stream>>>(
                        (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                        args.sparse_idx, args.gpu_neu_idx, args.dst, num_blocks);
                break;
            case 64:
                mul_mat_vec_sparse_q8_0<float, 64><<<grid, dim3(64, 1, 1), smem, stream>>>(
                        (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                        args.sparse_idx, args.gpu_neu_idx, args.dst, num_blocks);
                break;
            case 96:
                mul_mat_vec_sparse_q8_0<float, 96><<<grid, dim3(96, 1, 1), smem, stream>>>(
                        (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                        args.sparse_idx, args.gpu_neu_idx, args.dst, num_blocks);
                break;
            case 128:
                mul_mat_vec_sparse_q8_0<float, 128><<<grid, dim3(128, 1, 1), smem, stream>>>(
                        (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                        args.sparse_idx, args.gpu_neu_idx, args.dst, num_blocks);
                break;
            case 160:
                mul_mat_vec_sparse_q8_0<float, 160><<<grid, dim3(160, 1, 1), smem, stream>>>(
                        (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                        args.sparse_idx, args.gpu_neu_idx, args.dst, num_blocks);
                break;
            case 192:
                mul_mat_vec_sparse_q8_0<float, 192><<<grid, dim3(192, 1, 1), smem, stream>>>(
                        (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                        args.sparse_idx, args.gpu_neu_idx, args.dst, num_blocks);
                break;
            case 224:
                mul_mat_vec_sparse_q8_0<float, 224><<<grid, dim3(224, 1, 1), smem, stream>>>(
                        (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                        args.sparse_idx, args.gpu_neu_idx, args.dst, num_blocks);
                break;
            case 256:
            default:
                mul_mat_vec_sparse_q8_0<float, 256><<<grid, dim3(256, 1, 1), smem, stream>>>(
                        (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                        args.sparse_idx, args.gpu_neu_idx, args.dst, num_blocks);
                break;
        }
    } else {  
        // batch case  
        dim3 grid(num_gpu_neurons, src1_ncols, 1);  
        const int64_t num_blocks = ncols / QK8_0;  
          
        switch (block_size_best) {
            case 32:
                mul_mat_batch_sparse_q8_0<float, 32><<<grid, dim3(32, 1, 1), smem, stream>>>(
                    (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                    args.sparse_idx, args.gpu_neu_idx, args.dst,
                    ncols, nrows, src1_ncols);
                break;
            case 64:
                mul_mat_batch_sparse_q8_0<float, 64><<<grid, dim3(64, 1, 1), smem, stream>>>(
                    (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                    args.sparse_idx, args.gpu_neu_idx, args.dst,
                    ncols, nrows, src1_ncols);
                break;
            case 96:
                mul_mat_batch_sparse_q8_0<float, 96><<<grid, dim3(96, 1, 1), smem, stream>>>(
                    (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                    args.sparse_idx, args.gpu_neu_idx, args.dst,
                    ncols, nrows, src1_ncols);
                break;
            case 128:
                mul_mat_batch_sparse_q8_0<float, 128><<<grid, dim3(128, 1, 1), smem, stream>>>(
                    (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                    args.sparse_idx, args.gpu_neu_idx, args.dst,
                    ncols, nrows, src1_ncols);
                break;
            case 160:
                mul_mat_batch_sparse_q8_0<float, 160><<<grid, dim3(160, 1, 1), smem, stream>>>(
                    (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                    args.sparse_idx, args.gpu_neu_idx, args.dst,
                    ncols, nrows, src1_ncols);
                break;
            case 192:
                mul_mat_batch_sparse_q8_0<float, 192><<<grid, dim3(192, 1, 1), smem, stream>>>(
                    (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                    args.sparse_idx, args.gpu_neu_idx, args.dst,
                    ncols, nrows, src1_ncols);
                break;
            case 224:
                mul_mat_batch_sparse_q8_0<float, 224><<<grid, dim3(224, 1, 1), smem, stream>>>(
                    (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                    args.sparse_idx, args.gpu_neu_idx, args.dst,
                    ncols, nrows, src1_ncols);
                break;
            case 256:
            default:
                mul_mat_batch_sparse_q8_0<float, 256><<<grid, dim3(256, 1, 1), smem, stream>>>(
                    (const block_q8_0*)args.x, (const block_q8_1*)args.y,
                    args.sparse_idx, args.gpu_neu_idx, args.dst,
                    ncols, nrows, src1_ncols);
                break;
        }
    }  
}

void ggml_cuda_mul_mat_sparse_q(
        ggml_backend_cuda_context & ctx, 
        const ggml_tensor * src0, 
        const ggml_tensor * src1, 
        ggml_tensor * dst){
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT((dst->src[2]->data) != nullptr && "missing sparse_idx");

    float *   sparse_idx      = static_cast<float *>(dst->src[2]->data);
    int32_t * gpu_neu_idx     = dst->src[3] != NULL ? static_cast<int32_t *>(dst->src[3]->data) : NULL;
    int64_t   num_gpu_neurons = dst->src[3] ? dst->src[3]->ne[0] : dst->src[2]->ne[0];

    const int            cc   = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    cudaStream_t stream = ctx.stream();

    // set dst_dd_i as zero
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // quantize src1 into q8_1
    const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type,
            ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = ne11*ne10_padded * sizeof(block_q8_1)/(QK8_1*sizeof(int));
    const int64_t s13 = ne12*s12;

    const mmq_sparse_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
        ne00, ne01, ne1, s01, ne11, s1,
        ne02, ne12, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        false, ne1, sparse_idx, gpu_neu_idx, num_gpu_neurons};
    switch (src0->type)
    {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_cuda_sparse_q(ctx, args, stream);
            break;
        default:
            break;
    }
    return;
}