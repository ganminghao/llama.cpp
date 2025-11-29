#include "common.cuh"
#include "quantize.cuh"
#include "ggml.h"
#include "axpyq-sparse.cuh"

#ifdef GGML_CUDA_F16
typedef half dfloat; // dequantize float
typedef half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef float2 dfloat2;
#endif //GGML_CUDA_F16

// #define QK8_0 32
// #define QR8_0 1
// #define QI8_0 (QK8_0 / (4 * QR8_0))
// typedef struct {
//     half    d;              // delta
//     int8_t  qs[QK8_0];      // quants
// } block_q8_0;
// static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");
#define AXPY_BLOCK_X 512
#define AXPY_BLOCK_Y 1
#define AXPY_BLOCK_Z 256
#define GGML_CUDA_DMMV_X 32
#define GGML_CUDA_MMV_Y 1


static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

template <int qk, int qr> 
static __global__ void dequantize_mul_mat_axpy_sparse_pro(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows, const int num_gpu_neurons, const int32_t *lst, const float *idx) {
    const int thread_col = blockIdx.y * blockDim.x + threadIdx.x;
    const int col = 2 * thread_col;
    const int tid = threadIdx.x;
    const int wid = threadIdx.y;
    const int iqs = (col%qk) / qr; // x quant index
    const int row_offset = blockIdx.z * AXPY_BLOCK_Z;

    if (col >= ncols) {
        return;
    }

    __shared__ dfloat dst_tmp[AXPY_BLOCK_Y][AXPY_BLOCK_X*2];
    __shared__ dfloat y_tmp[AXPY_BLOCK_Z]; 

    dst_tmp[wid][tid] = 0.0;
    dst_tmp[wid][tid+AXPY_BLOCK_X] = 0.0;
    
    if (wid == 0) {
        if (lst) {
            for(int i=tid; i<AXPY_BLOCK_Z; i += AXPY_BLOCK_X) {
                y_tmp[i] = y[lst[i+row_offset]];
            }
        } else {
            for(int i=tid; i<AXPY_BLOCK_Z; i += AXPY_BLOCK_X) {
                // ((dfloat4*)y_tmp)[i] = *(dfloat4*)(&y[i*4+row_offset]);
                y_tmp[i] = y[i+row_offset];
            }
        }
    }
    __syncthreads();

#pragma unroll 8
    for(int gpu_row = wid; gpu_row < num_gpu_neurons; gpu_row += AXPY_BLOCK_Y) {        
        if(y_tmp[gpu_row] == 0.0) continue;

        const int ib = ((gpu_row + row_offset)*ncols + col) / qk; // x block index
        
        dfloat2 v;
        dequantize_q8_0(vx, ib, iqs, v);

        dst_tmp[wid][tid] += v.x * y_tmp[gpu_row];
        dst_tmp[wid][tid+AXPY_BLOCK_X] += v.y * y_tmp[gpu_row];
    }

    for (int offset = AXPY_BLOCK_Y / 2; offset > 0; offset >>= 1) {
        if (wid < offset) {
            dst_tmp[wid][tid] += dst_tmp[wid+offset][tid];
            dst_tmp[wid][tid+AXPY_BLOCK_X] += dst_tmp[wid][tid+AXPY_BLOCK_X];
        }
        __syncthreads();
    }

    if (wid == 0) {
        const int iybs = col - col%qk; // y block start index
        const int y_offset = qr == 1 ? 1 : qk/2;
        atomicAdd(&dst[iybs + iqs], dst_tmp[wid][tid]);
        atomicAdd(&dst[iybs + iqs + y_offset], dst_tmp[wid][tid+AXPY_BLOCK_X]); 
    }
}


template <int qk, int qr>
static __global__ void dequantize_mul_mat_axpy_sparse_batch(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows, const int num_gpu_neurons, int src1_ne0, int src1_ncols, const int32_t *lst, float *idx) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int gpu_row = blockIdx.y*blockDim.y + threadIdx.y;

    if (gpu_row >= num_gpu_neurons) {
        return;
    }
    int row = lst ? lst[gpu_row] : gpu_row;
    const int bid = blockIdx.y;

    extern __shared__ float shared_dst[]; // TODO:dynamic

    const int tid = threadIdx.x;

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;
    float * loop_idx = idx;
    dfloat * loop_y = (dfloat *)y;
    float * loop_dst = dst;

// partial sum for each thread
    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += GGML_CUDA_DMMV_X) {
        shared_dst[i+tid] = 0;
    }
    // __syncthreads();
    for (int col_id = 0; col_id < src1_ncols; col_id++) {
        __syncthreads();
        if (loop_idx[row] < 0.5f) {
            loop_dst += ncols;
            loop_idx += src1_ne0;
            loop_y += src1_ne0;
            continue;
        }
        

        for (int i = 0; i < ncols; i += iter_stride)
        {
            const int col = i + vals_per_iter * tid;
            const int ib = (gpu_row * ncols + col) / qk; // x block index
            const int iqs = (col % qk) / qr;         // x quant index
            const int iybs = col - col % qk;         // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
            for (int j = 0; j < vals_per_iter; j += 2)
            {
                // process 2 vals per j iter

                // dequantize
                // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
                dfloat2 v;
                dequantize_q8_0(vx, ib, iqs + j / qr, v);

                // matrix multiplication
                // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
                tmp = v.x * loop_y[row];
                shared_dst[iybs + iqs + j / qr + 0] = tmp;
                tmp = v.y * loop_y[row];
                shared_dst[iybs + iqs + j / qr + y_offset] = tmp;
            }
        }
        /* __syncthreads(); */

        for (int i = 0; i < ncols; i += GGML_CUDA_DMMV_X)
        {
            atomicAdd(&loop_dst[i + tid], shared_dst[i + tid]);
            shared_dst[i+tid] = 0;
        }
        loop_dst += ncols;
        loop_idx += src1_ne0;
        loop_y += src1_ne0;
    }
}

static void dequantize_axpy_sparse_vec_q8_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, const int num_gpu_neurons, cudaStream_t stream, const int32_t *lst, float *idx)  {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (ncols + AXPY_BLOCK_X*2 - 1) / AXPY_BLOCK_X / 2;
    const int block_num_z = nrows / AXPY_BLOCK_Z;  
    const dim3 block_nums(1, block_num_y, block_num_z);
    const dim3 block_dims(AXPY_BLOCK_X, AXPY_BLOCK_Y, 1);
    // dequantize_mul_mat_axpy<QK4_0, QR4_0, dequantize_q4_0>
    //     <<<block_nums, block_dims, ncols*sizeof(float), stream>>>(vx, y, dst, ncols, nrows);
    // printf("launch kernel: (%d, %d)\n", block_num_x, block_num_y);
    dequantize_mul_mat_axpy_sparse_pro<QK8_0, 1>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, num_gpu_neurons, AXPY_BLOCK_Z, lst, idx);
}

static void dequantize_axpy_sparse_batch_q8_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, const int num_gpu_neurons, int src1_rows, int src1_ncols, cudaStream_t stream, const int32_t *lst, float *idx) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_axpy_sparse_batch<QK8_0, 1>
        <<<block_nums, block_dims, ncols*sizeof(float), stream>>>(vx, y, dst, ncols, nrows, num_gpu_neurons, src1_rows, src1_ncols, lst, idx);
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
                              cudaStream_t                stream){
    const int64_t ne11 = src1->ne[1]; // input batch size
    const int64_t ne10 = src1->ne[0]; // input feature size
    const int64_t row_diff = row_high - row_low;

    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_CUDA_F16
    size_t ash;
    dfloat * src1_dfloat = nullptr; // dfloat == half

    bool src1_convert_f16 = src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_F16;

    if (src1_convert_f16) {
        src1_dfloat = (half *) ggml_cuda_pool_malloc(ne00*sizeof(half), &ash);
        ggml_cpy_f32_f16_cuda((const char *) src1_ddf_i, (char *) src1_dfloat, ne00,
                                ne00, 1, sizeof(float), 0, 0,
                                ne00, 1, sizeof(half),  0, 0, stream);
    }
#else
    const dfloat * src1_dfloat = (const dfloat *) src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_CUDA_F16
    const int64_t ncols = src0->ne[0];  // feature dimension
    const int64_t nrows = src1->ne[0];  // total number of neurons

    float *   sparse_idx      = static_cast<float *>(dst->src[2]->data);
    const int32_t * gpu_neu_idx     = dst->src[3] != NULL ? static_cast<int32_t *>(dst->src[3]->data) : NULL;
    const int64_t   num_gpu_neurons = dst->src[3] ? dst->src[3]->ne[0] : nrows;

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            if (ne11 == 1) {
                dequantize_axpy_sparse_vec_q8_0_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ncols, nrows, num_gpu_neurons, stream, gpu_neu_idx, sparse_idx);
            } else {
                dequantize_axpy_sparse_batch_q8_0_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ncols, nrows, num_gpu_neurons, ne10, src1_ncols, stream, gpu_neu_idx, sparse_idx);
            }
            break;
        default:
            GGML_ABORT("unsupported src0 type %s for ggml_cuda_op_axpy_sparse_q", ggml_type_name(src0->type));
    }

#ifdef GGML_CUDA_F16
    if (src1_convert_f16) {
        ggml_cuda_pool_free(src1_dfloat, ash);
    }
#endif // GGML_CUDA_F16

    (void) src1;
    (void) dst;
    (void) src1_ddq_i;
    (void) src1_ncols;
    (void) src1_padded_row_size;
}