#include "common.cuh"

struct mmq_sparse_args {
    const char *    x;
    ggml_type       type_x;
    const int *     y;
    const int32_t * ids_dst;
    const int32_t * expert_bounds;

    float * dst;

    int64_t ncols_x;    // K
    int64_t nrows_x;    // M
    int64_t ncols_dst;  // N

    int64_t stride_row_x;
    int64_t ncols_y;    // N = src1_ncols
    int64_t nrows_dst;  // M

    int64_t nchannels_x;
    int64_t nchannels_y;
    int64_t stride_channel_x;
    int64_t stride_channel_y;
    int64_t stride_channel_dst;

    int64_t nsamples_x;
    int64_t nsamples_y;
    int64_t stride_sample_x;
    int64_t stride_sample_y;
    int64_t stride_sample_dst;

    bool    use_stream_k;
    int64_t ncols_max;

    const float *   sparse_idx;
    const int32_t * gpu_neu_idx;
    int64_t         num_gpu_neurons;
};

void ggml_cuda_mul_mat_sparse_q(ggml_backend_cuda_context & ctx,
                                const ggml_tensor *         src0,
                                const ggml_tensor *         src1,
                                ggml_tensor *               dst);
