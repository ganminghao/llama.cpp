#include "llama-sparkinfer.h"

#include "ggml-cuda.h"
#include "llama-impl.h"

#include <cstring>
#include <numeric>

const static int k_spif_gpu_only = get_env_int("SPIF_GPU_ONLY", 0);

const bool sparkinfer_layer_cache::k_enable_spif_reload =
    (getenv("SPIF_PARALLEL") != nullptr && getenv("SPIF_RELOAD") != nullptr);

ggml_tensor * sparkinfer_layer_cache::build_reload_plan(ggml_context * ctx0,
                                                        ggml_tensor *  weight_only,
                                                        ggml_tensor *  cache_only) {
    ggml_tensor * result = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

    void * this_ptr = static_cast<void *>(this);
    memcpy(&(result->op_params[0]), static_cast<void *>(&this_ptr), sizeof(void *));

    result->op     = k_enable_spif_reload ? GGML_OP_RELOAD_PLAN : GGML_OP_VIEW;
    result->src[0] = weight_only;
    result->src[1] = cache_only;

    return result;
}

ggml_tensor * sparkinfer_layer_cache::build_reload_exec(ggml_context *         ctx0,
                                                        ggml_tensor *          cur,
                                                        sparkinfer_weight_type wt) {
    ggml_tensor * result = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

    result->op_params[0] = static_cast<int32_t>(wt);
    void * this_ptr      = static_cast<void *>(this);
    memcpy(&(result->op_params[1]), static_cast<void *>(&this_ptr), sizeof(void *));

    result->op     = k_enable_spif_reload ? GGML_OP_RELOAD_EXEC : GGML_OP_VIEW;
    result->src[0] = cur;

    return result;
}

void sparkinfer_layer_cache::sparkinfer_reload_plan() {
    float *   weight_only    = static_cast<float *>(weight_only_buf->data);
    float *   cache_only     = static_cast<float *>(cache_only_buf->data);
    int32_t * neuron_idx     = static_cast<int32_t *>(neuron_idx_buf->data);
    int32_t * group_maps     = static_cast<int32_t *>(this->group_maps->data);
    int32_t * neuron_mask    = static_cast<int32_t *>(this->neuron_mask->data);
    auto [n, m, g, n_g, m_g] = layer_cm;

    int weight_idx = 0;
    int cache_idx  = 0;
    reload_count   = 0;

    for (;;) {
        while (weight_idx < n_g && !weight_only[weight_idx]) {
            ++weight_idx;
        }
        while (cache_idx < n_g && !cache_only[cache_idx]) {
            ++cache_idx;
        }

        if (weight_idx >= n_g && cache_idx >= n_g) {
            break;
        }

        int32_t group_idx = group_maps[cache_idx];
        for (int k = 0; k < g; ++k) {
            neuron_mask[cache_idx * g + k] = 0;
        }
        group_maps[cache_idx] = -1;

        for (int k = 0; k < g; ++k) {
            neuron_mask[weight_idx * g + k] = 1;
            neuron_idx[group_idx * g + k]   = weight_idx * g + k;
        }
        group_maps[weight_idx] = group_idx;

        reload_plan[reload_count].weight_idx = weight_idx;
        reload_plan[reload_count].cache_idx  = group_idx;
        ++reload_count;

        ++weight_idx;
        ++cache_idx;
    }
}

sparkinfer_cache_manager::sparkinfer_cache_manager(const std::string & spif_ms_path,
                                                   llama_model &       model,
                                                   size_t              budget) {
    ggml_context *   ctx_meta    = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_meta,
    };
    gguf_context * ctx_gguf   = gguf_init_from_file(spif_ms_path.c_str(), gguf_params);
    const int32_t  group_size = gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "ffn_group_size"));
    const auto *   normalized_pattern =
        static_cast<const float *>(gguf_get_arr_data(ctx_gguf, gguf_find_key(ctx_gguf, "ffn_normalized_pattern")));

    ggml_init_params ctx_params = {
        /*.mem_size   = */ ggml_tensor_overhead() * 512,  // magic number here
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ctx_cpu = ggml_init(ctx_params);
    ctx_gpu = ggml_init(ctx_params);

    const auto & layers      = model.layers;
    const auto   n_layer     = model.hparams.n_layer;
    const auto   n_embd      = model.hparams.n_embd;
    const auto   n_ff        = model.hparams.n_ff(0);
    const auto   group_count = n_ff / group_size;
    const auto   type_size   = ggml_type_size(layers[0].ffn_up->type);
    is_gated_mlp             = !(layers[0].ffn_gate == nullptr);
    layer_caches             = std::vector<sparkinfer_layer_cache *>(n_layer);

    const auto       cache_budget = budget / (n_embd * type_size) / (is_gated_mlp ? 3 : 2);
    std::vector<int> cache_sizes(n_layer);
    for (size_t i = 0; i < cache_sizes.size(); ++i) {
        cache_sizes[i] =
            std::min((static_cast<uint32_t>(cache_budget * normalized_pattern[i] / group_size) * group_size), n_ff);
    }
    auto reorder_perms = std::vector<ggml_tensor *>(n_layer);

    // for simulation, we select some layers to store all their neurons
    auto cs_idx = std::vector<int>(n_layer);
    std::iota(cs_idx.begin(), cs_idx.end(), 0);
    std::nth_element(cs_idx.begin(), cs_idx.begin() + k_spif_gpu_only, cs_idx.end(),
                     [&](int i, int j) { return cache_sizes[i] > cache_sizes[j]; });
    std::for_each(cs_idx.begin(), cs_idx.begin() + k_spif_gpu_only, [&](int i) { cache_sizes[i] = n_ff; });

    auto create_tensor = [&](ggml_context * ctx, ggml_type type, std::vector<int64_t> ne, uint32_t il,
                             const char * name) {
        char tensor_name[GGML_MAX_NAME];
        std::snprintf(tensor_name, sizeof(tensor_name), "blk.%u.%s", il, name);
        ggml_tensor * tensor_meta = ggml_new_tensor(ctx, type, static_cast<int>(ne.size()), ne.data());
        return ggml_set_name(tensor_meta, tensor_name);
    };

    identity = create_tensor(ctx_gpu, GGML_TYPE_F32, { group_count, group_count }, 999, "identity");

    for (uint32_t il = 0; il < n_layer; ++il) {
        layer_caches[il] = new sparkinfer_layer_cache();

        auto * lc    = layer_caches[il];
        lc->layer_cm = {
            /*.n  = */ static_cast<int>(n_ff),
            /*.m  = */ static_cast<int>(cache_sizes[il]),
            /*.g  = */ static_cast<int>(group_size),
            /*.ng = */ static_cast<int>(n_ff / group_size),
            /*.mg = */ static_cast<int>(cache_sizes[il] / group_size),
        };
        lc->reload_plan = new copy_pair[lc->layer_cm.m]();
        lc->gpu_only    = (lc->layer_cm.m == lc->layer_cm.n);

        lc->ffn_pred_up     = layers[il].ffn_pred_up;
        lc->ffn_pred_down   = layers[il].ffn_pred_down;
        lc->ffn_pred_up_b   = layers[il].ffn_pred_up_b;
        lc->ffn_pred_down_b = layers[il].ffn_pred_down_b;

        lc->ffn_up     = layers[il].ffn_up;
        lc->ffn_gate   = layers[il].ffn_gate;
        lc->ffn_down   = layers[il].ffn_down_t;
        lc->ffn_up_b   = layers[il].ffn_up_b;
        lc->ffn_gate_b = layers[il].ffn_gate_b;
        lc->ffn_down_b = layers[il].ffn_down_b;

        lc->ffn_up_cache =
            create_tensor(ctx_gpu, layers[il].ffn_up->type, { n_embd, lc->layer_cm.m }, il, "ffn_up.cache");
        if (is_gated_mlp) {
            lc->ffn_gate_cache =
                create_tensor(ctx_gpu, layers[il].ffn_gate->type, { n_embd, lc->layer_cm.m }, il, "ffn_gate.cache");
        }
        lc->ffn_down_cache =
            create_tensor(ctx_gpu, layers[il].ffn_down_t->type, { n_embd, lc->layer_cm.m }, il, "ffn_down.cache");

        lc->neuron_idx  = create_tensor(ctx_gpu, GGML_TYPE_I32, { lc->layer_cm.m }, il, "ffn_neuron_idx");
        lc->group_maps  = create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->layer_cm.n_g }, il, "ffn_group_maps");
        lc->neuron_mask = create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->layer_cm.n }, il, "ffn_neuron_mask");
        lc->group_mask  = create_tensor(ctx_gpu, GGML_TYPE_F32, { lc->layer_cm.n_g }, il, "ffn_group_mask");
        lc->dfr_scores  = create_tensor(ctx_gpu, GGML_TYPE_F32, { lc->layer_cm.n_g }, il, "ffn_dfr_scores");

        lc->neuron_idx_buf  = create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->layer_cm.m }, il, "ffn_neuron_idx_buf");
        lc->weight_only_buf = create_tensor(ctx_cpu, GGML_TYPE_F32, { lc->layer_cm.n_g }, il, "ffn_weight_only_buf");
        lc->cache_only_buf  = create_tensor(ctx_cpu, GGML_TYPE_F32, { lc->layer_cm.n_g }, il, "ffn_cache_only_buf");
        lc->dfr_decay_pack  = create_tensor(ctx_cpu, GGML_TYPE_F32, { 3 }, il, "ffn_dfr_decay_data");

        reorder_perms[il] = create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->layer_cm.n }, il, "ffn_reorder_perms");
    }

    backend_cpu = ggml_backend_cpu_init();
    if (backend_cpu && ggml_get_first_tensor(ctx_cpu)) {
        buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cuda_host_buffer_type());
    }
    backend_gpu = ggml_backend_cuda_init(0);
    if (backend_gpu && ggml_get_first_tensor(ctx_gpu)) {
        buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);
    }

    for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
        const char *  name       = gguf_get_tensor_name(ctx_gguf, i);
        ggml_tensor * src_tensor = ggml_get_tensor(ctx_meta, name);
        ggml_tensor * dst_tensor = ggml_get_tensor(ctx_cpu, name);

        const auto nbytes = ggml_nbytes(src_tensor);
        ggml_backend_tensor_set(dst_tensor, src_tensor->data, 0, nbytes);
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);

    std::vector<float> f32_mat_buf(group_count * group_count);
    for (uint32_t i = 0; i < group_count; ++i) {
        f32_mat_buf[i * group_count + i] = 1.0f;
    }
    ggml_backend_tensor_set(identity, f32_mat_buf.data(), 0, ggml_nbytes(identity));

    std::vector<uint8_t> src_max_buf(n_embd * n_ff * type_size);
    std::vector<uint8_t> dst_max_buf(n_embd * n_ff * type_size);
    uint8_t *            src_buf               = src_max_buf.data();
    uint8_t *            dst_buf               = dst_max_buf.data();
    const static bool    k_enable_spif_reorder = (getenv("SPIF_REORDER") != nullptr);

    auto reorder_tensor_2d = [&](ggml_tensor * tensor, std::vector<int32_t> & perm) {
        const auto n_cols        = tensor->ne[0];
        const auto n_rows        = ggml_nrows(tensor);
        const auto row_size      = ggml_row_size(tensor->type, n_cols);
        const auto row_stride    = tensor->nb[1];
        const auto tensor_nbytes = ggml_nbytes(tensor);

        ggml_backend_tensor_get(tensor, src_buf, 0, tensor_nbytes);
        for (int new_row = 0; new_row < n_rows; ++new_row) {
            const auto old_row = perm[new_row];
            memcpy(dst_buf + new_row * row_stride, src_buf + old_row * row_stride, row_size);
        }
        ggml_backend_tensor_set(tensor, dst_buf, 0, tensor_nbytes);
    };
    auto reorder_tensor_1d = [&](ggml_tensor * tensor, std::vector<int32_t> & perm) {
        const auto n_elem        = tensor->ne[0];
        const auto elem_size     = ggml_row_size(tensor->type, 1);
        const auto elem_stride   = tensor->nb[0];
        const auto tensor_nbytes = ggml_nbytes(tensor);

        ggml_backend_tensor_get(tensor, src_buf, 0, tensor_nbytes);
        for (int new_i = 0; new_i < n_elem; ++new_i) {
            const auto old_i = perm[new_i];
            memcpy(dst_buf + new_i * elem_stride, src_buf + old_i * elem_stride, elem_size);
        }
        ggml_backend_tensor_set(tensor, dst_buf, 0, tensor_nbytes);
    };
    auto reorder_if_exists = [&](ggml_tensor * tensor, std::vector<int32_t> & perm) {
        if (sparkinfer_layer_cache::k_enable_spif_reload && k_enable_spif_reorder && tensor) {
            GGML_ASSERT(ggml_is_contiguous(tensor));
            if (tensor->ne[1] > 1) {
                reorder_tensor_2d(tensor, perm);
            } else {
                reorder_tensor_1d(tensor, perm);
            }
        }
    };

    auto perm_vec       = std::vector<int32_t>(n_ff);
    auto neuron_idx     = std::vector<int32_t>(n_ff);
    auto group_maps     = std::vector<int32_t>(n_ff, -1);
    auto neuron_mask    = std::vector<int32_t>(n_ff);
    auto group_mask     = std::vector<float>(n_ff);
    auto dfr_scores     = std::vector<float>(n_ff);
    auto dfr_decay_data = std::vector<float>({ 0.67f, 1.0f - 0.67f, 1.0f });

    for (uint32_t il = 0; il < n_layer; ++il) {
        auto * lc           = layer_caches[il];
        auto * reorder_perm = reorder_perms[il];

        ggml_backend_tensor_get(reorder_perm, perm_vec.data(), 0, ggml_nbytes(reorder_perm));
        reorder_if_exists(lc->ffn_pred_down, perm_vec);
        reorder_if_exists(lc->ffn_pred_down_b, perm_vec);
        reorder_if_exists(lc->ffn_up, perm_vec);
        reorder_if_exists(lc->ffn_up_b, perm_vec);
        if (is_gated_mlp) {
            reorder_if_exists(lc->ffn_gate, perm_vec);
            reorder_if_exists(lc->ffn_gate_b, perm_vec);
        }
        reorder_if_exists(lc->ffn_down, perm_vec);

        const auto cache_nbytes = ggml_nbytes(lc->ffn_up_cache);
        ggml_backend_tensor_set(lc->ffn_up_cache, lc->ffn_up->data, 0, cache_nbytes);
        if (is_gated_mlp) {
            ggml_backend_tensor_set(lc->ffn_gate_cache, lc->ffn_gate->data, 0, cache_nbytes);
        }
        ggml_backend_tensor_set(lc->ffn_down_cache, lc->ffn_down->data, 0, cache_nbytes);

        // [0, 1, ..., m-1]
        std::iota(neuron_idx.begin(), neuron_idx.begin() + lc->layer_cm.m, 0);
        // [0_0, 1_1, ..., (m/g)-1_(m/g)-1, -1_(m/g), ..., -1_(n/g)-1]
        std::fill_n(group_maps.begin(), lc->layer_cm.n_g, -1);
        std::iota(group_maps.begin(), group_maps.begin() + lc->layer_cm.m_g, 0);
        // [1_0, 1_1, ..., 1_m-1, 0_m, ..., 0_n-1]
        std::fill_n(neuron_mask.begin(), lc->layer_cm.n, 0);
        std::fill_n(neuron_mask.begin(), lc->layer_cm.m, 1);
        // [1_0, 1_1, ..., 1_(m/g)-1, 0_(m/g), ..., 0_(n/g)-1]
        std::fill_n(group_mask.begin(), lc->layer_cm.n_g, 0.0f);
        std::fill_n(group_mask.begin(), lc->layer_cm.m_g, 1.0f);
        // [0.0_0, 0.0_1, ..., 0.0_(m/g)-1, 0_(m/g), ..., 0_(n/g)-1]
        std::fill_n(dfr_scores.begin(), lc->layer_cm.n_g, 0.0f);

        ggml_backend_tensor_set(lc->neuron_idx, neuron_idx.data(), 0, ggml_nbytes(lc->neuron_idx));
        ggml_backend_tensor_set(lc->group_maps, group_maps.data(), 0, ggml_nbytes(lc->group_maps));
        ggml_backend_tensor_set(lc->neuron_mask, neuron_mask.data(), 0, ggml_nbytes(lc->neuron_mask));
        ggml_backend_tensor_set(lc->group_mask, group_mask.data(), 0, ggml_nbytes(lc->group_mask));
        ggml_backend_tensor_set(lc->dfr_scores, dfr_scores.data(), 0, ggml_nbytes(lc->dfr_scores));
        ggml_backend_tensor_set(lc->dfr_decay_pack, dfr_decay_data.data(), 0, ggml_nbytes(lc->dfr_decay_pack));

        const auto cache_n_mega_bytes = (cache_nbytes * (is_gated_mlp ? 3 : 2)) / (1024.0 * 1024.0);
        LLAMA_LOG_INFO("%s: [layer %2u] offloaded %6.2f MiB and cached %5d (%6.2f%%) neurons to GPU\n", __func__, il,
                       cache_n_mega_bytes, lc->layer_cm.m, lc->layer_cm.m * 100.0 / lc->layer_cm.n);
    }
    LLAMA_LOG_INFO("%s: the cache manger has totally %.2f MiB GPU memory footprint\n", __func__,
                   ggml_backend_buffer_get_size(buf_gpu) / (1024.0 * 1024.0));
}

sparkinfer_cache_manager::~sparkinfer_cache_manager() {
    auto format_array = [&](const float * data, size_t n) {
        std::string out = "[";
        char        buf[32];
        for (size_t i = 0; i < n; ++i) {
            snprintf(buf, sizeof(buf), "%.3f", data[i]);
            out += buf;
            if (i + 1 < n) {
                out += ", ";
            }
        }
        out += "]";
        return out;
    };

    std::vector<float> dfr_decays(layer_caches.size(), 0.0f);
    for (size_t il = 0; il < layer_caches.size(); ++il) {
        auto * lc      = layer_caches[il];
        dfr_decays[il] = static_cast<const float *>(lc->dfr_decay_pack->data)[0];

        delete[] lc->reload_plan;
        delete lc;
    }
    LLAMA_LOG_INFO("%s: final dfr decays are %s\n", __func__,
                   format_array(dfr_decays.data(), layer_caches.size()).c_str());

    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx_gpu);
    ggml_backend_free(backend_gpu);

    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_cpu);
    ggml_backend_free(backend_cpu);
}
