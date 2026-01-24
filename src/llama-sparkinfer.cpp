#include "llama-sparkinfer.h"

#include "ggml-cuda.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <cstring>
#include <memory>
#include <numeric>

const bool sparkinfer_layer_cache::k_enable_spif_reload =
    (getenv("SPIF_PARALLEL") != nullptr && getenv("SPIF_RELOAD") != nullptr);

ggml_tensor * sparkinfer_layer_cache::build_reload_plan(ggml_context * ctx0,
                                                        ggml_tensor *  weight_only,
                                                        ggml_tensor *  cache_only) {
    ggml_tensor * result = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

    void * this_ptr = (void *) this;
    memcpy(&(result->op_params[0]), (void *) &this_ptr, sizeof(void *));

    result->op     = k_enable_spif_reload ? GGML_OP_RELOAD_PLAN : GGML_OP_VIEW;
    result->src[0] = weight_only;
    result->src[1] = cache_only;

    return result;
}

ggml_tensor * sparkinfer_layer_cache::build_reload_exec(ggml_context *         ctx0,
                                                        ggml_tensor *          cur,
                                                        sparkinfer_weight_type wt) {
    ggml_tensor * result = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

    result->op_params[0] = (int32_t) wt;
    void * this_ptr      = (void *) this;
    memcpy(&(result->op_params[1]), (void *) &this_ptr, sizeof(void *));

    result->op     = k_enable_spif_reload ? GGML_OP_RELOAD_EXEC : GGML_OP_VIEW;
    result->src[0] = cur;

    return result;
}

void sparkinfer_layer_cache::sparkinfer_reload_plan() {
    float *   weight_only    = (float *) weight_only_buf->data;
    float *   cache_only     = (float *) cache_only_buf->data;
    int32_t * neuron_idx     = (int32_t *) neuron_idx_buf->data;
    int32_t * group_maps     = (int32_t *) this->group_maps->data;
    int32_t * neuron_mask    = (int32_t *) this->neuron_mask->data;
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

    reload_records.push_back(reload_count * (ffn_gate ? 3 : 2));
    dfr_decay_records.push_back(*(float *) dfr_decay_pack->data);
}

void sparkinfer_init_from_model_and_ctx(llama_model *   tgt_model,
                                        llama_context * tgt_ctx,
                                        llama_model *   dft_model,
                                        llama_context * dft_ctx,
                                        const char *    spif_ms_path,
                                        int64_t         vram_budget) {
    if (spif_ms_path == nullptr || *spif_ms_path == '\0') {
        return;
    }

    (void) dft_model;
    size_t free_bytes;
    size_t total_bytes;
    ggml_backend_cuda_get_device_memory(0, &free_bytes, &total_bytes);

    if (vram_budget == 0) {
        vram_budget = free_bytes;
    } else if (vram_budget > 0) {
        const int64_t budget_bytes = vram_budget * 1024 * 1024 * 1024;
        int64_t       used_bytes   = 0;
        for (auto * ctx : { tgt_ctx, dft_ctx }) {
            if (ctx != nullptr) {
                for (auto & buft_size : ctx->memory_breakdown()) {
                    if (ggml_backend_buft_is_host(buft_size.first)) {
                        continue;
                    }
                    used_bytes += buft_size.second.model;
                    used_bytes += buft_size.second.context;
                    used_bytes += buft_size.second.compute;
                }
            }
        }
        vram_budget = std::min<int64_t>(budget_bytes - used_bytes, free_bytes);
    } else {
        GGML_ABORT("fatal error");
    }
    vram_budget -= 384 * (1024 * 1024);
    GGML_ASSERT(vram_budget > 0 && "no vram left for initializing cache manager");

    tgt_ctx->spif_cm = std::make_unique<sparkinfer_cache_manager>(tgt_model, spif_ms_path, vram_budget);
}

sparkinfer_cache_manager::sparkinfer_cache_manager(llama_model * model,
                                                   const char *  spif_ms_path,
                                                   int64_t       vram_budget) {
    ggml_context *   ctx_meta    = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_meta,
    };
    gguf_context * ctx_gguf    = gguf_init_from_file(spif_ms_path, gguf_params);
    const int32_t  n_ffn_group = gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "ffn_group_size"));
    const float *  ffn_norm_pattern =
        (const float *) gguf_get_arr_data(ctx_gguf, gguf_find_key(ctx_gguf, "ffn_normalized_pattern"));

    ggml_init_params ctx_params = {
        /*.mem_size   = */ ggml_tensor_overhead() * 512,  // magic number here
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ctx_cpu = ggml_init(ctx_params);
    ctx_gpu = ggml_init(ctx_params);

    const auto & layers  = model->layers;
    const int    n_layer = model->hparams.n_layer;
    const int    n_embd  = model->hparams.n_embd;
    const int    n_ff    = model->hparams.n_ff(0);

    layer_caches.resize(n_layer);
    const bool k_enable_spif_reorder = (getenv("SPIF_REORDER") != nullptr);
    if (k_enable_spif_reorder) {
        reorder_perms.resize(n_layer);
    }

    auto layer_group_bytes = [&](const llama_layer & layer) {
        size_t bytes = 0;
        bytes += ggml_row_size(layer.ffn_up->type, n_embd) * n_ffn_group;
        if (layer.ffn_gate) {
            bytes += ggml_row_size(layer.ffn_gate->type, n_embd) * n_ffn_group;
        }
        bytes += ggml_row_size(layer.ffn_down_t->type, n_embd) * n_ffn_group;
        return bytes;
    };
    const auto       n_group = n_ff / n_ffn_group;
    std::vector<int> n_group_cache(n_layer, 0);
    std::vector<int> n_bytes_group(n_layer, 0);
    double           n_bytes_group_avg = 0.0;
    for (int il = 0; il < n_layer; ++il) {
        n_bytes_group[il] = layer_group_bytes(layers[il]);
        n_bytes_group_avg += double(n_bytes_group[il]) * ffn_norm_pattern[il];
    }
    const int n_group_cache_budget = std::min<int>(vram_budget / n_bytes_group_avg, n_layer * n_group);
    int       n_group_cache_used   = 0;
    for (int il = 0; il < n_layer; ++il) {
        n_group_cache[il] = std::min<int>(n_group_cache_budget * ffn_norm_pattern[il], n_group);
        n_group_cache_used += n_group_cache[il];
    }
    for (int n_group_cache_left = n_group_cache_budget - n_group_cache_used; n_group_cache_left > 0;) {
        int before_rr = n_group_cache_left;
        for (int il = 0; il < n_layer && n_group_cache_left > 0; ++il) {
            if (n_group_cache[il] < n_group) {
                ++n_group_cache[il];
                --n_group_cache_left;
            }
        }
        if (n_group_cache_left == before_rr) {
            break;
        }
    }

    auto create_tensor = [&](ggml_context * ctx, ggml_type type, std::vector<int64_t> ne, int il, const char * name) {
        char tensor_name[GGML_MAX_NAME];
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.%s", il, name);
        ggml_tensor * tensor_meta = ggml_new_tensor(ctx, type, (int) ne.size(), ne.data());
        return ggml_set_name(tensor_meta, tensor_name);
    };

    for (int il = 0; il < n_layer; ++il) {
        layer_caches[il] = new sparkinfer_layer_cache();

        auto * lc    = layer_caches[il];
        lc->layer_cm = {
            /*.n  = */ (int) n_ff,
            /*.m  = */ (int) n_group_cache[il] * n_ffn_group,
            /*.g  = */ (int) n_ffn_group,
            /*.ng = */ (int) n_ff / n_ffn_group,
            /*.mg = */ (int) n_group_cache[il],
        };
        lc->reload_plan = new copy_pair[lc->layer_cm.m]();
        lc->gpu_only    = (lc->layer_cm.m == lc->layer_cm.n);

        lc->ffn_pred_up     = layers[il].ffn_pred_up;
        lc->ffn_pred_down   = layers[il].ffn_pred_down;
        lc->ffn_pred_up_b   = layers[il].ffn_pred_up_b;
        lc->ffn_pred_down_b = layers[il].ffn_pred_down_b;

        lc->ffn_up       = layers[il].ffn_up;
        lc->ffn_gate     = layers[il].ffn_gate;
        lc->ffn_down     = layers[il].ffn_down_t;
        lc->ffn_up_b     = layers[il].ffn_up_b;
        lc->ffn_gate_b   = layers[il].ffn_gate_b;
        lc->ffn_down_b   = layers[il].ffn_down_b;
        lc->ffn_up_cache = create_tensor(ctx_gpu, lc->ffn_up->type, { n_embd, lc->layer_cm.m }, il, "ffn_up.cache");
        if (lc->ffn_gate) {
            lc->ffn_gate_cache =
                create_tensor(ctx_gpu, lc->ffn_gate->type, { n_embd, lc->layer_cm.m }, il, "ffn_gate.cache");
        }
        lc->ffn_down_cache =
            create_tensor(ctx_gpu, lc->ffn_down->type, { n_embd, lc->layer_cm.m }, il, "ffn_down.cache");

        lc->neuron_idx  = create_tensor(ctx_gpu, GGML_TYPE_I32, { lc->layer_cm.m }, il, "ffn_neuron_idx");
        lc->group_maps  = create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->layer_cm.n_g }, il, "ffn_group_maps");
        lc->neuron_mask = create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->layer_cm.n }, il, "ffn_neuron_mask");
        lc->group_mask  = create_tensor(ctx_gpu, GGML_TYPE_F32, { lc->layer_cm.n_g }, il, "ffn_group_mask");
        lc->dfr_scores  = create_tensor(ctx_gpu, GGML_TYPE_F32, { lc->layer_cm.n_g }, il, "ffn_dfr_scores");

        lc->neuron_idx_buf  = create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->layer_cm.m }, il, "ffn_neuron_idx_buf");
        lc->weight_only_buf = create_tensor(ctx_cpu, GGML_TYPE_F32, { lc->layer_cm.n_g }, il, "ffn_weight_only_buf");
        lc->cache_only_buf  = create_tensor(ctx_cpu, GGML_TYPE_F32, { lc->layer_cm.n_g }, il, "ffn_cache_only_buf");
        lc->dfr_decay_pack  = create_tensor(ctx_cpu, GGML_TYPE_F32, { 3 }, il, "ffn_dfr_decay_data");

        if (k_enable_spif_reorder) {
            reorder_perms[il] = create_tensor(ctx_cpu, GGML_TYPE_I32, { lc->layer_cm.n }, il, "ffn_reorder_perms");
        }
    }
    group_identity = create_tensor(ctx_gpu, GGML_TYPE_F32, { n_group, n_group }, 999, "ffn_group_identity");

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

        if (src_tensor != nullptr && dst_tensor != nullptr) {
            ggml_backend_tensor_set(dst_tensor, src_tensor->data, 0, ggml_nbytes(dst_tensor));
        }
    }
    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);

    std::vector<float> f32_mat_buf(n_group * n_group);
    for (int i = 0; i < n_group; ++i) {
        f32_mat_buf[i * n_group + i] = 1.0f;
    }
    ggml_backend_tensor_set(group_identity, f32_mat_buf.data(), 0, ggml_nbytes(group_identity));

    std::vector<uint8_t> buf_reorder_src(ggml_row_size(GGML_TYPE_F32, n_embd) * n_ff);
    std::vector<uint8_t> buf_reorder_dst(ggml_row_size(GGML_TYPE_F32, n_embd) * n_ff);
    auto *               src_buf = buf_reorder_src.data();
    auto *               dst_buf = buf_reorder_dst.data();

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
        if (tensor) {
            GGML_ASSERT(ggml_is_contiguous(tensor));
            if (tensor->ne[1] > 1) {
                reorder_tensor_2d(tensor, perm);
            } else {
                reorder_tensor_1d(tensor, perm);
            }
        }
    };

    std::vector<int32_t> perm_vec(n_ff);
    std::vector<int32_t> neuron_idx(n_ff);
    std::vector<int32_t> group_maps(n_ff);
    std::vector<int32_t> neuron_mask(n_ff);
    std::vector<float>   group_mask(n_ff);
    std::vector<float>   dfr_scores(n_ff);
    std::vector<float>   dfr_decay_pack({ k_spif_init_dfr_decay, 1.0f - k_spif_init_dfr_decay, 1.0f });

    for (int il = 0; il < n_layer; ++il) {
        auto * lc = layer_caches[il];

        if (k_enable_spif_reorder) {
            auto * reorder_perm = reorder_perms[il];
            ggml_backend_tensor_get(reorder_perm, perm_vec.data(), 0, ggml_nbytes(reorder_perm));
            reorder_if_exists(lc->ffn_pred_down, perm_vec);
            reorder_if_exists(lc->ffn_pred_down_b, perm_vec);
            reorder_if_exists(lc->ffn_up, perm_vec);
            reorder_if_exists(lc->ffn_up_b, perm_vec);
            reorder_if_exists(lc->ffn_gate, perm_vec);
            reorder_if_exists(lc->ffn_gate_b, perm_vec);
            reorder_if_exists(lc->ffn_down, perm_vec);
            reorder_if_exists(lc->ffn_down_b, perm_vec);
        }

        size_t cache_nbytes = 0;
        ggml_backend_tensor_set(lc->ffn_up_cache, lc->ffn_up->data, 0, ggml_nbytes(lc->ffn_up_cache));
        cache_nbytes += ggml_nbytes(lc->ffn_up_cache);
        if (lc->ffn_gate) {
            ggml_backend_tensor_set(lc->ffn_gate_cache, lc->ffn_gate->data, 0, ggml_nbytes(lc->ffn_gate_cache));
            cache_nbytes += ggml_nbytes(lc->ffn_gate_cache);
        }
        ggml_backend_tensor_set(lc->ffn_down_cache, lc->ffn_down->data, 0, ggml_nbytes(lc->ffn_down_cache));
        cache_nbytes += ggml_nbytes(lc->ffn_down_cache);

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
        ggml_backend_tensor_set(lc->dfr_decay_pack, dfr_decay_pack.data(), 0, ggml_nbytes(lc->dfr_decay_pack));

        const double cache_n_mega_bytes = cache_nbytes / (1024.0 * 1024.0);
        LLAMA_LOG_INFO("%s: [layer %2d] offloaded %6.2f MiB and cached %5d (%6.2f%%) neurons to device\n", __func__, il,
                       cache_n_mega_bytes, lc->layer_cm.m, lc->layer_cm.m * 100.0 / lc->layer_cm.n);
    }
    LLAMA_LOG_INFO("%s: the cache manger has totally %.2f MiB GPU memory footprint\n", __func__,
                   ggml_backend_buffer_get_size(buf_gpu) / (1024.0 * 1024.0));
}

sparkinfer_cache_manager::~sparkinfer_cache_manager() {
    auto format_container = [&](const auto & container, const char * fmt) {
        std::string out = "[";
        char        buf[32];
        size_t      i = 0;
        for (auto & v : container) {
            snprintf(buf, sizeof(buf), fmt, v);
            out += buf;
            if (++i < container.size()) {
                out += ", ";
            }
        }
        out += "]";
        return out;
    };

    const auto         n_layer = layer_caches.size();
    std::vector<float> dfr_decays(n_layer, 0.0f);
    for (size_t il = 0; il < n_layer; ++il) {
        auto * lc      = layer_caches[il];
        dfr_decays[il] = ((const float *) lc->dfr_decay_pack->data)[0];

        if (k_spif_dfr_debug > 1) {
            LLAMA_LOG_INFO("%s: [layer %2zu] reload_records = %s\n", __func__, il,
                           format_container(lc->reload_records, "%d").c_str());
            LLAMA_LOG_INFO("%s: [layer %2zu] dfr_decay_records = %s\n", __func__, il,
                           format_container(lc->dfr_decay_records, "%.3f").c_str());
        }

        delete[] lc->reload_plan;
        delete lc;
    }
    if (k_spif_dfr_debug > 0) {
        LLAMA_LOG_INFO("%s: final dfr decays (for all layers) are %s\n", __func__,
                       format_container(dfr_decays, "%.3f").c_str());
    }

    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx_gpu);
    ggml_backend_free(backend_gpu);

    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_cpu);
    ggml_backend_free(backend_cpu);
}
