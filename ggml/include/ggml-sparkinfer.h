#pragma once

#include "ggml.h"

enum sparkinfer_weight_type { SPIF_FFN_UP = 0, SPIF_FFN_GATE = 1, SPIF_FFN_DOWN = 2 };

typedef struct {
    int n, m, g;   // neuron_count, neuron_cache_size, group_size
    int n_g, m_g;  // group_count, group_cache_size
} cache_meta;

typedef struct {
    int weight_idx, cache_idx;
} copy_pair;

struct sparkinfer_layer_cache {
    ggml_tensor * layer_ffn_pred_up     = nullptr;
    ggml_tensor * layer_ffn_pred_down   = nullptr;
    ggml_tensor * layer_ffn_pred_up_b   = nullptr;
    ggml_tensor * layer_ffn_pred_down_b = nullptr;

    ggml_tensor * layer_ffn_up     = nullptr;
    ggml_tensor * layer_ffn_gate   = nullptr;
    ggml_tensor * layer_ffn_down   = nullptr;
    ggml_tensor * layer_ffn_up_b   = nullptr;
    ggml_tensor * layer_ffn_gate_b = nullptr;
    ggml_tensor * layer_ffn_down_b = nullptr;

    ggml_tensor * ffn_up_cache   = nullptr;
    ggml_tensor * ffn_gate_cache = nullptr;
    ggml_tensor * ffn_down_cache = nullptr;

    ggml_tensor * sparse_idx  = nullptr;
    ggml_tensor * reload_up   = nullptr;
    ggml_tensor * reload_gate = nullptr;
    ggml_tensor * reload_down = nullptr;

    ggml_tensor * neuron_idx  = nullptr;
    ggml_tensor * group_maps  = nullptr;
    ggml_tensor * neuron_mask = nullptr;
    ggml_tensor * group_mask  = nullptr;
    ggml_tensor * dfr_scores  = nullptr;

    cache_meta  layer_cm;
    copy_pair * reload_plan;
    int         num_ops;

    ggml_tensor * weight_only_buf;
    ggml_tensor * cache_only_buf;
    ggml_tensor * neuron_idx_buf;

    sparkinfer_layer_cache()  = default;
    ~sparkinfer_layer_cache() = default;

    ggml_tensor * build_reload(ggml_context *         ctx0,
                               ggml_tensor *          weight_only,
                               ggml_tensor *          cache_only,
                               sparkinfer_weight_type spif_wt);
    void          sparkinfer_reload_plan(const float * weight_only, const float * cache_only, int32_t * neuron_idx);
};
