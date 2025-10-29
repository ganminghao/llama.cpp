#pragma once

#include "ggml-sparkinfer.hpp"
#include "llama-model.h"

struct sparkinfer_cache_manager {
    std::vector<sparkinfer_layer_cache *> layer_caches;

    ggml_context *        ctx_cpu     = nullptr;
    ggml_context *        ctx_gpu     = nullptr;
    ggml_backend_t        backend_cpu = nullptr;
    ggml_backend_t        backend_gpu = nullptr;
    ggml_backend_buffer_t buf_cpu     = nullptr;
    ggml_backend_buffer_t buf_gpu     = nullptr;

    float         dfr_decay    = 0.75f;
    bool          is_gated_mlp = false;
    ggml_tensor * identity     = nullptr;

    sparkinfer_cache_manager(const std::string & spif_ms_path, llama_model & model);
    ~sparkinfer_cache_manager();
};
