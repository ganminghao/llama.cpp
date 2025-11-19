#pragma once

#include "ggml-sparkinfer.hpp"
#include "llama-model.h"

#include <cstdlib>

inline int get_env_int(const char * env, int default_value) {
    if (const char * p = getenv(env)) {
        return atoi(p);
    }
    return default_value;
}

const static int k_dfr_decay = get_env_int("DFR_DECAY", 69);

struct sparkinfer_cache_manager {
    std::vector<sparkinfer_layer_cache *> layer_caches;

    ggml_context *        ctx_cpu     = nullptr;
    ggml_context *        ctx_gpu     = nullptr;
    ggml_backend_t        backend_cpu = nullptr;
    ggml_backend_t        backend_gpu = nullptr;
    ggml_backend_buffer_t buf_cpu     = nullptr;
    ggml_backend_buffer_t buf_gpu     = nullptr;

    float         dfr_decay    = k_dfr_decay / 100.0f;
    bool          is_gated_mlp = false;
    ggml_tensor * identity     = nullptr;

    sparkinfer_cache_manager(const std::string & spif_ms_path, llama_model & model);
    ~sparkinfer_cache_manager();
};
