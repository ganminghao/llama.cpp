#pragma once

#include "ggml-sparkinfer.hpp"
#include "llama.h"

#include <cstdlib>
#include <vector>

struct sparkinfer_cache_manager {
    std::vector<sparkinfer_layer_cache *> layer_caches;

    ggml_context *        ctx_cpu     = nullptr;
    ggml_context *        ctx_gpu     = nullptr;
    ggml_backend_t        backend_cpu = nullptr;
    ggml_backend_t        backend_gpu = nullptr;
    ggml_backend_buffer_t buf_cpu     = nullptr;
    ggml_backend_buffer_t buf_gpu     = nullptr;

    bool          has_ffn_gate   = false;
    ggml_tensor * group_identity = nullptr;

    sparkinfer_cache_manager(llama_model * model, const char * spif_ms_path, int64_t vram_budget);
    ~sparkinfer_cache_manager();
};
