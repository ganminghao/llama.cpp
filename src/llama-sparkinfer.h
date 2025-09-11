#pragma once

#include "llama.h"
#include "llama-memory.h"

#include "ggml-cpp.h"

#include <vector>

struct llama_cparams;
struct llama_model;
struct llama_context;

#include <vector>

struct layer_cache{
    layer_cache();
    ~layer_cache() = default;

    ggml_tensor * get_cpu_gate(ggml_context * ctx);
    ggml_tensor * get_cpu_up(ggml_context * ctx);
    ggml_tensor * get_cpu_down_t(ggml_context * ctx);

    ggml_tensor * get_gpu_gate(ggml_context * ctx);
    ggml_tensor * get_gpu_up(ggml_context * ctx);
    ggml_tensor * get_gpu_down_t(ggml_context * ctx);

    ggml_tensor * get_gpu_neu_idx(ggml_context * ctx);
    ggml_tensor * get_gpu_neu_mask(ggml_context * ctx);
    ggml_tensor * get_DFR(ggml_context * ctx);

    ggml_tensor * reload_gate(ggml_context * ctx);
    ggml_tensor * reload_up(ggml_context * ctx);
    ggml_tensor * reload_down_t(ggml_context * ctx);

private:

    int     layer_idx                               = -1;
    bool    full_gpu                                = false;

    // full ffn weights and biases on cpu
    ggml_tensor * cpu_ffn_gate                = nullptr;
    ggml_tensor * cpu_ffn_up                  = nullptr;
    ggml_tensor * cpu_ffn_down_t              = nullptr;

    // cached ffn weights and biases on gpu
    ggml_tensor * gpu_ffn_gate                = nullptr;
    ggml_tensor * gpu_ffn_up                  = nullptr;
    ggml_tensor * gpu_ffn_down_t              = nullptr;

    ggml_tensor * ffn_gpu_neu_idx             = nullptr;
    ggml_tensor * ffn_gpu_neu_mask            = nullptr;

    // DFR Related
    const double dfr_decay              = 0.95;
    ggml_tensor * DFR                    = nullptr;
};


struct sparkinfer_cache_manager: public llama_memory_i{
    sparkinfer_cache_manager(const llama_model & model, const llama_cparams & cparams);
    ~sparkinfer_cache_manager() = default;

    // override the function in llama-memory-i
    void clear() override;
    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;
    
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
   
    bool get_can_edit() const override;

    layer_cache * get_layer_cache(int layer_idx) const{
        if(layer_idx < 0 || layer_idx >= n_layer){
            return nullptr;
        }
        return layer_caches[layer_idx];
    }

private:
    int64_t n_layer = 0;
    std::vector<layer_cache*> layer_caches;
};