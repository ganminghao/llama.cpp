#pragma once

#include "llama.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml-cpp.h"

#include <vector>
#include <queue>
#include <unordered_map>
#include <deque>
#include <functional>
#include <cstring> // for memcpy
#include <utility> // for std::pair
#include <cstdint>
#include <ggml-cuda.h>

// Using forward declaration, rather than including the whole head file, is recommended when you use only the pointers of these classed instead of their instances.
struct llama_cparams;
struct llama_model;
struct llama_context;

struct sparkinfer_reload_plan {
    std::vector<int64_t> groups_to_reload;
    std::vector<int64_t> slots_for_reload;
    std::vector<int64_t> new_gpu_neu_idx_vec;
    std::vector<int64_t> new_gpu_neu_mask_vec;

    // [YPX] [C] 我决定不用 ggml_tensor *，还是用 vector；到最后传入 GGML_OP_RELOAD 的时候再搞成 ggml_tensor *
    // ggml_tensor * new_gpu_neu_idx_cpu;
    // ggml_tensor * new_gpu_neu_mask_cpu;

    // Constructor
    sparkinfer_reload_plan(
        const std::vector<int64_t>& reload_groups,
        const std::vector<int64_t>& reload_slots,
        const std::vector<int64_t>& new_neu_idx,
        const std::vector<int64_t>& new_neu_mask):  
        groups_to_reload(reload_groups),
        slots_for_reload(reload_slots),
        new_gpu_neu_idx_vec(new_neu_idx),
        new_gpu_neu_mask_vec(new_neu_mask){};
};

struct layer_cache{

public:

    layer_cache(const llama_layer & layer, int64_t layer_idx, double dfr_decay, double dfr_bonus);
    ~layer_cache() = default;

    /* [YPX] [C] cache_manager 和 layer_cache 根本不管理这些变量，不需要也不应该在这里保存
    ggml_tensor * get_cpu_gate(ggml_context * ctx);
    ggml_tensor * get_cpu_up(ggml_context * ctx);
    ggml_tensor * get_cpu_down_t(ggml_context * ctx);

    ggml_tensor * get_gpu_gate(ggml_context * ctx);
    ggml_tensor * get_gpu_up(ggml_context * ctx);
    ggml_tensor * get_gpu_down_t(ggml_context * ctx);

    ggml_tensor * get_gpu_neu_idx(ggml_context * ctx);
    ggml_tensor * get_gpu_neu_mask(ggml_context * ctx);

    ggml_tensor * reload_gate(ggml_context * ctx);
    ggml_tensor * reload_up(ggml_context * ctx);
    ggml_tensor * reload_down_t(ggml_context * ctx);
    */

    ggml_tensor * get_group_to_neurons_map(ggml_context);

    sparkinfer_reload_plan plan_reload(ggml_tensor * sparse_idx);

private:

    int64_t layer_idx   = -1;
    bool  full_gpu    = false;

    /* [YPX] [C] 同上，这些变量没有必要了，因为 cache_manager 和 layer_cache 只负责给出 reload 方案，而不负责具体的 reload 行为
    // full ffn weights and biases on cpu
    ggml_tensor * cpu_ffn_gate                  = nullptr;
    ggml_tensor * cpu_ffn_up                    = nullptr;
    ggml_tensor * cpu_ffn_down_t                = nullptr;
    ggml_tensor * cpu_ffn_gate_b                = nullptr;
    ggml_tensor * cpu_ffn_up_b                  = nullptr;
    ggml_tensor * cpu_ffn_down_b                = nullptr;

    // cached ffn weights and biases on gpu
    ggml_tensor * gpu_ffn_gate                  = nullptr;
    ggml_tensor * gpu_ffn_up                    = nullptr;
    ggml_tensor * gpu_ffn_down_t                = nullptr;
    ggml_tensor * gpu_ffn_gate_b                = nullptr;
    ggml_tensor * gpu_ffn_up_b                  = nullptr;
    ggml_tensor * gpu_ffn_down_b                = nullptr;
    */

    ggml_tensor * ffn_gpu_neu_idx             = nullptr;
    ggml_tensor * ffn_gpu_neu_mask            = nullptr;

    std::vector<int64_t> gpu_neu_idx_vec;
    std::vector<int64_t> gpu_neu_mask_vec;

    int64_t layer_neuron_count;     // number of neurons in the layer
    int64_t layer_group_count;      // number of groups in the layer
    int64_t layer_group_size;       // number of neurons in each group
    int64_t layer_neuron_capacity;  // number of neurons that can be cached on GPU
    int64_t layer_group_capacity;   // number of groups that can be cached on GPU

    // groups and neurons mapping structures
    ggml_tensor * group_to_neurons_map_tensor      = nullptr; // not used in layer_cache but stored here, only used by GGML_OP_RELOAD node for doing memcpy.
    std::vector<std::vector<int64_t>> group_to_neurons_map; // used in update_idx_and_mask()
    std::vector<int64_t> neuron_to_group_map; // used in update_activated_group_mask()

    // activated group mask, used only inside layer_cache
    std::vector<int64_t> activated_group_mask;

    // DFR related structures
    double dfr_decay = 0.75;
    double dfr_bonus = 1.00;

    std::vector<double> group_dfr_tracker;

    // [YPX] [Q] 这么写可以吗？
    using DFR_Node = std::pair<double, int64_t>;

    std::priority_queue<DFR_Node, std::vector<DFR_Node>, std::greater<DFR_Node>> group_dfr_heap;

    // slots related structures
    std::vector<int64_t>        group_to_slot_map;
    std::vector<int64_t>        slot_to_group_map;
    std::vector<int64_t>        groups_to_reload;
    std::vector<int64_t>        slots_for_reload;

    // utility functions for generating reload plan
    bool update_activated_group_mask(const std::vector<int64_t>& sparse_idx);
    bool update_DFR();
    bool update_slots();
    bool update_idx_and_mask();
    sparkinfer_reload_plan generate_reload_plan();
    
};


struct sparkinfer_cache_manager: public llama_memory_i{
    sparkinfer_cache_manager(const llama_model & model, const llama_cparams & cparams);
    ~sparkinfer_cache_manager() = default;

    // [YPX] [C] 这些函数直接设置为默认实现，因为因为不需要用到它们
    // override the function in llama-memory-i
    void clear() override {}
    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override { return true; }
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override {}
    void seq_keep(llama_seq_id seq_id)                                                          override {}
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override {}
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int64_t d) override {}
    
    llama_pos seq_pos_min(llama_seq_id seq_id) const override { return 0; }
    llama_pos seq_pos_max(llama_seq_id seq_id) const override { return 0; }
   
    bool get_can_edit() const override { return false; }

    layer_cache * get_layer_cache(int64_t layer_idx) const{
        if(layer_idx < 0 || layer_idx >= n_layer){
            return nullptr;
        }
        return layer_caches[layer_idx];
    }

private:
    int64_t n_layer = 0;
    double dfr_decay = 0.75;
    double dfr_bonus = 1.00;
    std::vector<layer_cache*> layer_caches;
};