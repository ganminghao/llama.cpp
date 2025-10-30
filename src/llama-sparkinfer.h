#pragma once

#include "llama-model.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
#include "ggml-spif.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <ggml-backend.h>
#include <numeric>
#include <ggml-cuda.h>
#include <list>
#include <omp.h>
#include <inttypes.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <list>  // LRU list
#include <queue> // DFR priority_queue
#include <set>   // set difference

enum class sparkinfer_reload_strategy : int {
    USE_DFR = 0,
    USE_LRU = 1,
    USE_FIFO = 2,
    USE_OPT = 3,
    USE_CLOCK = 4,
    USE_NRU = 5,
    USE_PLACEHOLDER = 6,
};

// ==================================================================================================
//                                      !!! 全局唯一的控制开关 !!!
//                          通过修改这个值，然后重新编译，即可切换所有相关算子的行为
constexpr sparkinfer_reload_strategy SPARKINFER_RELOAD_STRATEGY = sparkinfer_reload_strategy::USE_DFR;
// ==================================================================================================

/*
* @brief 管理单个层中FFN权重神经元组的GPU缓存。
*/
struct sparkInfer_layer_cache {

public:

    // --- 后端与上下文 ---
    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    struct ggml_context* tmp_ctx = nullptr; // 用于创建临时视图

    // --- CPU 权重 (数据源) ---
    struct ggml_tensor * cpu_ffn_gate   = nullptr;
    struct ggml_tensor * cpu_ffn_up     = nullptr;
    struct ggml_tensor * cpu_ffn_down_t = nullptr;

    // --- GPU 缓存池 (目标) ---
    ggml_backend_buffer_t gpu_weights_buffer = nullptr;
    struct ggml_tensor* gpu_ffn_gate_cache   = nullptr;
    struct ggml_tensor* gpu_ffn_up_cache     = nullptr;
    struct ggml_tensor* gpu_ffn_down_t_cache = nullptr;

    // --- 稀疏预测相关张量 ---
    struct ggml_tensor * sparse_idx         = nullptr; // 在 GPU 上, 每次 plan 前需要从 GPU 同步
    struct ggml_tensor * ffn_gpu_neu_idx    = nullptr; // 在 GPU 上
    struct ggml_tensor * ffn_gpu_neu_mask   = nullptr; // 在 GPU 上

    // --- (待清理) 多 GPU 遗留? ---
    // [YPX] [Todo] 这很可能是多GPU split用的，待团队确认是否可移除
    std::vector<ggml_backend_buffer_t> split_idx_allocated_buffers;

    // --- 核心超参数 ---
    int layer_neuron_count     = 0; // 每层神经元总量
    int layer_group_count      = 0; // 每层分组总量
    int layer_group_size       = 0; // 每层分组大小
    int layer_group_capacity   = 0; // GPU缓存池能容纳的分组数量
    
    // --- 状态映射 (核心) ---
    std::unordered_map<int, int> group_to_slot_hash; // group_id -> slot_id

    // --- Reload Plan Result (核心) ---
    reload_plan_result plan_result;

    // 实现LRU(最近最少使用)的替换策略，存储的是【原始神经元索引】, ofc we dont use lru, remove later
    // std::list<int64_t> lru_tracker;
    // std::unordered_map<int64_t, std::list<int64_t>::iterator> lru_map;

    // --- DFR 替换策略状态 ---
    std::vector<float> dfr_scores; // size = layer_group_count
    float dfr_decay_rate = 0.9f;

    // --- 统计信息 ---
    size_t reload_group_count=0;
    size_t offloaded_bytes=0;

public:

    sparkInfer_layer_cache() = default;
    ~sparkInfer_layer_cache();

    /**
     * @brief 初始化层的缓存。
     * 
     * @param layer llama模型中的层。
     * @param backend 用于GPU操作的后端。
     * @param initial_gpu_neuron_indices 初始需要加载到GPU的神经元原始索引列表。
     * @return true 如果初始化成功。
     */
    bool init(int layer_idx, llama_model& model, llama_layer& layer, ggml_backend_t backend, const std::vector<int64_t>& initial_gpu_neu_idx);

    // Sparkinfer reload (graph building)
    ggml_tensor * build_reload_plan(ggml_context * ctx, ggml_tensor * sparse_idx, const int il);
    ggml_tensor * build_reload_exec(ggml_context * ctx, ggml_tensor * sparse_idx, const char * name, const int il);

    // Sparkinfer reload (real implementation)
    bool spif_reload_plan(ggml_tensor * tensor);
    bool spif_reload_exec(ggml_tensor * tensor);

    /**
     * @brief (核心逻辑) 计算 reload plan。
     * 由 GGML_RELOAD_PLAN 的 CUDA 算子函数调用。假设该算子已经执行了 sparse_idx 从 GPU 到 CPU 的拷贝。
     * 步骤：
     * 1. 计算 Top-K (groups_to_ensure)
     * 2. 计算与当前状态的差集
     * 3. 原子地更新内部的 group_to_slot_hash
     * 4. 返回需要执行的 reload/evict plan
     * @return reload_plan_result 包含 groups_to_reload 和 slots_for_evict
     */
    const reload_plan_result & spif_reload_plan(){
        // 确保 sparse_idx 的后端在 CPU 上
        GGML_ASSERT(ggml_backend_get_backend(sparse_idx->backend) == GGML_BACKEND_CPU, "[Error] sparse_idx tensor must be on CPU backend before planning reload.");
        // 1. 第一步：执行 `_spif_reload_plan_count_activated_neurons()`，得到 `group_activated_neurons_count` 数组
        // 2. 第二步：执行 `_spif_reload_plan_get_groups_to_ensure()`，得到 `groups_to_ensure` 数组
        // 3. 第三步：执行 `_spif_reload_plan_compute_diff_and_update_state()`，计算差集并更新状态
        return _spif_reload_plan_compute_diff_and_update_state(_spif_reload_plan_get_groups_to_ensure(_spif_reload_plan_count_activated_neurons()));
    }

private:
    /**
     * @brief 更新缓存的元数据。
     */
    // [YPX] [B] 现在更新元数据（也就是这个类保管的张量）的工作应该由 Reload 算子的 compute_forward() 来负责，但是 init() 函数要调用这个，怎么办？干脆集成到 init() 函数里面算了。
    // [YPX] [Todo] 把这个函数集成到 init() 里面。
    void update_mappings(int64_t neuron_idx, int64_t slot_idx);

    /**
     * @brief [Plan Step 1] 统计激活的神经元数量。
     * @return std::vector<int> group_activated_neuron_count 每个组激活的神经元数量。
     */
    std::vector<int> _spif_reload_plan_count_activated_neurons();

    /**
     * @brief [Plan Step 2] 根据 DFR (或 LRU 等) 策略，计算出应在 GPU 上的 Top-K 组。
     * @param group_activated_neuron_count Step 1 的结果。
     * @return std::vector<int> groups_to_ensure 应该确保在 GPU 上的组列表。
     */
    std::vector<int> _spif_reload_plan_get_groups_to_ensure(const std::vector<int>& group_activated_neuron_count);

    /**
     * @brief [Plan Step 2.1] DFR 策略的具体实现。
     */
    std::vector<int> _spif_reload_plan_use_dfr(const std::vector<int>& group_activated_neuron_count, std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>>& min_heap);
    
    // [YPX] [Todo] 其他策略，比如 _spif_reload_plan_use_lru()

    /**
     * @brief [Plan Step 3] 计算差集并更新状态。
     * @param groups_to_ensure Step 2 的结果。
     * @return reload_plan_result 包含 groups_to_reload 和 slots_for_evict。
     */
    const reload_plan_result & _spif_reload_plan_compute_diff_and_update_state(const std::vector<int>& groups_to_ensure);

};


/**
 * @brief 全局神经元缓存管理器，管理模型中所有层的缓存。
 *
 */
struct sparkInfer_cache_manager {

public:

    // [YPX] [Q] 已知 compute_forward 全部都是用 C 语言写的，这里的 vector<sparkInfer_layer_cache*> 真的靠得住吗？
    std::vector<sparkInfer_layer_cache*> layer_caches;
    llama_model *model = nullptr;
    size_t total_offloaded_bytes=0;

    /**
     * @brief 在推理前，准备好指定层所需的一组神经元。
     *
     * @param layer_idx 层的索引。
     * @param required_neuron_indices 需要确保在GPU上的神经元原始索引列表。
     * @param out_gpu_slot_indices [输出参数] 填充更新后的GPU槽位索引，用于计算图。
     */
    // [YPX] [C] 不需要了。
    // void prepare_hot_neurons(int layer_idx, const std::vector<int64_t>& required_neuron_indices, std::vector<int64_t>& out_gpu_slot_indices);

    bool init(llama_model &p_model, ggml_backend_t gpu_backend);
};

/**
 * @brief 负责加载和应用分割张量的类。
 * 
 * 该类从指定的GGUF分割文件中加载张量，并根据模型层的GPU卸载比例将它们分配到CPU或GPU上下文中。
 * 它还管理分配的缓冲区，以确保正确的内存使用。
 */
struct sparkinfer_split_loader {
    std::string fname;
    struct gguf_context * ctx_gguf = nullptr;
    struct ggml_context * ctx_meta = nullptr;

    std::vector<ggml_backend_buffer_t> allocated_buffers;
    // 分别管理CPU和GPU张量的元数据
    struct ggml_context * ctx_cpu_tensors = nullptr;
    struct ggml_context * ctx_gpu_tensors = nullptr;

    int n_tensors = 0;
    uint64_t vram_required = 0;
    uint64_t layer_neuron_count = 0;
    uint64_t layer_group_count = 0;
    uint64_t layer_group_size = 0;

    sparkinfer_split_loader(const std::string & fname);

    ~sparkinfer_split_loader();

    /**
     * @brief 加载分割张量并将其应用到模型层。
     *        为动态张量预分配最大容量。
     * 
     * @param model 要应用张量的llama模型。
     * @param gpu_backend 用于卸载的GPU后端句柄，如果只想用CPU则为nullptr。
     * @return true 如果成功。
     */
    bool load_and_apply_split(llama_model & model, ggml_backend_t gpu_backend);

private:
    struct ggml_tensor* get_tensor_meta_from_gguf(int layer_idx, const std::string& suffix);
    struct ggml_tensor* create_static_tensor_in_ctx(ggml_context* ctx, int layer_idx, const std::string& suffix);
};

void debug_print_tensor_i64_to_file(FILE* log_file, const struct ggml_tensor* tensor);

sparkInfer_cache_manager* sparkinfer_init_and_manage_ffn_cache(struct llama_model* model, ggml_backend_t gpu_backend);

static bool sparkinfer_load_gpu_split_from_split_file(llama_model & model, std::string split_path, size_t vram_allocatable_bytes);

static bool sparkinfer_load_gpu_split_with_budget(llama_model_loader & ml, llama_model & model, size_t vram_allocatable_bytes, bool no_cache);

size_t sparkinfer_load_gpu_split_and_offload_weight(llama_model_loader & ml, llama_model & model, size_t vram_budget_bytes,bool no_cache);