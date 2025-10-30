#include "llama-sparkinfer.h"

sparkInfer_layer_cache::~sparkInfer_layer_cache() {
    if (gpu_weights_buffer) {
        ggml_backend_buffer_free(gpu_weights_buffer);
    }
    if (tmp_ctx) {
        ggml_free(tmp_ctx);
    }
}

// [YPX] [C] Init function is replaced by the constructor, which has not been merged yet.
bool sparkInfer_layer_cache:: init(int layer_idx, llama_model& model, llama_layer& layer, ggml_backend_t backend, const std::vector<int64_t>& initial_gpu_neu_idx) {
    bool has_gate = true; // GTODO: gate diverage
    bool full_gpu = layer.gpu_offload_ratio >= 1;

    // init backend
    gpu_backend             = backend;
    cpu_backend             = ggml_backend_cpu_init();

    // cpu side tensors
    cpu_ffn_gate            = has_gate ? layer.ffn_gate : nullptr;
    cpu_ffn_up              = layer.ffn_up;
    cpu_ffn_down_t          = layer.ffn_down_t;

    // mappings
    ffn_gpu_neu_idx             = layer.ffn_gpu_neu_idx;
    ffn_gpu_neu_mask            = layer.ffn_gpu_neu_mask;
    ffn_gpu_group_idx           = layer.ffn_gpu_group_idx;
    ffn_gpu_group_mask          = layer.ffn_gpu_group_mask;
    ffn_neuron_to_group_map     = layer.ffn_neuron_to_group_map;
    
    // metadata
    layer_neuron_count      = model.layer_neuron_count; // 每层神经元总数
    layer_group_size        = model.layer_group_size; // 每层分组大小
    layer_group_count       = model.layer_group_count; // 每层分组数量
    neuron_cache_capacity   = initial_gpu_neu_idx.size();
    if (neuron_cache_capacity == 0) return true; // 无需卸载

    // init DRF_score from ffn_gpu_neu_mask, simply cpy the mask to score with int64_t to float conversion
    // maybe we coulf optimize this later
    if(!full_gpu){
        struct ggml_init_params params = { ggml_tensor_overhead() * 2, NULL, true };
        tmp_ctx = ggml_init(params);
        dfr_score = ggml_new_tensor_1d(tmp_ctx, GGML_TYPE_F32, ffn_gpu_neu_mask->ne[0]);
        ggml_set_name(dfr_score, (std::string("blk.") + std::to_string(layer_idx) + std::string(".ffn_dfr_score")).c_str());

        // alloc buffer for dfr_score
        ggml_backend_buffer_t dfr_score_buffer = ggml_backend_alloc_buffer(cpu_backend, ggml_nbytes(dfr_score));
        if (!dfr_score_buffer) {
            LLAMA_LOG_ERROR("%s: failed to allocate CPU buffer for dfr_score\n", __func__);
            return false;
        }
        ggml_backend_tensor_alloc(dfr_score_buffer, dfr_score, ggml_backend_buffer_get_base(dfr_score_buffer));

        const int64_t n_mask = ffn_gpu_neu_mask->ne[0];
        int64_t * mask_data = (int64_t*)ffn_gpu_neu_mask->data;
        float * score_data = new float[ffn_gpu_neu_mask->ne[0]];
        for(int64_t i=0; i<n_mask; i++){
            score_data[i] = (float)(mask_data[i]);
        }
        ggml_backend_tensor_set(dfr_score, score_data, 0, ggml_nbytes(dfr_score));  // if we could get buffer from score directly, we could optimize this by direct cpy
        delete[] score_data;
    }

    /* debug info */ 
    // FILE* log_file = fopen("debug_split_info.log", "a");
    // if (log_file == NULL) {
    //     // 如果文件打开失败，可以打印一个错误到 stderr 然后继续，或者直接退出
    //     perror("Failed to open log file");
    //     // return; // 或者根据你的错误处理逻辑决定是否返回
    // }
    // std::time_t result = std::time(nullptr);
    // fprintf(log_file, "\n--- Debugging Layer %d split info into file, timestamp %s ---", layer_idx, std::ctime(&result)); // 添加一些上下文信息
    // debug_print_tensor_i64_to_file(log_file, ffn_gpu_neu_idx);
    // debug_print_tensor_i64_to_file(log_file, ffn_gpu_neu_mask);
    // debug_print_tensor_i64_to_file(log_file, ffn_gpu_group_idx);
    // debug_print_tensor_i64_to_file(log_file, ffn_gpu_group_mask);
    // debug_print_tensor_i64_to_file(log_file, ffn_neuron_to_group_map);
    // fflush(log_file);
    // fclose(log_file);
    /* debug info end */ 

    GGML_ASSERT(neuron_cache_capacity <= layer_neuron_count && "we required neuron_cache_capacity <= layer_neuron_count");

    // 1. 计算并分配ffn buffer size
    const size_t single_mat_size = ggml_backend_buft_get_alloc_size(
        ggml_backend_get_default_buffer_type(gpu_backend),
        cpu_ffn_down_t // 使用其中一个矩阵作为尺寸参考
    );
    // 我们只缓存部分行，所以要按比例计算
    const size_t single_cache_size = (single_mat_size / layer_neuron_count) * neuron_cache_capacity;
    
    // 总大小 = 3个矩阵的缓存大小之和
    const size_t total_gpu_buffer_size = single_cache_size * 3;

    gpu_weights_buffer = ggml_backend_alloc_buffer(gpu_backend, total_gpu_buffer_size);
    if (!gpu_weights_buffer) {
        LLAMA_LOG_ERROR("%s: failed to allocate GPU buffer for layer cache\n", __func__);
        return false;
    }

    // 2. 在GPU缓存池中创建代表缓存张量
    struct ggml_init_params params = { ggml_tensor_overhead() * 6, NULL, true };
    tmp_ctx = ggml_init(params);

    void* current_addr = ggml_backend_buffer_get_base(gpu_weights_buffer);
    
    char gate_name[64];

    if(has_gate) {
        gpu_ffn_gate_cache = ggml_new_tensor_2d(tmp_ctx, cpu_ffn_gate->type, cpu_ffn_gate->ne[0], neuron_cache_capacity);
        snprintf(gate_name, sizeof(gate_name), "blk.%d.ffn_gpu_gate.weight", layer_idx);
        ggml_set_name(gpu_ffn_gate_cache, gate_name);
        ggml_backend_tensor_alloc(gpu_weights_buffer, gpu_ffn_gate_cache, current_addr);
        current_addr = (char*)current_addr + single_cache_size;
    }

    gpu_ffn_up_cache = ggml_new_tensor_2d(tmp_ctx, cpu_ffn_up->type, cpu_ffn_up->ne[0], neuron_cache_capacity);
    snprintf(gate_name, sizeof(gate_name), "blk.%d.ffn_gpu_up.weight", layer_idx);
    ggml_set_name(gpu_ffn_up_cache, gate_name);
    ggml_backend_tensor_alloc(gpu_weights_buffer, gpu_ffn_up_cache, current_addr);
    current_addr = (char*)current_addr + single_cache_size;

    gpu_ffn_down_t_cache = ggml_new_tensor_2d(tmp_ctx, cpu_ffn_down_t->type, cpu_ffn_down_t->ne[0], neuron_cache_capacity);
    snprintf(gate_name, sizeof(gate_name), "blk.%d.ffn_gpu_down_t.weight", layer_idx);
    ggml_set_name(gpu_ffn_down_t_cache, gate_name);
    ggml_backend_tensor_alloc(gpu_weights_buffer, gpu_ffn_down_t_cache, current_addr);
    
    // 将新的GPU缓存张量赋给llama_layer
    layer.ffn_gpu_gate = has_gate ? gpu_ffn_gate_cache : nullptr;
    layer.ffn_gpu_up   = gpu_ffn_up_cache;
    layer.ffn_gpu_down_t = gpu_ffn_down_t_cache;

    // 3. init slot_to_neuron_map from ffn_gpu_neu_idx, and copy data to buffer
    auto t_start = ggml_time_ms();
    slot_to_neuron_map.resize(neuron_cache_capacity, -1);

    auto batch_copy_neurons = [&](ggml_tensor* cpu_src, ggml_tensor* gpu_dst_cache, const std::vector<int64_t>& indices, const bool full_gpu) {
        if(full_gpu){
            const size_t full_tensor_bytes = ggml_nbytes(cpu_src);
            ggml_backend_tensor_set(gpu_dst_cache, cpu_src->data, 0, full_tensor_bytes);
        }else{
            const int64_t n_embd = cpu_src->ne[0];
            const size_t row_size_bytes = ggml_row_size(cpu_src->type, n_embd);
            
            // 在CPU上分配一个临时暂存缓冲区
            std::vector<char> staging_buffer(row_size_bytes * indices.size());
            
            for (size_t i = 0; i < indices.size(); ++i) {
                const int64_t neuron_idx = indices[i];
                
                // 源地址：在完整CPU张量中的位置
                char* src_ptr = (char*)cpu_src->data + neuron_idx * cpu_src->nb[1];
                
                // 目标地址：在暂存缓冲区中的位置
                char* dst_ptr = staging_buffer.data() + i * row_size_bytes;
                
                memcpy(dst_ptr, src_ptr, row_size_bytes);
            }
            
            ggml_backend_tensor_set(gpu_dst_cache, staging_buffer.data(), 0, staging_buffer.size());
        }
    };

    if(has_gate) batch_copy_neurons(cpu_ffn_gate, gpu_ffn_gate_cache, initial_gpu_neu_idx, full_gpu);
    batch_copy_neurons(cpu_ffn_up, gpu_ffn_up_cache, initial_gpu_neu_idx, full_gpu);
    batch_copy_neurons(cpu_ffn_down_t, gpu_ffn_down_t_cache, initial_gpu_neu_idx, full_gpu);

    // 更新元数据
    offloaded_bytes += ggml_nbytes(gpu_ffn_up_cache) * (has_gate ? 3 : 2); // 每个神经元有3个矩阵
    for (size_t i = 0; i < initial_gpu_neu_idx.size(); ++i) {
        int64_t neuron_idx = initial_gpu_neu_idx[i];
        int64_t slot_idx = i;
        update_mappings(neuron_idx, slot_idx);
    }

    auto t_end = ggml_time_ms();
    LLAMA_LOG_INFO("%s: layer %d offload in %lld ms, cached %d neurons %s\n", __func__, layer_idx, t_end - t_start, neuron_cache_capacity, full_gpu?"(full_gpu)":" ");

    return true;
}

// Sparkinfer reload (graph building)
ggml_tensor * sparkInfer_layer_cache::build_reload_plan(ggml_context * ctx, ggml_tensor * sparse_idx, const int il){
    // [YPX] [Todo] Implementation
    return nullptr;
}

ggml_tensor * sparkInfer_layer_cache::build_reload_exec(ggml_context * ctx, ggml_tensor * sparse_idx, const char * name, const int il){
    // [YPX] [Todo] Implementation
    return nullptr;
}

// [YPX] [Q] Maybe these two functions are already implemented with names given by me?
bool sparkInfer_layer_cache::spif_reload_plan(ggml_tensor * tensor){
    // [YPX] [Todo] Implementation
    return true;
}

bool sparkInfer_layer_cache::spif_reload_exec(ggml_tensor * tensor){
    // [YPX] [Todo] Implementation
    return true;
}

// --- Reload Plan Implementation ---
/**
 * @brief [Plan Step 1] 统计激活的神经元数量。
 * @return std::vector<int> group_activated_neuron_count 每个组激活的神经元数量。
 */
std::vector<int> sparkInfer_layer_cache::_spif_reload_plan_count_activated_neurons(){
    std::vector<int> group_activated_neuron_count(this->layer_group_count, 0);
    const int* p_sparse_idx = (const int*)this->sparse_idx->data;

    for (int neuron_idx = 0; neuron_idx < this->layer_neuron_count; ++neuron_idx) {
        if (p_sparse_idx[neuron_idx]) { // 1 = activated
            int group_idx = neuron_idx / this->layer_group_size;
            if (group_idx < this->layer_group_count) { // 安全检查
                group_activated_neuron_count[group_idx]++;
            }
        }
    }
    return group_activated_neuron_count;
}

/**
 * @brief [Plan Step 2] 根据 DFR (或 LRU 等) 策略，计算出应在 GPU 上的 Top-K 组。
 * @param group_activated_neuron_count Step 1 的结果。
 * @return std::vector<int> groups_to_ensure 应该确保在 GPU 上的组列表。
 */
std::vector<int> sparkInfer_layer_cache::_spif_reload_plan_get_groups_to_ensure(const std::vector<int>& group_activated_neuron_count){
    // 使用 constexpr if 达到编译时分派
    if constexpr (SPARKINFER_RELOAD_STRATEGY == sparkinfer_reload_strategy::USE_DFR) {
        return _spif_reload_plan_use_dfr(group_activated_neuron_count);
    } else if constexpr (SPARKINFER_RELOAD_STRATEGY == sparkinfer_reload_strategy::USE_LRU) {
        // _spif_reload_plan_use_lru(group_activated_neuron_count); // 调用 LRU 实现
        GGML_LOG_ERROR("%s: [Error] LRU strategy not implemented yet.\n", __func__); 
    } else {
        GGML_LOG_WARN("%s: [Warning] Unknown Reload Strategy, fall back to DFR\n", __func__);
        return _spif_reload_plan_use_dfr(group_activated_neuron_count); // 默认使用 DFR
    }
}

/**
 * @brief [Plan Step 2.1] DFR 策略的具体实现。
 */
std::vector<int> sparkInfer_layer_cache::_spif_reload_plan_use_dfr(const std::vector<int>& group_activated_neuron_count) {
    typedef std::pair<float, int> ScoreGroupPair;
    std::priority_queue<ScoreGroupPair, std::vector<ScoreGroupPair>, std::greater<ScoreGroupPair>> min_heap;

    for (int group_id = 0; group_id < this->layer_group_count; ++group_id) {
        //  dfr_score = dfr_score * decay + is_activated * (1 - decay)
        float is_activated = (group_activated_neuron_count[group_id] > 0) ? 1.0f : 0.0f;
        float old_score = this->dfr_scores[group_id];
        float new_score = old_score * this->dfr_decay_rate + is_activated * (1.0f - this->dfr_decay_rate);
        
        this->dfr_scores[group_id] = new_score; // 持久化更新 DFR 分数

        // 维护 Top-K 最小堆
        if (min_heap.size() < (size_t)this->layer_group_capacity) {
            min_heap.push({new_score, group_id});
        } else if (new_score > min_heap.top().first) {
            min_heap.pop();
            min_heap.push({new_score, group_id});
        }
    }

    // 从堆中提取 Top-K 组 (groups_to_ensure)
    std::vector<int> groups_to_ensure;
    groups_to_ensure.reserve(this->layer_group_capacity);
    while (!min_heap.empty()) {
        groups_to_ensure.push_back(min_heap.top().second); // .second 是 group_id
        min_heap.pop();
    }
    return groups_to_ensure;
}

/**
 * @brief [Plan Step 3] 计算差集并更新状态。
 * @param groups_to_ensure Step 2 的结果。
 * @return reload_plan_result 包含 plan_result.groups_to_reload 和 plan_result.slots_for_evict。
 */
const reload_plan_result & sparkInfer_layer_cache::_spif_reload_plan_compute_diff_and_update_state(const std::vector<int>& groups_to_ensure) {
    // Clear previous plan results
    this->plan_result.groups_to_reload.clear();
    this->plan_result.slots_for_evict.clear();

    // 1. O(k) 建立目标组的哈希集合
    std::unordered_set<int> target_groups(groups_to_ensure.begin(), groups_to_ensure.end());

    // 2. O(k) 遍历当前 GPU 上的组 (group_to_slot_hash)
    //    找出 Hit (保留), Miss (需驱逐)
    std::vector<int> groups_to_evict;
    
    for (auto it = this->group_to_slot_hash.begin(); it != this->group_to_slot_hash.end(); /* no increment */) {
        int group_in_gpu = it->first;
        int slot_id      = it->second;

        if (target_groups.find(group_in_gpu) == target_groups.end()) {
            // Miss: 在 GPU, 但不在 Target -> 驱逐
            groups_to_evict.push_back(group_in_gpu);
            this->plan_result.slots_for_evict.push_back(slot_id);
            
            // 更新状态: 从哈希表中移除
            it = this->group_to_slot_hash.erase(it);
        } else {
            // Hit: 在 GPU, 也在 Target -> 保留
            it++;
        }
    }

    // 3. O(k) 遍历目标组, 找出需要新加载的组
    for (const int group_id : target_groups) {
        if (this->group_to_slot_hash.find(group_id) == this->group_to_slot_hash.end()) {
            // 在 Target, 但不在 GPU (哈希表) -> 加载
            this->plan_result.groups_to_reload.push_back(group_id);
        }
    }

    // 4. 关键断言：空出的槽位必须等于需要加载的组
    GGML_ASSERT(this->plan_result.slots_for_evict.size() == this->plan_result.groups_to_reload.size());

    // 5. 更新状态: 将新加载的组“放回”哈希表，复用空闲 slot
    for (size_t i = 0; i < this->plan_result.groups_to_reload.size(); ++i) {
        int group_to_load = this->plan_result.groups_to_reload[i];
        int slot_to_use   = this->plan_result.slots_for_evict[i]; // 复用被驱逐的 slot
        
        this->group_to_slot_hash[group_to_load] = slot_to_use;
    }

    return this->plan_result;
}

// --- sparkInfer_cache_manager Implementation ---

sparkInfer_cache_manager::~sparkInfer_cache_manager() {
    for (auto layer_cache : layer_caches) {
        if (layer_cache) {
            delete layer_cache;
        }
    }
}

bool sparkInfer_cache_manager:: init(llama_model &p_model, ggml_backend_t gpu_backend) {
    this->model = &p_model;
    const int n_layers = model->hparams.n_layer;
    layer_caches.resize(n_layers);

    const int64_t t_start_us = ggml_time_us();
    LLAMA_LOG_INFO("%s: Initializing neuron cache for %d layers...\n", __func__, n_layers);

    // #pragma omp parallel for
    for (int i = 0; i < n_layers; ++i) {
        llama_layer &layer = model->layers[i];
        if (layer.gpu_offload_ratio == 0.f || !layer.ffn_gpu_neu_idx) {
            LLAMA_LOG_INFO("%s: layer %d won't split partial tensor to GPU", __func__, i);
            continue; // 该层不卸载或没有索引信息
        }

        // 从ffn_gpu_neu_idx读取需要加载的神经元
        struct ggml_tensor* initial_gpu_neu_idx = layer.ffn_gpu_neu_idx;
        std::vector<int64_t> gpu_neu_idx_vec(initial_gpu_neu_idx->ne[0]);
        
        // 注意：这里需要从设备或主机内存中获取数据 TAG
        ggml_backend_tensor_get(initial_gpu_neu_idx, gpu_neu_idx_vec.data(), 0, ggml_nbytes(initial_gpu_neu_idx));

        sparkInfer_layer_cache * spif_layer = new sparkInfer_layer_cache();
        bool init_layer_success = spif_layer->init(i, p_model, layer, gpu_backend, gpu_neu_idx_vec);
        layer_caches[i] = spif_layer;
        
        if (!init_layer_success) {
            LLAMA_LOG_ERROR("%s: failed to initialize cache for layer %d\n", __func__, i);
            throw std::runtime_error("Failed to initialize layer cache");
        }
        total_offloaded_bytes += layer_caches[i]->offloaded_bytes;
    }
    
    // 在所有层初始化后，可以释放掉模型中用于加载的静态索引缓冲区
    // for (auto& buffer : model->split_idx_allocated_buffers) {
    //     ggml_backend_buffer_free(buffer);
    // }
    // model->split_idx_allocated_buffers.clear();
    // if (model->ctx_cpu_idx_tensors) {
    //     ggml_free(model->ctx_cpu_idx_tensors);
    //     model->ctx_cpu_idx_tensors = nullptr;
    // }
    // if (model->ctx_gpu_idx_tensors) {
    //     ggml_free(model->ctx_gpu_idx_tensors);
    //     model->ctx_gpu_idx_tensors = nullptr;
    // }


    const int64_t t_end_us = ggml_time_us();
    LLAMA_LOG_INFO("%s: Neuron cache initialized successfully. (%.2f ms)\n", __func__, (t_end_us - t_start_us) / 1000.0);

    return true;
}

/*
void sparkInfer_cache_manager:: prepare_hot_neurons(int layer_idx, const std::vector<int64_t>& required_neuron_indices, std::vector<int64_t>& out_gpu_slot_indices) {
    if (layer_caches[layer_idx]->layer_neuron_capacity == 0) return;

    out_gpu_slot_indices.resize(required_neuron_indices.size());
    for (size_t i = 0; i < required_neuron_indices.size(); ++i) {
        out_gpu_slot_indices[i] = layer_caches[layer_idx]->ensure_neuron_on_gpu(required_neuron_indices[i]);
    }

    // 实际应用中，你需要用 out_gpu_slot_indices 的数据更新计算图中使用的索引张量。
    // 例如：
    // struct ggml_tensor* graph_neu_idx = model->layers[layer_idx].ffn_gpu_neu_idx;
    // ggml_backend_tensor_set(graph_neu_idx, out_gpu_slot_indices.data(), 0, sizeof(int32_t) * out_gpu_slot_indices.size());
}
*/


sparkinfer_split_loader:: sparkinfer_split_loader(const std::string & fname) : fname(fname) {
    struct gguf_init_params params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx_meta,
    };

    ctx_gguf = gguf_init_from_file(fname.c_str(), params);
    if (!ctx_gguf) {
        throw std::runtime_error("无法打开或加载分割文件: " + fname);
    }

    // 读取元数据
    int key_idx = gguf_find_key(ctx_gguf, "split.vram_capacity");
    if (key_idx >= 0) {
        vram_required = gguf_get_val_u64(ctx_gguf, key_idx);
    } else {
            LLAMA_LOG_WARN("警告: 'split.vram_capacity' key not found in %s\n", fname.c_str());
    }

    key_idx = gguf_find_key(ctx_gguf, "split.layer_neuron_count");
    if (key_idx >= 0) {
        layer_neuron_count = gguf_get_val_u64(ctx_gguf, key_idx);
    } else {
            LLAMA_LOG_WARN("警告: 'split.layer_neuron_count' key not found in %s\n", fname.c_str());
    }

    key_idx = gguf_find_key(ctx_gguf, "split.layer_group_count");
    if (key_idx >= 0) {
        layer_group_count = gguf_get_val_u64(ctx_gguf, key_idx);
    } else {
            LLAMA_LOG_WARN("警告: 'split.layer_group_count' key not found in %s\n", fname.c_str());
    }

    layer_group_size = layer_neuron_count / layer_group_count;
    n_tensors = gguf_get_n_tensors(ctx_gguf);
    
    LLAMA_LOG_INFO("%s: 成功加载分割文件 '%s' 的信息和数据. 张量数量: %d, VRAM需求: %zu MB\n",
        __func__, fname.c_str(), n_tensors, (size_t)vram_required);
}

sparkinfer_split_loader:: ~sparkinfer_split_loader() {
    if (ctx_gguf) gguf_free(ctx_gguf);
    if (ctx_meta) ggml_free(ctx_meta); // ctx_meta 现在管理着数据，ggml_free会释放它
    // ctx_cpu_tensors 和 ctx_gpu_tensors 会被模型接管，由模型管理其生命周期
    
    // 分配的缓冲区也应由模型管理
    // for (auto & buffer : allocated_buffers) {
    //     ggml_backend_buffer_free(buffer);
    // }
}

bool sparkinfer_split_loader:: load_and_apply_split(llama_model & model, ggml_backend_t gpu_backend) {
    const int n_layers = model.hparams.n_layer;

    if (n_tensors != n_layers * 5) {
        LLAMA_LOG_ERROR("%s: 错误: 分割文件中的张量数量 (%d) 与模型层数 (%d * 5) 不匹配\n",
            __func__, n_tensors, n_layers);
        return false;
    }
    if (layer_neuron_count <= 0) {
        throw std::runtime_error("layer_neuron_count 必须为正数");
    }

    const int64_t t_start_us = ggml_time_us();

    // 1. 初始化用于管理 CPU 和 GPU 张量元数据的上下文
    struct ggml_init_params params = { ggml_tensor_overhead() * n_tensors, NULL, true };
    ctx_cpu_tensors = ggml_init(params);
    ctx_gpu_tensors = ggml_init(params);
    if (!ctx_cpu_tensors || !ctx_gpu_tensors) {
        throw std::runtime_error("无法初始化CPU/GPU张量上下文");
    }

    // 2. 遍历层，创建目标张量元数据并分配到正确的上下文中
    for (int il = 0; il < n_layers; ++il) {
        llama_layer & layer = model.layers[il];
        
        struct ggml_tensor* temp_neu_idx_meta = get_tensor_meta_from_gguf(il, "ffn_gpu_neu_idx");
        if (!temp_neu_idx_meta) throw std::runtime_error("无法从GGUF获取 ffn_gpu_neu_idx 元数据");

        int64_t initial_gpu_neurons = temp_neu_idx_meta->ne[0];

        bool offload_layer = gpu_backend && initial_gpu_neurons > 0;
        layer.gpu_offload_ratio = float(initial_gpu_neurons) / float(layer_neuron_count);
        ggml_context* target_ctx = offload_layer ? ctx_gpu_tensors : ctx_cpu_tensors;

        LLAMA_LOG_INFO("%s: layer %2d: offload %.2f%% to GPU\n", __func__, il, layer.gpu_offload_ratio * 100.0f);

        // --- 在目标上下文中创建张量元数据 ---
        layer.ffn_gpu_neu_idx = create_static_tensor_in_ctx(target_ctx, il, "ffn_gpu_neu_idx");
        layer.ffn_gpu_group_idx = create_static_tensor_in_ctx(target_ctx, il, "ffn_gpu_group_idx");
        layer.ffn_gpu_neu_mask = create_static_tensor_in_ctx(ctx_cpu_tensors, il, "ffn_gpu_neu_mask");
        layer.ffn_gpu_group_mask = create_static_tensor_in_ctx(ctx_cpu_tensors, il, "ffn_gpu_group_mask");
        layer.ffn_neuron_to_group_map = create_static_tensor_in_ctx(ctx_cpu_tensors, il, "ffn_neuron_to_group_map");

        if (!layer.ffn_gpu_neu_idx || !layer.ffn_gpu_group_idx || !layer.ffn_gpu_neu_mask || !layer.ffn_gpu_group_mask || !layer.ffn_neuron_to_group_map) {
            throw std::runtime_error("在目标上下文中创建张量元数据失败，层 " + std::to_string(il));
        }
    }
    
    // 3. 为CPU和GPU上下文批量分配缓冲区
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (ggml_get_first_tensor(ctx_cpu_tensors)) {
        ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(ctx_cpu_tensors, cpu_backend);
        if (!cpu_buffer) {
            ggml_backend_free(cpu_backend);
            throw std::runtime_error("为CPU张量分配缓冲区失败");
        }
        allocated_buffers.push_back(cpu_buffer);
    }

    if (gpu_backend && ggml_get_first_tensor(ctx_gpu_tensors)) {
        ggml_backend_buffer_t gpu_buffer = ggml_backend_alloc_ctx_tensors(ctx_gpu_tensors, gpu_backend);
        if (!gpu_buffer) {
            ggml_backend_free(cpu_backend); // cpu_backend 依然需要释放
            throw std::runtime_error("为GPU张量分配缓冲区失败");
        }
        allocated_buffers.push_back(gpu_buffer);
    }

    // 4. 修改点: 使用 ggml_backend_tensor_set 从已加载数据的源张量设置目标张量
    for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
        const char* name = gguf_get_tensor_name(ctx_gguf, i);
        
        // 源张量 (数据已由 gguf_init 加载到其 ->data 指针中)
        struct ggml_tensor* src_tensor = ggml_get_tensor(ctx_meta, name);
        if (!src_tensor) {
                throw std::runtime_error("在源 GGUF 上下文中未找到张量: " + std::string(name));
        }
        if (!src_tensor->data) {
            throw std::runtime_error("源张量数据未加载(data is null): " + std::string(name));
        }

        // 查找目标张量 (已由后端分配好缓冲区)
        struct ggml_tensor* dst_tensor = ggml_get_tensor(ctx_cpu_tensors, name);
        if (!dst_tensor) {
            dst_tensor = ggml_get_tensor(ctx_gpu_tensors, name);
        }

        if (dst_tensor) {
            // 使用 ggml_backend_tensor_set 将数据从 host (src_tensor->data) 复制到 device (dst_tensor)
            // 这会自动处理 CPU->CPU 或 CPU->GPU 的情况
            const size_t nbytes = ggml_nbytes(src_tensor);
            ggml_backend_tensor_set(dst_tensor, src_tensor->data, 0, nbytes);
        } else {
            throw std::runtime_error("在目标上下文中未找到张量: " + std::string(name));
        }
    }

    const int64_t t_end_us = ggml_time_us();
    LLAMA_LOG_INFO("%s: 分割数据复制完毕，耗时 %.2f 毫秒\n", __func__, (t_end_us - t_start_us) / 1000.0);
    

    // 5. 保存数据到 llama_model (逻辑不变)
    model.ctx_cpu_idx_tensors = ctx_cpu_tensors;
    model.ctx_gpu_idx_tensors = ctx_gpu_tensors;

    model.layer_neuron_count = layer_neuron_count;
    model.layer_group_count = layer_group_count;
    model.layer_group_size = layer_group_size;
    model.split_idx_allocated_buffers = allocated_buffers;

    return true;
}

struct ggml_tensor * sparkinfer_split_loader:: get_tensor_meta_from_gguf(int layer_idx, const std::string& suffix) {
    char name[GGML_MAX_NAME];
    snprintf(name, sizeof(name), "blk.%d.%s", layer_idx, suffix.c_str());
    return ggml_get_tensor(ctx_meta, name);
}

struct ggml_tensor * sparkinfer_split_loader:: create_static_tensor_in_ctx(ggml_context* ctx, int layer_idx, const std::string& suffix) {
    struct ggml_tensor* t_meta = get_tensor_meta_from_gguf(layer_idx, suffix.c_str());
    if (!t_meta) {
        LLAMA_LOG_WARN("%s: 无法找到张量元数据 blk.%d.%s\n", __func__, layer_idx, suffix.c_str());
        return nullptr;
    }
    struct ggml_tensor* new_t = ggml_dup_tensor(ctx, t_meta);
    ggml_set_name(new_t, t_meta->name);
    return new_t;
}



void debug_print_tensor_i64_to_file(FILE* log_file, const struct ggml_tensor* tensor) {
    if (!log_file) {
        return;
    }

    if (!tensor) {
        fprintf(log_file, "debug_print_tensor_i64: tensor is NULL\n");
        return;
    }

    // 检查类型 (这部分不变)
    if (tensor->type != GGML_TYPE_I64) {
        fprintf(log_file, "debug_print_tensor_i64: tensor '%s' is not INT64 (type = %d)\n", tensor->name, tensor->type);
        return;
    }

    // 检查是否为一维
    const int ne0 = tensor->ne[0];  // 第一维大小
    const int ne1 = tensor->ne[1];
    const int ne2 = tensor->ne[2];
    const int ne3 = tensor->ne[3];

    if (ne1 != 1 || ne2 != 1 || ne3 != 1) {
        fprintf(log_file, "debug_print_tensor_i64: tensor '%s' is not 1D (shape: %d x %d x %d x %d)\n", tensor->name, ne0, ne1, ne2, ne3);
        return;
    }

    const int64_t* data_to_print;
    int64_t* cpu_buffer = NULL; // 用于存放从GPU拷贝回来的数据的临时缓冲区

    // 检查张量是否在主机（CPU）内存中
    // 如果 tensor->buffer 为 NULL，则它也在主机内存中
    if (tensor->buffer == NULL || ggml_backend_buffer_is_host(tensor->buffer)) {
        // 如果在 CPU 上，直接使用它的数据指针
        data_to_print = (const int64_t*)tensor->data;
    } else {
        // 如果在 GPU 上，需要将数据拷贝回 CPU
        const size_t tensor_size_bytes = ggml_nbytes(tensor);

        // 1. 在 CPU 上分配一个临时缓冲区
        cpu_buffer = (int64_t*)malloc(tensor_size_bytes);
        if (!cpu_buffer) {
            fprintf(log_file, "debug_print_tensor_i64: failed to allocate CPU buffer for tensor '%s'\n", tensor->name);
            return;
        }

        // 2. 从 GPU (tensor->data) 拷贝数据到 CPU (cpu_buffer)
        // ggml_backend_tensor_get 会自动处理不同后端的拷贝
        ggml_backend_tensor_get(tensor, cpu_buffer, 0, tensor_size_bytes);

        // 3. 让我们的打印指针指向这个新的 CPU 缓冲区
        data_to_print = cpu_buffer;
    }

    if (!data_to_print) {
        fprintf(log_file, "debug_print_tensor_i64: tensor '%s' data is NULL after backend check\n", tensor->name);
        if (cpu_buffer) free(cpu_buffer); // 如果分配了内存，记得释放
        return;
    }

    fprintf(log_file, "ggml_tensor name(%s) (INT64, 1D, size = %d):\n[", tensor->name, ne0);
    for (int i = 0; i < ne0; i++) {
        fprintf(log_file, "%" PRId64, data_to_print[i]);
        if (i < ne0 - 1) fprintf(log_file, ", ");
    }
    fprintf(log_file, "]\n");

    // 如果我们为 GPU 数据分配了临时缓冲区，现在就释放它
    if (cpu_buffer) {
        free(cpu_buffer);
    }
}

sparkInfer_cache_manager* sparkinfer_init_and_manage_ffn_cache(struct llama_model* model, ggml_backend_t gpu_backend) {
    if (!model || !gpu_backend) {
        throw std::invalid_argument("sparkinfer_init_and_manage_ffn_cache: model or gpu_backend is null");
    }

    auto cache_manager = new sparkInfer_cache_manager(); //GTODO: use unique_ptr?
    if (!cache_manager->init(*model, gpu_backend)) {
        throw std::runtime_error("sparkinfer_init_and_manage_ffn_cache: failed to initialize cache manager");
    }

    return cache_manager;
}

static bool sparkinfer_load_gpu_split_from_split_file(llama_model & model, std::string split_path, size_t vram_allocatable_bytes) {

    sparkinfer_split_loader loader(split_path);
    
    ggml_backend_t gpu_backend = ggml_backend_cuda_init(0); 
    if (!gpu_backend) {
        fprintf(stderr, "No GPU backend found\n");
        return false;
    }
    return loader.load_and_apply_split(model,gpu_backend);
}

static bool sparkinfer_load_gpu_split_with_budget(llama_model_loader & ml, llama_model & model, size_t vram_allocatable_bytes, bool no_cache) {
    std::string cached_split_path = ml.fname + ".sparkinfer_split_idx";//"/share/models/prosparse-7b-gguf-w-our-predictor/prosparse-7b.gguf.sparkinfer_split_idx"
    // get basedir from cached split path
    std::string model_basedir;
    if (cached_split_path.find_last_of('/') != std::string::npos) {
        model_basedir = cached_split_path.substr(0, cached_split_path.find_last_of('/'));
    }

    if (no_cache) {
        LLAMA_LOG_INFO("%s: no cache, will generate gpu split\n", __func__);
    } else {
        LLAMA_LOG_INFO("%s: loading gpu split from '%s'\n", __func__, cached_split_path.c_str());
    }

    // Load GPU split from previously generated cache
    if (access(cached_split_path.c_str(), F_OK) == 0 && !no_cache) {
        if (sparkinfer_load_gpu_split_from_split_file(model, cached_split_path, vram_allocatable_bytes)) {
            return true;
        }
        LLAMA_LOG_ERROR("%s: error: failed to apply previously generated gpu split from '%s'\n", __func__, cached_split_path.c_str());
        throw std::runtime_error("Failed to load GPU split from cache");
    }

    // Generate GPU split
    std::string activation_path = std::string(model_basedir)+"/activation";    

    // Calculate solver parameters
    ggml_tensor * ffn_up = model.layers[0].ffn_up;
    ggml_tensor * ffn_gate = model.layers[0].ffn_gate;
    int slice_size = ffn_up->ne[1] * ggml_type_size(ffn_up->type) / ggml_blck_size(ffn_up->type);
    // For model arch with FFN gate, the gate is also sliced, otherwise only the up and down matrices are sliced
    int vram_bytes_per_slice = slice_size * (ffn_gate ? 3 : 2); // GTODO: 4.5 or 3?
    int neuron_cap = floor((double)vram_allocatable_bytes / vram_bytes_per_slice) * 4;

    LLAMA_LOG_INFO("invoking sparkinfer Python module to generate gpu split for %.2f MiB of VRAM\n", vram_allocatable_bytes / 1024.0 / 1024.0);

    std::stringstream command_ss;
    command_ss << "python -m sparkinfer"
               << " --model-path " << model_basedir
               << " --layer " << model.hparams.n_layer
               << " --neuron " << ffn_up->ne[1]
               << " --neuron-capacity " << neuron_cap
               << " --vram-capacity " << vram_allocatable_bytes
               << " --output " << cached_split_path;
    if (system(command_ss.str().c_str()) != 0 || access(cached_split_path.c_str(), F_OK) != 0) {
        LLAMA_LOG_ERROR("%s: error: failed to generate gpu split\n", __func__);
        return false;
    }

    return sparkinfer_load_gpu_split_from_split_file(model, cached_split_path, vram_allocatable_bytes);
}

size_t sparkinfer_load_gpu_split_and_offload_weight(llama_model_loader & ml, llama_model & model, size_t vram_budget_bytes,bool no_cache) {
    // log the VRAM budget
    LLAMA_LOG_INFO("%s: VRAM budget is %.2f MiB\n", __func__, vram_budget_bytes / 1024.0 / 1024.0);

    if (!sparkinfer_load_gpu_split_with_budget(ml, model, vram_budget_bytes, no_cache)) {
        LLAMA_LOG_ERROR("%s: error: failed to generate gpu split, an empty one will be used\n", __func__);
    }

    ggml_backend_t gpu_backend = ggml_backend_cuda_init(0); 
    if (!gpu_backend) {
        fprintf(stderr, "No GPU backend found\n");
        return false;
    }
    sparkInfer_cache_manager* spif_cache = sparkinfer_init_and_manage_ffn_cache(&model, gpu_backend);
    if (!spif_cache) {
        throw std::runtime_error("Failed to initialize neuron cache manager");
    }

    model.spif_cache = spif_cache;
    size_t total_offloaded_bytes = spif_cache->total_offloaded_bytes;
    LLAMA_LOG_INFO("%s: offloaded %.2f MiB of FFN weights to GPU\n", __func__, total_offloaded_bytes / 1024.0 / 1024.0);

    return total_offloaded_bytes;
}

// ----------Spakinfer reload (graph building)----------- //
ggml_tensor * sparkInfer_layer_cache:: build_reload_plan(ggml_context * ctx, ggml_tensor * sparse_idx, const int il){
    GGML_ASSERT(sparse_idx && "sparse_idx is required for reloading");

    // a demo tensor for graph biulding
    ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    
    // added reload params
    memcpy(&result->op_params[0], this, sizeof(sparkInfer_layer_cache*)); // Pass the pointer to SparkInfer_layer_cache

    result->op = GGML_OP_RELOAD_PLAN;
    result->src[0] = sparse_idx;
    return result;
}

ggml_tensor * sparkInfer_layer_cache:: build_reload_exec(ggml_context * ctx, ggml_tensor * plan_done, const char * name, const int il){
    GGML_ASSERT(ffn_gpu_neu_idx && "ffn_gpu_neu_idx is required for reloading");
    GGML_ASSERT(plan_done && "plan_done is required for reloading");

    ggml_tensor * gpu_ffn = nullptr;
    ggml_tensor * cpu_ffn = nullptr;

    if (std::string(name) == "gate") {
        gpu_ffn = gpu_ffn_gate_cache;
        cpu_ffn = cpu_ffn_gate;
    } else if (std::string(name) == "up") {
        gpu_ffn = gpu_ffn_up_cache;
        cpu_ffn = cpu_ffn_up;
    } else if (std::string(name) == "down") {
        gpu_ffn = gpu_ffn_down_t_cache;
        cpu_ffn = cpu_ffn_down_t;
    } else {
        GGML_ASSERT(false && "unsupported name for reload");
    }

    // ggml_tensor * result = ggml_view_2d(ctx, gpu_ffn_up_cache, gpu_ffn_up_cache->ne[0], gpu_ffn_up_cache->ne[1], 0, 0);
    ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    // added reload params
    memcpy(&result->op_params[0], this, sizeof(sparkInfer_layer_cache*)); // Pass the pointer to SparkInfer_layer_cache
    
    result->op = GGML_OP_RELOAD_EXEC;
    result->src[0] = gpu_ffn;
    result->src[1] = cpu_ffn;
    result->src[2] = plan_done; // just a dependency

    return result;
}
// --------------------------------------------------------- //



// ----------Spakinfer reload (real implmentation)----------- //
// this is a C wrapper to call llama-sparkinfer function, used in ggml codebase for relaoding operation
extern "C" {  
    bool ggml_spif_reload_plan(ggml_spif_context* ctx, ggml_tensor * tensor) {  
        sparkInfer_layer_cache * spif_cache = reinterpret_cast<sparkInfer_layer_cache*>(ctx);  
        return spif_cache->spif_reload_plan(tensor);
    }

    bool ggml_spif_reload_exec(ggml_spif_context* ctx, ggml_tensor * tensor) {  
        sparkInfer_layer_cache * spif_cache = reinterpret_cast<sparkInfer_layer_cache*>(ctx);  
        return spif_cache->spif_reload_exec(tensor);
    }
}

// Sparkinfer reload plan (算子 1 + 2)
bool sparkInfer_layer_cache:: spif_reload_plan(ggml_tensor * tensor) {
    // Extraction
    struct ggml_tensor * sparse_idx      = tensor->src[0];
    
    //...
    
    reload_plan_result * rpr = new reload_plan_result();
    return true;
}

// Sparkinfer reload exec (算子 3)
bool sparkInfer_layer_cache:: spif_reload_exec(ggml_tensor * tensor) {
    // ...
    return true;
}
// --------------------------------------------------------- //