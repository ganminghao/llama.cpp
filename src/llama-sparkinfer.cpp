#include "llama-sparkinfer.h"

layer_cache::layer_cache(const llama_layer & layer, int64_t layer_idx, double dfr_decay, double dfr_bonus){

    // ====================================================
    // 1. initialize all member variables to default values
    // ====================================================

    // 1.1 layer idx and full_gpu
    this->layer_idx = layer_idx;
    this->full_gpu = layer.gpu_offload_ratio >= 1.0 ? true : false; // [YPX] [Q] 询问甘明昊：这样初始化是否正确？

    // 1.2 init ffn_gpu_neu_idx/mask
    this->ffn_gpu_neu_idx = layer.ffn_gpu_neu_idx;
    this->ffn_gpu_neu_mask = layer.ffn_gpu_neu_mask;

    // 1.3 init layer_(neuron/group)_(count/size/capacity)
    this->layer_neuron_count    = ffn_gpu_neu_mask->ne[1];
    this->layer_group_size      = layer.ffn_group_size;
    this->layer_group_count     = layer_neuron_count / layer_group_size;
    this->layer_neuron_capacity = ffn_gpu_neu_idx->ne[0];
    this->layer_group_capacity  = layer_neuron_capacity / layer_group_size;
    
    // 1.4 init std::vector gpu_neu_idx/mask_vec
    this->gpu_neu_idx_vec.resize(ffn_gpu_neu_idx->ne[0], 0);
    memcpy(gpu_neu_idx_vec.data(), ffn_gpu_neu_idx->data, ffn_gpu_neu_idx->ne[0] * sizeof(int64_t));
    this->gpu_neu_mask_vec.resize(ffn_gpu_neu_mask->ne[0], 0);
    memcpy(gpu_neu_mask_vec.data(), ffn_gpu_neu_mask->data, ffn_gpu_neu_mask->ne[0] * sizeof(int64_t));

    // 1.5 init group_to_neurons and neuron_to_group_map
    // 1.5.1. get group_to_neurons_map
    this->group_to_neurons_map_tensor = layer.ffn_group_to_neurons_map;
    // 1.5.2. convert group_to_neurons_map_tensor to std::vector<std::vector<int64_t>>
    // [YPX] [C] 我问过 AI，逐元素拷贝更快还是对每一个一维 vector 作 memcpy 更快，它说编译优化使两种方案的差距可以忽略不计
    this->group_to_neurons_map.resize(layer_group_count);
    for(int64_t i=0; i<layer_group_count; i++){
        this->group_to_neurons_map[i].resize(layer_group_size);
        for(int64_t j=0; j<layer_group_size; j++){
            int64_t neuron_idx = ((int64_t *)group_to_neurons_map_tensor->data)[i * layer_group_size + j];
            this->group_to_neurons_map[i][j] = neuron_idx;
        }
    }
    // 1.5.3. get neuron_to_group_map_tensor
    // Note: this tensor is only used in the constructor, so we don't need to store it as a member variable
    ggml_tensor * neuron_to_group_map_tensor   = layer.ffn_neuron_to_group_map; 
    // 1.5.4. convert neuron_to_group_map_tensor to std::vector<int64_t>
    this->neuron_to_group_map.resize(neuron_to_group_map_tensor->ne[0], -1);
    memcpy(neuron_to_group_map.data(), neuron_to_group_map_tensor->data, neuron_to_group_map_tensor->ne[0] * sizeof(int64_t));

    // 1.6 init activated_group_mask
    this->activated_group_mask.resize(layer_group_capacity);

    // 1.7 init dfr related data structures but don't fill them
    this->dfr_bonus = dfr_bonus;
    this->dfr_decay = dfr_decay;
    this->group_dfr_tracker.resize(layer_group_count, 0);

    // 1.8 init slots related data structures but don't fill them
    this->group_to_slot_map.resize(layer_group_count, -1);
    this->slot_to_group_map.resize(layer_group_capacity, -1);

    // ====================================================
    // 2. fill dfr related and slots related data structures
    // ====================================================

    // 2.1 convert ffn_gpu_neu_idx to a vector named "sparse_idx", to cheat the functions below
    std::vector<int64_t> sparse_idx;
    for(int64_t i=0; i<ffn_gpu_neu_idx->ne[0]; i++){
        sparse_idx.push_back(((int64_t *)ffn_gpu_neu_idx->data)[i]);
    }

    // 2.2 call utility functions
    if(update_activated_group_mask(sparse_idx) && update_DFR() && update_slots()){
        LLAMA_LOG_DEBUG("%s: Layer cache for layer %d initialized successfully.\n");
    } else{
        LLAMA_LOG_DEBUG("[Error] %s: Failed to initialize layer cache for layer %d.\n");
    }

}

sparkinfer_reload_plan layer_cache::plan_reload(ggml_tensor * sparse_idx_tensor){

    // 1. convert sparse_idx_tensor to a vector named "sparse_idx"
    std::vector<int64_t> sparse_idx;
    for(int64_t i=0; i<sparse_idx_tensor->ne[0]; i++){
        sparse_idx.push_back(((int64_t *)sparse_idx_tensor->data)[i]);
    }

    // 2. call utility functions
    update_activated_group_mask(sparse_idx);
    update_DFR();
    update_slots();
    update_idx_and_mask();
    return generate_reload_plan();

}

bool layer_cache::update_activated_group_mask(const std::vector<int64_t>& sparse_idx) {

    // 1. reset activated_group_mask to all 0
    std::fill(activated_group_mask.begin(), activated_group_mask.end(), 0);

    // 2. Use vector to count activated neurons per group
    std::vector<int64_t> group_activation_counts(layer_group_count, 0);
    for (int64_t neuron_idx : sparse_idx) {
        int64_t group_idx = neuron_to_group_map[neuron_idx];
        if (group_idx >= 0 && group_idx < layer_group_count) { // Security check
            group_activation_counts[group_idx]++;
        }
    }

    // 3. convert the result int64_to <count, group_id> for quick sorting
    std::vector<std::pair<int64_t, int64_t>> top_k_candidates;
    top_k_candidates.reserve(layer_group_count);
    for (int64_t i = 0; i < layer_group_count; ++i) {
        if (group_activation_counts[i] > 0) { // only consider groups that has at least one activated neuron
            top_k_candidates.push_back({group_activation_counts[i], i});
        }
    }

    // 4. use std::partial_sort to find top-k groups
    const int64_t k = std::min(layer_group_capacity, (int64_t)top_k_candidates.size()); // ensure k doesn't exceed the count of candidates
    std::partial_sort(
        top_k_candidates.begin(),
        top_k_candidates.begin() + k,
        top_k_candidates.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first; // descend by activated neurons count
        }
    );

    // 5. update activated_group_mask
    for (int64_t i = 0; i < k; ++i) {
        int64_t group_idx = top_k_candidates[i].second; // .second is group_idx
        activated_group_mask[group_idx] = 1;
    }

    return true;

}

bool layer_cache::update_DFR(){

    // 1. iterate through activated_group_mask, and update group_dfr_tracker
    //      DFR equation:
    //      if activated (mask == 1): new_value = current_value * decay_rate + bonus; 
    //      else (mask == 0)        : new_value = current_value * decay_rate
    for(int64_t i=0; i<layer_group_count; i++){
        group_dfr_tracker[i] = group_dfr_tracker[i] * dfr_decay + activated_group_mask[i] * dfr_bonus;
    }

    // 2. clear group_dfr_heap, and push all groups int64_to it
    while(!group_dfr_heap.empty()){
        group_dfr_heap.pop();
    }
    for(int64_t i=0; i<layer_group_count; i++){
        group_dfr_heap.push(DFR_Node(i, group_dfr_tracker[i]));
    }
    return true;

}

bool layer_cache::update_slots(){

    // 1. compute groups_to_reload according to activated_group_mask and group_to_slot_map(to make sure whether a group is already in a slot)
    groups_to_reload.clear();
    for(int64_t i=0; i<layer_group_count; i++){
        if(activated_group_mask[i] == 1 && group_to_slot_map[i] == -1){
            groups_to_reload.push_back(i);
        }
    }

    // 2. compute slots_for_reload according to the size of groups_to_reload, select the k-th smallest dfr score groups to evict, and find their corresponding slots idx
    slots_for_reload.clear();
    int64_t k = groups_to_reload.size();
    if(k == 0) return true;

    std::vector<int64_t> groups_to_evict;
    groups_to_evict.reserve(k); // pre-allocate to boost efficiency

    // We need to evict some groups && their are still enought slots for eviction
    while (groups_to_evict.size() < k && !group_dfr_heap.empty()) {
        DFR_Node top = group_dfr_heap.top();
        group_dfr_heap.pop();

        // Only when this group is currently on gpu can you evict it.
        if (group_to_slot_map[top.second] != -1) {
            groups_to_evict.push_back(top.second);
        }
    }

    // Security check: if there isn't enough slots to be evicted, show error.
    if (groups_to_evict.size() < k) {
        LLAMA_LOG_ERROR("%s: could not find enough slots to evict\n", __func__);
        return false;
    }

    // 3. update group_to_slot_map and slot_to_group_map according to groups_to_reload and slots_for_reload (make sure the mapping is one-to-one)
    for(int64_t i=0; i<k; i++){
        int64_t group_idx = groups_to_reload[i];
        int64_t evict_group_idx = groups_to_evict[i];
        int64_t slot_idx = group_to_slot_map[evict_group_idx];

        // update group_to_slot_map
        group_to_slot_map[group_idx] = slot_idx;
        group_to_slot_map[evict_group_idx] = -1;

        // update slot_to_group_map
        slot_to_group_map[slot_idx] = group_idx;
    }
    return true;
}

bool layer_cache::update_idx_and_mask(){

    // ====================================================
    // 1. update gpu_neu_idx/mask_vec according to group_to_slot_map, slot_to_group_map and group_to_neurons_map
    // ====================================================

    // 1.1 update gpu_neu_idx_vec
    gpu_neu_idx_vec.clear();
    for(int64_t slot_idx=0; slot_idx<layer_group_capacity; slot_idx++){
        int64_t group_idx = slot_to_group_map[slot_idx];
        if(group_idx != -1){
            for(int64_t j=0; j<layer_group_size; j++){
                int64_t neuron_idx = group_to_neurons_map[group_idx][j];
                gpu_neu_idx_vec.push_back(neuron_idx);
            }
        }
    }

    // 1.2 update gpu_neu_mask_vec
    std::fill(gpu_neu_mask_vec.begin(), gpu_neu_mask_vec.end(), 0);
    for(int64_t slot_idx=0; slot_idx<layer_group_capacity; slot_idx++){
        int64_t group_idx = slot_to_group_map[slot_idx];
        if(group_idx != -1){
            for(int64_t j=0; j<layer_group_size; j++){
                int64_t neuron_idx = group_to_neurons_map[group_idx][j];
                gpu_neu_mask_vec[neuron_idx] = 1;
            }
        }
    }

    // ====================================================
    // 2. copy gpu_neu_idx/mask_vec back to ffn_gpu_neu_idx/mask
    // ====================================================
    memcpy(ffn_gpu_neu_idx->data, gpu_neu_idx_vec.data(), gpu_neu_idx_vec.size() * sizeof(int64_t));
    memcpy(ffn_gpu_neu_mask->data, gpu_neu_mask_vec.data(), gpu_neu_mask_vec.size() * sizeof(int64_t));
    return true;

}

sparkinfer_reload_plan layer_cache::generate_reload_plan(){
    return sparkinfer_reload_plan(groups_to_reload, slots_for_reload, gpu_neu_idx_vec, gpu_neu_mask_vec);
}

sparkinfer_cache_manager::sparkinfer_cache_manager(const llama_model & model, const llama_cparams & params){
    n_layer = model.hparams.n_layer;
    for(int64_t i=0; i<n_layer; i++){
        layer_caches.push_back(new layer_cache(model.layers[i], i, dfr_decay, dfr_bonus));
    }
}