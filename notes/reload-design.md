# Reload 设计文档

## 1. reload_plan 潜在的更高效实现——使用集合差集

### 函数功能转移

- 现在按照甘明昊的思路，由于 `op_params` 的强大特性，我们可以在算子函数里面调用上层类 `sparkinfer_cache_manage` 的成员函数。于是乎，我们决定把产生 `sparkinfer_reload_plan` 的逻辑（算子一、二的逻辑）转移到 `sparkinfer_cache_manager::spif_reload_plan()` 里面实现。

### 术语澄清

- 原本的算子一要产生的是 `groups_to_load`（真正要进行传输的组）而不是 `groups_to_reload`（应该在 GPU 上的组）。我认为，我应该把后面那个变量的名字改为 `groups_to_ensure`，表示应该在 GPU 上的组，用 `groups_to_reload` 表示应该在 GPU 上但实际上不在，需要被 reload 的组。我们用 `N` 表示 `layer_group_count`，用 `k` 表示 `layer_group_capacity`。

### `spif_cache` 和 算子函数 的交互方式

- 把 `groups_to_reload` 和 `slots_to_evict` 作为 `sparkInfer_layer_cahce` 的成员变量
- 把 `sparse_idx` 作为 `sparkInfer_layer_cahce` 的成员 `tensor`
- 每次进行 reload 时：
    - 算子函数执行拷贝，把 `sparse_idx` 从 GPU 拷贝下来，拷贝到 `spif_cache` 的那个 `ggml_tensor * sparse_idx` 中
    - 算子函数调用 `spif_cache->spif_reload_plan()`，返回一个 `reload_plan_result`
    - 算子函数访问 `spif_cache->spif_layer[li]->groups_to_reload`，以及 `slots_to_evict`，并直接 for 循环遍历它们，执行拷贝

`reload_plan_result` 的定义如下：

```cpp
// ggml-spif.h
struct reload_plan_result {
    std::vector<int> groups_to_reload;
    std::vector<int> slots_for_evict;
};
typedef struct reload_plan_result reload_plan_result;
```

### 管理类 `spif_cache` 的设计思路

成员变量：

- `std::vector<int> dfr_score(layer_group_count)`
- `ggml_tensor * sparse_idx`
- `std::vector<int> groups_to_reload`
- `std::vector<int> slots_for_reload`
- `float dfr_decay_rate`
- `std::unorderd_map<int, int> group_to_slot_hash`
- 其他指向 gpu weights cache 和 cpu weights cache 的 ggml_tensor 指针

### 成员函数 `spif_reload_plan()`

- **使用哪一种 Reload 策略**：可以在 [llama-sparkinfer.h](../src/llaam-sparkinfer.h)里面通过宏来设置。

核心逻辑：

1. 第一步：执行 `spif_reload_plan_count_activated_neurons()`，得到 `group_activated_neurons_count` 数组
    - 算法：遍历所有组和所有神经元，时间复杂度为 $O(N + k) \approx O(N)$
2. 第二步：执行 `spif_reload_plan_get_groups_to_ensure()` 函数，该函数调用 `spif_reload_plan_use_dfr()` 等等策略函数，得到 `groups_to_ensure` 数组
    - 算法：小根堆，每次排序的时候重建，时间复杂度为 $O(N \log k)$
3. 第三步：执行 `spif_reload_plan_get_groups_to_reload()` 和 `spif_reload_plan_get_slot_to_evict()`，根据 `groups_to_ensure` 数组和 `slot_to_group_map`，算出 `groups_to_reload` 和 `slot_to_evict`
    - 算法：新的思路，也就是直接把 `slot_to_group_map` 删除，把 `group_to_slot_map` 改为 `group_to_slot_hash`，首先把 `groups_to_ensure` 转换为 set，再和 `group_to_slot_hash` 的所有 key 组成的集合进行集合差集运算，然后对于差集出来的两个集合 `groups_to_reload` 和 `groups_to_evict`，我们遍历 `groups_to_evict`，在 `group_to_slot_hash` 里面查找其 slot 编号；时间复杂度为建立 `groups_to_ensure` 的 set 的时间复杂度 $O(k)$ + 获取 `group_to_slot_hash` 的所有 key 组成 set $O(k)$ + 差集运算的时间复杂度 $O(k)$ + 查找的总时间复杂度 $O(k)$ = $O(k)$
4. 第四步：执行 `spif_reload_plan_update_mappings()` 函数，如果上一步选择算法一，就更新 `slot_to_group_map` 和 `group_to_slot_map`，如果上一步选择算法二，就更新 `group_to_slot_hash`。

## 2. Reload 算子设计——如何执行异步拷贝？