# Reload 算子设计

> 本文已经过时，最新的算子设计请移步 [reload 设计文档](reload-design.md)

> 关于 sparkinfer_cache_manager 的设计思路，请移步 [sparkinfer_cache_manager 设计文档](sparkinfer_cache_manager_design.md)

YPX 的个人注释风格解释：

```cpp
// [YPX] [B]    Bug     : 发现的 Bug 或者需要优化的地方
// [YPX] [C]    Comment : 普通注释，无特别含义
// [YPX] [Q]    Question: 疑问，需要进一步确认的地方
// [YPX] [Todo] Todo    : 已经确定好的待办事项
```

## 1. 三种 map 方式

### 1.1 CPU 端 weights 保留原始权重，不按组重排（当前技术路线）

- 算子一：使用 `neuron_to_gpu_map` 来确定神经元和组之间的关系。
- 算子二：该算子只根据组下标进行计算，组和神经元之间的 map 方式对该算子无影响。
- 算子三：如何实现 group-wise reload 呢？对于每一个需要 reload 的组，需要这么做：
    1. 查阅 `gpu_to_neurons_map`，先在 CPU 上执行 `layer_group_size` 次 copy，把某一个组的神经元一个一个搬到统一的 tmp cache 里面
    2. 然后执行一个 H2D copy，把这 `layer_group_size` 个 neurons 一起传上去

### 1.2 CPU 端 weights 按组重排且不训练新 predictor（需要加入的支持之一）

- 这个方式最严重的问题在于，我们的 predictor 是基于旧的下标训练的。其给出的 `sparse_idx`，比如 `[2, 7, 5, 3]`，其中的 `7` 说的应该是原始权重中的第 `7` 个神经元，而不是 CPU 权重重排后的第 `7` 个神经元。所以，我们要么重新训练 predictor，使之输出的 `sparse_idx` 是新下标，要么我们就需要保留「神经元旧下标」和「神经元新下标」之间的映射结构，也就是 `old_to_new_map` 和 `new_to_old_map`

- 算子一：将 `sparse_idx` 里面的下标通过 `old_to_new_map` 将 predictor 给出的「神经元旧下标」转换为「神经元新下标」，并通过「神经元新下标 % `layer_group_size`」的值来确定神经元和组之间的关系
- 算子二：该算子只根据组下标进行计算，组和神经元之间的 map 方式对该算子无影响
- 算子三：实现 group-wise reload 变得无比容易，直接在原来的 CPU 权重张量上面执行 H2D copy 即可
    - 潜在问题：算子三要负责更新元数据，并给 GPU 提供足够其进行计算的信息。那么，我们需要为 GPU 提供 `old_to_new_map` 吗？
    - 似乎不需要哦！我们只需要在 CPU 端保留这个 `old_to_new_map` 即可
    - 为什么呢？因为 GPU 依赖 `ffn_neu_idx` 这个张量来进行计算，而在 CPU 端，我们设置填入 `ffn_neu_idx` 的神经元下标是原始权重文件中的下标即可。但是，我们怎么知道应该填入哪些「神经元新下标」呢？这就需要 `new_to_old_map` 了

### 1.3 CPU 端 weights 按组重排且训练新 predictor（需要加入的支持之二）

- 算子一：`sparse_idx` 里面是重排后的权重文件中的神经元的下标，也即「神经元新下标」，我们通过「神经元下标 % `layer_group_size`」的值来确定神经元和组之间的关系
- 算子二：该算子只根据组下标进行计算，组和神经元之间的 map 方式对该算子无影响
- 算子三：实现 group-wise reload 变得无比容易，直接在原来的 CPU 权重张量上面执行 H2D copy 即可
    - `ffn_neu_idx` 这个张量里填入的神经元下标是重排后的权重文件中的下标

### 1.4 解决三种 map 方式的方法：自定义宏

```cpp
// In "llama-sparkinfer.h"
#define SPARKINFER_WEIGHT_LAYOUT_ORIGINAL         0
#define SPARKINFER_WEIGHT_LAYOUT_REORDERED_COMPAT 1
#define SPARKINFER_WEIGHT_LAYOUT_REORDERED_NATIVE 2

// ===================================================================
//  !!! 全局唯一的控制开关 !!!
//  通过修改这个值，然后重新编译，即可切换所有相关算子的行为
#define SPARKINFER_WEIGHT_LAYOUT_MODE SPARKINFER_WEIGHT_LAYOUT_ORIGINAL
// ===================================================================
```

## 2. Reload 计算图节点调用链和其他支撑结构

`ggml_graph_compute()` -> `ggml_graph_compute_thread()` -> `ggml_compute_forward()` -> `reload_weights()`(或其他具体实现函数)

## 3. Reload 实现函数设计

### 3.1 总体思路

如果全部在一个算子里面实现，那就需要传入 12+ 个输入张量，而 GGML_MAX_SRC 只能支持 10 个输入张量。

```cpp
struct ggml_tensor * gpu_weights            = tensor->src[0]; // [YPX] [C] 3 个
struct ggml_tensor * cpu_weights            = tensor->src[1]; // [YPX] [C] 3 个
struct ggml_tensor * gpu_bias               = tensor->src[2]; // [YPX] [C] 2 个
struct ggml_tensor * cpu_bias               = tensor->src[3]; // [YPX] [C] 2 个
struct ggml_tensor * sparse_idx             = tensor->src[4];
struct ggml_tensor * gpu_neu_idx            = tensor->src[5];
struct ggml_tensor * gpu_neu_mask           = tensor->src[6];
struct ggml_tensor * DFR_score              = tensor->src[7];
struct ggml_tensor * group_to_neurons_map   = tensor->src[8];
struct ggml_tensor * neuron_to_group_map    = tensor->src[9];
struct ggml_tensor * group_to_slot_map      = tensor->src[10];
struct ggml_tensor * slot_to_group_map      = tensor->src[11];
```

方向一：改 `GGML_MAX_SRC` 的值（存在风险）。

方向二：需要拆成多个算子。三个算子均实现在 [`ggml-cpu.c`](../ggml/src/ggml-cpu/ggml-cpu.c) 中。

### 3.2 算子一：GGML_RELOAD_PLAN

> 有关 Reload Strategy 的选择，请参见：[Reload 策略设计](reload_strategy_design.md)

```cpp
// In "llama-sparkinfer.h"
#define SPARKINFER_RELOAD_USE_DFR           0
#define SPARKINFER_RELOAD_USE_LRU           1
#define SPARKINFER_RELOAD_USE_FIFO          2
#define SPARKINFER_RELOAD_USE_OPT           3
#define SPARKINFER_RELOAD_USE_CLOCK         4
#define SPARKINFER_RELOAD_USE_NRU           5
#define SPARKINFER_RELOAD_USE_PLACEHOLDER   6

// ===================================================================
//  !!! 全局唯一的控制开关 !!!
//  通过修改这个值，然后重新编译，即可切换所有相关算子的行为
#define SPARKINFER_RELOAD_STRATEGY SPARKINFER_RELOAD_USE_DFR
// ===================================================================
```

输入：

```cpp
struct ggml_tensor * sparse_idx             ;
struct ggml_tensor * gpu_neu_idx            ;
struct ggml_tensor * gpu_neu_mask           ;
struct ggml_tensor * DFR_score              ;
struct ggml_tensor * group_to_neurons_map   ; // [YPX] [C] 后续 CPU 权重重排后可能不需要了
struct ggml_tensor * neuron_to_group_map    ;
// struct ggml_tensor * old_to_new_map       ; // [YPX] [C] 后续 CPU 权重重排后，CPU 可能用不到，但是可能需要传给 GPU？
```

输出：

```cpp
struct ggml_tensor * groups_to_reload       ; // int32
```

技术难点：

- 主要的技术点在于遍历统计和 Top-k 排序，我应该参考哪个算子？或者，我应该使用 C 语言自带的排序算法 qsort？
- 另外，我需要创建一个数据结构来存储 `group_stat`，C 语言下的 `pair<int, int>`(`count, group_idx`) 实现。

伪代码：

```cpp
// --- 具体的实现函数 ---
// 实现1: 针对原始布局
static inline void _sparkinfer_reload_plan_count_activated_neurons_original(...) {
    // Logic using neuron_to_group_map
}

// 实现2: 针对重排兼容布局
static inline void _sparkinfer_reload_plan_count_activated_neurons_reordered_compat(...) {
    // Logic using new_idx = old_to_new_map[old_idx] and then new_idx / group_size
}

// --- 门面函数 ---
// 这个函数是上层逻辑的唯一入口，它的内部实现由宏在编译时确定
// [YPX] [C] 使用 inline，避免栈开销，爽！
static inline void sparkinfer_reload_plan_count_activated_neurons(...) {
#if SPARKINFER_WEIGHT_LAYOUT_MODE == SPARKINFER_WEIGHT_LAYOUT_ORIGINAL
    _sparkinfer_reload_plan_count_activated_neurons_original(...);
#elif SPARKINFER_WEIGHT_LAYOUT_MODE == SPARKINFER_WEIGHT_LAYOUT_REORDERED_COMPAT
    _sparkinfer_reload_plan_count_activated_neurons_reordered_compat(...);
#elif SPARKINFER_WEIGHT_LAYOUT_MODE == SPARKINFER_WEIGHT_LAYOUT_REORDERED_NATIVE
    // ... For future implementation
#endif
}

static inline void _sparkinfer_reload_plan_get_groups_by_dfr(...) {
    // [YPX] [C] 省略具体逻辑
}

// [YPX] [C] 此处省略剩余方法

static inline void sparkinfer_reload_plan_get_groups_to_reload(...){
#if SPARKINFER_RELOAD_STRATEGY == SPARKINFER_RELOAD_USE_DFR
    _sparkinfer_reload_plan_get_groups_by_dfr(...);
#elif
    // [YPX] [C] 此处省略剩余方法
#endif
}

// --- 你的主规划函数 ---
void sparkinfer_reload_plan(...) {
    // 1. 调用门面函数，完全无需关心底层是哪种布局
    sparkinfer_reload_plan_count_activated_neurons(...);

    // 2. 调用与布局无关的函数
    sparkinfer_reload_plan_get_groups_to_reload(...);
}
```

### 3.3 算子二：GGML_RELOAD_EVICT

输入：

```cpp
struct ggml_tensor * groups_to_reload_idx   ; // int32
struct ggml_tensor * group_to_slot_map      ;
struct ggml_tensor * slot_to_group_map      ;
```

输出：

```cpp
struct ggml_tensor * slots_to_evict_idx     ; // int32, 不定长
```

> ⚠️ **注意**：算子二虽然名字里面带一个 Evict，但是 **它并不做任何 Evict 的操作** ，它只是计算出哪些 slot 需要被 Evict。

技术难点：

- 主要的技术点在于高效地对比找出要被 evict 的 slots，我应该如何设计以避免 O(n^2) 的复杂度？
- 我的初步想法：
    - 首先，我们需要处理哪些数据结构？
        - `groups_to_reload_idx`：这是一个包含需要重新加载的组的列表，是 group_idx 的数组，对 group_idx 无序。
        - `group_to_slot_map`：这是一个映射，表示每个组当前所在的槽位，-1 表示不在 GPU 上，按 group_idx 升序，对 slot_idx 无序。
        - `slot_to_group_map`：这是一个映射，表示每个槽位当前存储的组，应当都被合法的 group_idx 填充，对 group_idx 无序，按 slot_idx 升序。
        - 目标是找出 `slots_to_evict_idx`，即那些当前存储的组不在 `groups_to_reload_idx` 列表中的槽位。
    - 第 1 步：创建 `groups_to_evict_mask` 张量，大小等于 `layer_group_count`，用 `0` 填充。
    - 第 2 步：创建 `slot_to_evict_idx` 张量，其瞬时大小为 `slot_to_evict_total_num`。不过貌似不需要创建，因为 `ggml_backend_sched` 已经帮我们创建好了？
    - 第 3 步：创建 `groups_to_reload_mask`，用 `0` 填充，遍历 `group_to_reload_idx`，将遍历到的组设置为 `1`。
    - 第 3 步：使用双指针算法，第一个指针 `pt_group_idx` 遍历 `groups_to_reload_idx`，第二个指针 `pt_slot_idx` 遍历 `slots_to_group_map`，当第一个指针发现自己所表示的组没有在 GPU 上的时候，移动第二个指针直到找到一个不需要被 reload 的组，并将这个组的 slot 编号存进 `slot_to_evict_idx` 里面。
- 伪代码表示如下：

```c
int pt_group_idx            = 0;
int pt_slot_idx             = 0;
int slot_to_evict_total_num = 0;
for(pt_group_idx = 0; pt_group_idx < groups_to_reload_idx->ne[0]; ++pt_group_idx){
    if(group_to_slot_map[groups_to_reload_idx[pt_group_idx]] == -1){
        while(pt_slot_idx < layer_group_capacity /* or another name, such as `gpu_group_capacity` */){
            if(groups_to_reload_mask[slot_to_group_map[pt_slot_idx]] == 0){
                slot_to_evict_idx[slot_to_evict_total_num++] = pt_slot_idx++;
                // [YPX] [Q] 是否要在这里加入更新 slot_to_group_map 的逻辑？
                break;
            }
            ++pt_slot_idx;
        }
        // [YPX] [Q] 是否要在这里加入更新 group_to_slot_map 的逻辑？
    }
}
GGML_ASSERT("这里 ASSERT 什么比较好？");
```

Gemini 优化的版本：

```c
/* 在你的 sparkinfer_reload_evict 函数中 */

// --- 准备工作 ---
const int64_t * groups_to_reload_data = (const int64_t *)groups_to_reload->data;
const int num_groups_to_reload = groups_to_reload->ne[0]; // 假设这是不定长方案中的实际长度

const int32_t * group_to_slot_map_data = (const int32_t *)group_to_slot_map->data;
const int32_t * slot_to_group_map_data = (const int32_t *)slot_to_group_map->data;

// [YPX] [B] slots_to_evict_idx 应该是这个 tensor 的输出张量，也就是 tensor->data 而不是 slot_to_evict->data 了，它的长度到底怎么确定，或者说，怎么设置？改 tensor->ne[0]？
int32_t * slots_to_evict_data = (int32_t *)slots_to_evict->data;

const int num_total_groups = group_to_slot_map->ne[0];
const int num_gpu_slots = slot_to_group_map->ne[0];

// --- 步骤 1: 统计有多少个热门组已经在 GPU 上，多少个需要新槽位 ---
int num_groups_needing_slots = 0;
for (int i = 0; i < num_groups_to_reload; ++i) {
    int64_t group_id = groups_to_reload_data[i];
    if (group_to_slot_map_data[group_id] == -1) {
        num_groups_needing_slots++;
    }
}

// --- 步骤 2: 创建一个 mask 来快速判断一个组是不是热门组 (O(N_reload)) ---
// 使用 alloca 在栈上分配临时内存，比 malloc 更快，函数结束自动释放
// [YPX] [B] 这里其实不应该用 alloca，因为 ggml_compute_params 有一个 wdata 可以用！而且 bool 是什么鬼？这是 ggml_cpu.c，不是 ggml_cpu.cpp！
bool *is_hot_group_mask = (bool *)alloca(sizeof(bool) * num_total_groups);
memset(is_hot_group_mask, 0, sizeof(bool) * num_total_groups);

for (int i = 0; i < num_groups_to_reload; ++i) {
    int64_t group_id = groups_to_reload_data[i];
    is_hot_group_mask[group_id] = true;
}

// --- 步骤 3: 遍历所有 GPU 槽位，找出可以被 evict 的“冷门”槽位 (O(N_slots)) ---
int num_slots_found_to_evict = 0;
for (int slot_idx = 0; slot_idx < num_gpu_slots; ++slot_idx) {
    // 如果需要 evict 的槽位已经找够了，就提前退出
    if (num_slots_found_to_evict >= num_groups_needing_slots) {
        break;
    }

    int group_id_in_slot = slot_to_group_map_data[slot_idx];

    // 如果这个槽位里的组不是热门组，那么它就可以被 evict
    if (!is_hot_group_mask[group_id_in_slot]) {
        slots_to_evict_data[num_slots_found_to_evict] = slot_idx;
        num_slots_found_to_evict++;
    }
}

// --- 步骤 4: 完整性断言 ---
// [YPX] [Q] 是否要在这里加入更新 ..._map 的逻辑？
// [回答] **绝对不要！** 这个算子的唯一职责就是“计算”出 slots_to_evict。
// 任何修改状态的操作（更新 map）都应该在最后的 GGML_RELOAD_EXEC 算子中，
// 在数据拷贝确认完成后，原子性地完成。这叫做“职责分离”，能让你的代码更健壮。
// [YPX] [Todo] 如果是这样的话，要考虑一个问题：算子三有没有足够的信息来更新？我们需要传入 slot_to_group_map 和 group_to_slot_map，并且要考虑会不会爆 GGML_MAX_SRC。如果分成 4 个算子，就没有原子性了。如果改动 GGML_MAX_SRC 的话，就根本不需要设置三个算子了。

// [YPX] [Q] GGML_ASSERT("这里 ASSERT 什么比较好？");
// [回答] 断言“我们找到的可供驱逐的槽位数量”，必须等于“我们需要新槽位的热门组数量”。
// 如果不等，说明我们的缓存逻辑出现了不一致的致命错误。
GGML_ASSERT(num_slots_found_to_evict == num_groups_needing_slots && "Eviction plan failed: mismatch between slots needed and slots found");

// [YPX] [Todo] 如果使用不定长方案，还需要把 num_slots_found_to_evict 写入一个 size 张量，或者在 slots_to_evict_data 末尾写入 -1 标记。
```

### 3.4 算子三：GGML_RELOAD_EXEC

输入：

```cpp
struct ggml_tensor * groups_to_reload       ; // [YPX] [C] 来自算子一
struct ggml_tensor * slots_to_evict         ; // [YPX] [C] 来自算子二
struct ggml_tensor * gpu_weights            ; // [YPX] [C] 调用三次，分别对应三个 weight
struct ggml_tensor * cpu_weights            ; // [YPX] [C] 调用三次，分别对应三个 weight
struct ggml_tensor * gpu_bias               ; // [YPX] [C] 调用两次，分别对应两个 bias
struct ggml_tensor * cpu_bias               ; // [YPX] [C] 调用两次，分别对应两个 bias
```

输出：

```cpp
// [YPX] [C] 无输出，直接修改 gpu_weights 和 gpu_bias
```

Gemini 优化的第二个版本（实际采用的版本）

```cpp
// --- 具体的实现函数 ---
static inline void _sparkinfer_reload_exec_aggregate_original(...) {
    // 分散拷贝：使用 group_to_neurons_map 找到神经元，然后逐个拷贝到临时缓冲区
}

static inline void _sparkinfer_reload_exec_aggregate_reordered(...) {
    // 连续拷贝：直接计算偏移量，进行一次 memcpy
}

static inline void _sparkinfer_reload_exec_update_metadata_original(...) {
    // 用原始下标更新 ffn_gpu_neu_idx
}

static inline void _sparkinfer_reload_exec_update_metadata_reordered_compat(...) {
    // 同样是用原始下标更新 ffn_gpu_neu_idx (因为GPU侧需要原始下标)
}

static inline void _sparkinfer_reload_exec_update_metadata_reordered_native(...) {
    // 用重排后的新下标更新 ffn_gpu_neu_idx
}


// --- 门面函数 ---
static inline void sparkinfer_reload_exec_aggregate(...) {
#if SPARKINFER_WEIGHT_LAYOUT_MODE == SPARKINFER_WEIGHT_LAYOUT_ORIGINAL
    _sparkinfer_reload_exec_aggregate_original(...);
#else // REORDERED_COMPAT 和 REORDERED_NATIVE 在聚合数据时逻辑是相同的
    _sparkinfer_reload_exec_aggregate_reordered(...);
#endif
}

static inline void sparkinfer_reload_exec_update_metadata(...) {
#if SPARKINFER_WEIGHT_LAYOUT_MODE == SPARKINFER_WEIGHT_LAYOUT_ORIGINAL
    _sparkinfer_reload_exec_update_metadata_original(...);
#elif SPARKINFER_WEIGHT_LAYOUT_MODE == SPARKINFER_WEIGHT_LAYOUT_REORDERED_COMPAT
    _sparkinfer_reload_exec_update_metadata_reordered_compat(...);
#elif SPARKINFER_WEIGHT_LAYOUT_MODE == SPARKINFER_WEIGHT_LAYOUT_REORDERED_NATIVE
    _sparkinfer_reload_exec_update_metadata_reordered_native(...);
#endif
}


// --- 你的主执行函数 ---
void sparkinfer_reload_exec(...) {
    for (/* each group to reload */) {
        // 调用门面函数，主逻辑保持整洁
        sparkinfer_reload_exec_aggregate(...);
        // ... H2D copy ...
    }
    // 更新元数据
    sparkinfer_reload_exec_update_metadata(...);
}
```