# 后端调度

## 节点的区别

- **leaf**：如果 `op == NONE` 且不是参数（param），说明这是常量或输入，不会参与梯度计算，视为叶子节点
- **node**：存储动态数据，通常是中间计算结果和输入嵌入。
    - reload节点属于node节点

## 何时调度

- 将大批次（sbatch）拆分为多个小批次（ubatch），大小受 cparams.n_ubatch 限制。
- 对每个 ubatch：
    - 分配 KV 缓存 slot（kv_self->find_slot）。
    - 重置调度器并设置回调
    - 构建计算图（graph_build）：包括 attention、FFN 等层。
    - **分配后端内存`ggml_backend_sched_alloc_graph(sched.get(), gf);`。 HERE**
        - 图划分：`ggml_backend_sched_split_graph(sched, graph);`，将整个计算图划分为若干个 子图（splits），每个子图将被分配到同一个后端设备上执行。
        - 为每个子图分配内存：`ggml_backend_sched_alloc_splits(sched)`， 实际的内存分配阶段
    - 将输入绑定到图的输入tensor
    - 执行计算（graph_compute）。
    - 异步提取结果

## 图分割 ggml_backend_sched_split_graph

<!-- 
const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0; 

帮助开发者理解调度器如何将计算图拆分成多个“子图（splits）”，并为每个操作分配后端，从而诊断性能问题或多后端调度逻辑是否符合预期。 
-->

- Pass 1：为已有内存分配的张量分配后端，预分配
    - 如果张量已有 buffer（已分配内存），使用该 buffer 所在后端
    - 如果是 view 张量（如 reshape、slice），继承源张量（view_src）的后端
    - 如果是图的输入（input flag），默认放在最后一个后端（通常是 CPU）
    - 如果操作SRC涉及权重（weights），优先使用权重所在后端
        - 优先“卸载（offload）”到更高优先级后端，如CPU->GPU
    - 默认返回 -1（未分配）,eg:如中间张量、无权重、非输入

不会修改用户已经手动分配的后端

**结果**：
- 叶子：全部分配了后端（权重或偏置在模型初始化期间分配了缓冲区）。
- 节点：如果源包含叶子，继承叶子的后端（如果叶子有不同的后端，选择优先级最高的后端）。一些 CPU 上的节点可能被卸载到 GPU。
- 图输入：分配到 CPU。

可以查看 logs/log.txt

- Pass 2：“传播”已有后端分配，减少未分配（backend_id = -1）的节点数量
    - 双向传播（先 GPU 向下/向上，再所有后端向下/向上）
        第一轮：只传播 GPU（非 CPU）分配，方向：从前往后（down） + 从后往前（up）
        第二轮：传播所有后端（包括 CPU），方向：down + up
        这样确保：GPU 区域优先连成一片，CPU 仅在必要时使用
    - 跳过 view 类操作

- Pass 3：升级 + 补全未分配节点
    - 对已分配节点：尝试 升级（upgrade）到更高优先级（编号更小）的后端（如从 CPU 升级到 GPU）
    - 对未分配节点（backend_id == -1）：选择 支持该 op 且能兼容最多输入的后端

- Pass 4：兜底处理 view 和 src 的后端
    - view 继承源后端
    - src 继承消费者后端

- Pass 5 是将计算图划分为多个“连续同后端”的子图（splits），并在子图边界处自动插入数据拷贝操作（隐式或显式），以确保跨设备数据依赖正确。 

日志里有图划分的结果

```md
eg:第20层

## SPLIT #17: CUDA0 # 2 inputs: [blk.20.ffn_gate.weight (  86M)] [blk.20.ffn_gpu_neu_mask (  86K)] 

## SPLIT #18: CUDA0 # 1 inputs: [blk.20.ffn_up.weight (  86M)] 

## SPLIT #19: CUDA0 # 1 inputs: [blk.20.ffn_down_t.weight (  86M)] 

## SPLIT #32: CPU # 2 inputs: [ffn_norm-20 (  16K)] [pred_out-20 (  43K)] 

## SPLIT #33: CUDA0 # 0 inputs

## SPLIT #34: CPU # 1 inputs: [ffn_gate_sparse_gpu-20 (  43K)] 

## SPLIT #35: CUDA0 # 0 inputs

## SPLIT #36: CPU # 1 inputs: [ffn_up_sparse_gpu-20 (  43K)] 

## SPLIT #37: CUDA0 # 2 inputs: [ffn_gate_par-20 (  43K)] [ffn_down_sparse-20 (  16K)] 
```

应该这里把pred_out确保在前面的split就可以应该，计算图那调整


## 计算

- llama_context::graph_compute
    - ggml_backend_sched_graph_compute_async
        - ggml_backend_sched_compute_splits：在多个后端（如 CPU、GPU）上异步执行一个已被划分的计算图，同时处理张量在不同后端之间的数据拷贝与同步

## ggml_backend_sched_compute_splits

- 主循环-遍历所有 splits
    - 阶段 1：输入张量拷贝（跨后端），拷贝到缓冲区
        - 用户输入（GGML_TENSOR_FLAG_INPUT）同步
        - 中间张量（非用户输入）异步拷贝
    - 阶段 2：执行计算图
        - 直接异步执行整个 split 的子图。ggml_backend_graph_compute_async -> 调用对应后端的函数指针
            - ggml_backend_cpu_graph_compute
            - 
            
### ggml_graph_plan