KV cache

0. 在构建计算图的时候，store and get multi-decodes 全局的 ggml_tensors from kv_cache:
        // llama-graph.cpp line 1542
        // store to KV cache
        {
            ggml_build_forward_expand(gf, kv_self->cpy_k(ctx0, k_cur, il));
            ggml_build_forward_expand(gf, kv_self->cpy_v(ctx0, v_cur, il));
        }

        const auto & kq_mask = inp->get_kq_mask();

        // get kv tensors
        ggml_tensor * q = q_cur;
        ggml_tensor * k = kv_self->get_k(ctx0, il);
        ggml_tensor * v = kv_self->get_v(ctx0, il);

1. 在 graph_build 中, kv_cache 通过 llama_graph_context 中的 memory 取出来实例化：
        // llama-graph.cpp line 1540
        const llama_kv_cache_recurrent * kv_self = static_cast<const llama_kv_cache_recurrent *>(memory);

2. 这个 memory 在 llama_graph_context 初始化的时候通过 llm_graph_params 中的 memory 赋值
        // llama-graph.cpp
        memory           (params.memory),

3. params 来自于 llama-context.cpp graph_build 的时候构建
        llm_graph_result_ptr llama_context::graph_build(
                    ggml_context * ctx,
                    ggml_cgraph * gf,
            const llama_ubatch & ubatch,
                    llm_graph_type gtype) {
            return model.build_graph(
                    {
                        /*.ctx         =*/ ctx,
                        /*.arch        =*/ model.arch,
                        /*.hparams     =*/ model.hparams,
                        /*.cparams     =*/ cparams,
                        /*.ubatch      =*/ ubatch,
                        /*.sched       =*/ sched.get(),
                        /*.backend_cpu =*/ backend_cpu,
                        /*.cvec        =*/ &cvec,
                        /*.loras       =*/ &loras,
                        /*.memory      =*/ memory.get(),  // here!
                        /*.cross       =*/ &cross,
                        /*.n_outputs   =*/ n_outputs,
                        /*.cb          =*/ graph_get_cb(),
                    }, gf, gtype);
        }
        因此，llama_context 里面有这个 memory 成员：
        // llama-context.h
        std::unique_ptr<llama_memory_i> memory;

4. 这个 memory 成员在 llama-context 初始化的时候跟着初始化：
        // llama-context.cpp:
        // init the memory module
        if (!hparams.vocab_only) {
            llama_memory_params params_mem = {
                /*.type_k   =*/ params.type_k,
                /*.type_v   =*/ params.type_v,
                /*.swa_full =*/ params.swa_full,
            };

            memory.reset(model.create_memory(params_mem, cparams));
        }


下面仿照 kvcache 的管理加入 kvcache-like sparkinfer-cache-manager:

1. 新建 memory 子类在 llama-context 中
        std::unique_ptr<llama_memory_i> spif_cache_mng;

2(对应上面 4). 在 llama-context 初始化时候初始这个 spif_cache_mng
   仿照kvcache，spif_cache_mng 的初始化被 llama-model 里的函数调用
        //llama-context.cpp
        // init sparkinfer_cache_manager if use sparkinfer
        if (!hparams.vocab_only && llama_use_sparkinfer(&model)) {
           spif_cache_mng.reset(model.create_spif_cache_mng(cparams));
        }

        // llama-model.cpp
        llama_memory_i * llama_model::create_spif_cache_mng(llama_cparams & cparams) const {
                llama_memory_i * res;

                switch (arch) {
                        case LLM_ARCH_PRO_SPARSE_LLAMA:
                        case LLM_ARCH_OPT:
                        {
                        res = new sparkinfer_cache_manager(*this, cparams);
                        }
                        break;
                        default:
                        {
                        res = nullptr;
                        }
                }

                return res;
        }

3. 下面完善 sparkinfer_cache_manager 的构造函数，以及类的成员构建

        3.1 对于构造函数，目前传入的参数是 model 和 cparams 的引用，通过 model 足以初始化 layer_caches 中的 tensor 指针。
            TODO: layer_caches 和 sparkinfer_cache_manager()的构造函数实现 in llama-sparkinfer.cpp
        3.2 成员和成员函数的设计：
                目前成员设计只是最简单的 ggml_tensor *， 用 private？
                然后设计 get_xxx 和 reload_xxx? (注意：这里的 reload 不是真正的 reload 实现，只是标记这个节点要 reload，依旧是在 build_graph 过程中！)

4.
        4.1 先把 sparkinfer_cache_manager 这个成员先加入 llama_graph_params定义中）
        4.2 （对应上面 3）然后在构建 llama_graph_context 的时候由 llama_context 里面的 spif_cache_mng 指针赋值给 llama_graph_params
        4.3 在 llama_graph_context 里面加入成员：
                const llama_memory_i      * spif_cache;
        4.3 （对应上面 2）再由 llama_graph_params 赋值给 llama_graph_context
                spif_cache       (params.spif_cache)

5. 1. 在 graph_build 中, kv_cache 通过 llama_graph_context 中的 memory 取出来实例化：
        // llama-graph.cpp line 871
        const sparkinfer_cache_manager * spif_cache = static_cast<const sparkinfer_cache_manager*>(spif_cache);
        const layer_cache * spif_layer = spif_cache ? spif_cache->get_layer_cache(il) : nullptr;




