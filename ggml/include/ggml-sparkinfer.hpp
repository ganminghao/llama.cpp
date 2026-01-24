#pragma once

#include "ggml.h"

#include <algorithm>
#include <deque>
#include <future>

enum sparkinfer_weight_type { SPIF_FFN_UP = 1, SPIF_FFN_GATE, SPIF_FFN_DOWN };

typedef struct {
    int n, m, g;   // n_neuron, n_neuron_cache, n_ffn_group
    int n_g, m_g;  // n_group, n_group_cache
} cache_meta;

typedef struct {
    int weight_idx, cache_idx;
} copy_pair;

inline int get_env_int(const char * env, int default_value) {
    if (const char * p = getenv(env)) {
        return atoi(p);
    }
    return default_value;
}

const int    k_spif_dfr_debug          = get_env_int("SPIF_DFR_DEBUG", 0);
const float  k_spif_init_dfr_decay     = get_env_int("SPIF_INIT_DFR_DECAY", 67) / 100.0f;
const float  k_spif_dx_dfr_decay       = get_env_int("SPIF_DX_DFR_DECAY", 50) / 1000.0f;
const size_t k_spif_reload_window_size = get_env_int("SPIF_RELOAD_WINDOW_SIZE", 4);

struct sparkinfer_layer_cache {
    ggml_tensor * ffn_pred_up     = nullptr;
    ggml_tensor * ffn_pred_down   = nullptr;
    ggml_tensor * ffn_pred_up_b   = nullptr;
    ggml_tensor * ffn_pred_down_b = nullptr;

    ggml_tensor * ffn_up     = nullptr;
    ggml_tensor * ffn_gate   = nullptr;
    ggml_tensor * ffn_down   = nullptr;
    ggml_tensor * ffn_up_b   = nullptr;
    ggml_tensor * ffn_gate_b = nullptr;
    ggml_tensor * ffn_down_b = nullptr;

    ggml_tensor * ffn_up_cache   = nullptr;
    ggml_tensor * ffn_gate_cache = nullptr;
    ggml_tensor * ffn_down_cache = nullptr;

    ggml_tensor * sparse_idx  = nullptr;
    ggml_tensor * reload_up   = nullptr;
    ggml_tensor * reload_gate = nullptr;
    ggml_tensor * reload_down = nullptr;

    ggml_tensor * neuron_idx  = nullptr;
    ggml_tensor * group_maps  = nullptr;
    ggml_tensor * neuron_mask = nullptr;
    ggml_tensor * group_mask  = nullptr;
    ggml_tensor * dfr_scores  = nullptr;

    cache_meta  layer_cm;
    copy_pair * reload_plan;
    bool        gpu_only;

    const static bool  k_enable_spif_reload;
    size_t             reload_count       = 0;
    size_t             reload_window_size = k_spif_reload_window_size;
    std::deque<size_t> reload_records;
    std::deque<float>  dfr_decay_records;

    ggml_tensor * weight_only_buf = nullptr;
    ggml_tensor * cache_only_buf  = nullptr;
    ggml_tensor * neuron_idx_buf  = nullptr;
    ggml_tensor * dfr_decay_pack  = nullptr;

    sparkinfer_layer_cache()  = default;
    ~sparkinfer_layer_cache() = default;

    ggml_tensor * build_reload_plan(ggml_context * ctx0, ggml_tensor * weight_only, ggml_tensor * cache_only);
    ggml_tensor * build_reload_exec(ggml_context * ctx0, ggml_tensor * cur, sparkinfer_weight_type spif_wt);
    void          sparkinfer_reload_plan();
};

// sparkinfer async kernel caller and io executor
struct SingleThreadExecutor {
    enum SparkinferWaitType { SPIF_WAIT_MUL_MAT_SPARSE = 0, SPIF_WAIT_AXPY_SPARSE };

    SingleThreadExecutor() {
        worker_ = std::thread([this] { loop(); });
    }

    SingleThreadExecutor(const SingleThreadExecutor &)             = delete;
    SingleThreadExecutor & operator=(const SingleThreadExecutor &) = delete;
    SingleThreadExecutor(SingleThreadExecutor &&)                  = delete;
    SingleThreadExecutor & operator=(SingleThreadExecutor &&)      = delete;

    ~SingleThreadExecutor() { stop(); }

    template <class F, class... Args> static auto make_bound(F && f, Args &&... args) {
        using Fn  = std::decay_t<F>;
        using Tup = std::tuple<std::decay_t<Args>...>;

        return [fn = Fn(std::forward<F>(f)), tup = Tup(std::forward<Args>(args)...)]() mutable {
            return std::apply(fn, tup);
        };
    }

    template <class F, class... Args> void post(F && f, Args &&... args) {
        auto bound = make_bound(std::forward<F>(f), std::forward<Args>(args)...);
        enqueue_io(std::move(bound));
    }

    template <class F, class... Args> auto submit(SparkinferWaitType wait_type, F && f, Args &&... args) {
        using R = std::invoke_result_t<F, Args...>;

        auto bound    = make_bound(std::forward<F>(f), std::forward<Args>(args)...);
        auto task_ptr = std::make_shared<std::packaged_task<R()>>(std::move(bound));
        auto fut      = task_ptr->get_future();
        auto wrapper  = [task_ptr]() {
            (*task_ptr)();
        };

        bool          need_notify = false;
        AnchorState * anchor      = anchor_ref(wait_type);

        {
            std::lock_guard<std::mutex> lock(mtx_);

            if (!anchor->has_anchor || !anchor->active) {
                tasks_.emplace_back(std::move(wrapper));
                need_notify = true;
            } else {
                anchor->pending.emplace_back(std::move(wrapper));
            }
        }

        if (need_notify) {
            cv_.notify_one();
        }

        return fut;
    }

    void make_anchor(SparkinferWaitType wait_type, float * dfr_decay) {
        AnchorState * anchor = anchor_ref(wait_type);

        {
            std::lock_guard<std::mutex> lock(mtx_);
            GGML_ASSERT(anchor->pending.empty());
            anchor->has_anchor = true;
            anchor->active     = true;
        }

        enqueue_io([this, anchor, dfr_decay] {
            std::deque<std::function<void()>> to_move;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                to_move.swap(anchor->pending);
                anchor->active = false;

                for (auto & fn : to_move) {
                    tasks_.emplace_back(std::move(fn));
                }
            }

            if (!to_move.empty()) {
                cv_.notify_one();
            }

            if (k_spif_dx_dfr_decay > 0.0f && dfr_decay) {
                dfr_decay[0] *= 1.0f + (to_move.empty() ? -k_spif_dx_dfr_decay : k_spif_dx_dfr_decay);
                dfr_decay[0] = std::clamp(dfr_decay[0], 0.05f, 0.95f);
                dfr_decay[1] = 1.0f - dfr_decay[0];
            }
        });
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (!worker_.joinable()) {
                return;
            }
            tasks_.emplace_back(std::function<void()>{});
        }
        cv_.notify_one();
        worker_.join();
    }

    struct AnchorState {
        bool                              has_anchor = false;
        bool                              active     = false;
        std::deque<std::function<void()>> pending;
    };

    AnchorState anchor_mm_sparse_;
    AnchorState anchor_axpy_sparse_;

    AnchorState * anchor_ref(SparkinferWaitType wait_type) {
        switch (wait_type) {
            case SPIF_WAIT_MUL_MAT_SPARSE:
                return &anchor_mm_sparse_;
            case SPIF_WAIT_AXPY_SPARSE:
                return &anchor_axpy_sparse_;
            default:
                GGML_ABORT("anchor_ref: invalid wait_type");
        }
    }

    std::mutex                        mtx_;
    std::condition_variable           cv_;
    std::deque<std::function<void()>> tasks_;
    std::deque<std::function<void()>> io_tasks_;
    std::thread                       worker_;

    template <class F> void enqueue(F && fn) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            tasks_.emplace_back(std::forward<F>(fn));
        }
        cv_.notify_one();
    }

    template <class F> void enqueue_io(F && fn) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            io_tasks_.emplace_back(std::forward<F>(fn));
        }
        cv_.notify_one();
    }

    void loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this] { return !tasks_.empty() || !io_tasks_.empty(); });

                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                } else if (!io_tasks_.empty()) {
                    task = std::move(io_tasks_.front());
                    io_tasks_.pop_front();
                } else {
                    continue;
                }
            }

            if (!task) {
                break;
            }

            task();
        }
    }
};
