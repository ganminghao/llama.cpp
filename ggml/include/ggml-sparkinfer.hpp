#pragma once

#include "ggml.h"

#include <deque>
#include <functional>
#include <future>
#include <thread>

enum sparkinfer_weight_type { SPIF_FFN_UP = 1, SPIF_FFN_GATE, SPIF_FFN_DOWN };

typedef struct {
    int n, m, g;   // neuron_count, neuron_cache_size, group_size
    int n_g, m_g;  // group_count, group_cache_size
} cache_meta;

typedef struct {
    int weight_idx, cache_idx;
} copy_pair;

struct sparkinfer_layer_cache {
    ggml_tensor * layer_ffn_pred_up     = nullptr;
    ggml_tensor * layer_ffn_pred_down   = nullptr;
    ggml_tensor * layer_ffn_pred_up_b   = nullptr;
    ggml_tensor * layer_ffn_pred_down_b = nullptr;

    ggml_tensor * layer_ffn_up     = nullptr;
    ggml_tensor * layer_ffn_gate   = nullptr;
    ggml_tensor * layer_ffn_down   = nullptr;
    ggml_tensor * layer_ffn_up_b   = nullptr;
    ggml_tensor * layer_ffn_gate_b = nullptr;
    ggml_tensor * layer_ffn_down_b = nullptr;

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

    static const bool k_enable_spif_reload;
    size_t            reload_cnt      = 0;
    size_t            reload_wnd_size = 8;

    ggml_tensor * weight_only_buf;
    ggml_tensor * cache_only_buf;
    ggml_tensor * neuron_idx_buf;

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

    template <class F, class... Args> void post(F && f, Args &&... args) {
        enqueue_io(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    }

    template <class F, class... Args> auto submit(SparkinferWaitType wait_type, F && f, Args &&... args) {
        using R = std::invoke_result_t<F, Args...>;

        std::packaged_task<R()> task(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        auto                    fut      = task.get_future();
        auto                    task_ptr = std::make_shared<std::packaged_task<R()>>(std::move(task));
        std::function<void()>   wrapper  = [task_ptr]() {
            (*task_ptr)();
        };

        bool need_notify = false;

        {
            std::lock_guard<std::mutex> lock(mtx_);
            AnchorState &               anchor = anchor_ref(wait_type);

            if (!anchor.has_anchor || !anchor.active) {
                tasks_.push_back(std::move(wrapper));
                need_notify = true;
            } else {
                anchor.pending.push_back(std::move(wrapper));
            }
        }

        if (need_notify) {
            cv_.notify_one();
        }

        return fut;
    }

    void make_anchor(SparkinferWaitType wait_type) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            AnchorState &               anchor = anchor_ref(wait_type);

            GGML_ASSERT(anchor.pending.empty());

            anchor.has_anchor = true;
            anchor.active     = true;
        }

        enqueue_io([this, wait_type] {
            std::deque<std::function<void()>> to_move;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                AnchorState &               anchor = anchor_ref(wait_type);

                to_move.swap(anchor.pending);
                anchor.active = false;

                for (auto & fn : to_move) {
                    tasks_.push_back(std::move(fn));
                }
            }

            if (!to_move.empty()) {
                cv_.notify_one();
            }
        });
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (!worker_.joinable()) {
                return;
            }
            tasks_.push_back(std::function<void()>{});
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

    AnchorState & anchor_ref(SparkinferWaitType wait_type) {
        switch (wait_type) {
            case SPIF_WAIT_MUL_MAT_SPARSE:
                return anchor_mm_sparse_;
            case SPIF_WAIT_AXPY_SPARSE:
                return anchor_axpy_sparse_;
            default:
                GGML_ABORT("anchor_ref: invalid wait_type");
        }
    }

    std::mutex                        mtx_;
    std::condition_variable           cv_;
    std::deque<std::function<void()>> tasks_;
    std::deque<std::function<void()>> io_tasks_;
    std::thread                       worker_;

    void enqueue(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            tasks_.push_back(std::move(fn));
        }
        cv_.notify_one();
    }

    void enqueue_io(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            io_tasks_.push_back(std::move(fn));
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
