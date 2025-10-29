#pragma once

#include "ggml.h"

#include <deque>
#include <functional>
#include <future>
#include <thread>

enum sparkinfer_weight_type { SPIF_FFN_UP = 0, SPIF_FFN_GATE = 1, SPIF_FFN_DOWN = 2 };

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

    static const bool k_enable_spif_reload;
    cache_meta        layer_cm;
    copy_pair *       reload_plan;
    int               num_ops;

    ggml_tensor * weight_only_buf;
    ggml_tensor * cache_only_buf;
    ggml_tensor * neuron_idx_buf;

    sparkinfer_layer_cache()  = default;
    ~sparkinfer_layer_cache() = default;

    ggml_tensor * build_reload(ggml_context *         ctx0,
                               ggml_tensor *          weight_only,
                               ggml_tensor *          cache_only,
                               sparkinfer_weight_type spif_wt);
    void          sparkinfer_reload_plan(const float * weight_only, const float * cache_only, int32_t * neuron_idx);
};

// sparkinfer async kernel caller and io executor
struct SingleThreadExecutor {
    explicit SingleThreadExecutor(bool drop_on_stop = false) : drop_(drop_on_stop), stop_flag_(false) {
        worker_ = std::thread([this] { loop(); });
    }

    SingleThreadExecutor(const SingleThreadExecutor &)             = delete;
    SingleThreadExecutor & operator=(const SingleThreadExecutor &) = delete;

    ~SingleThreadExecutor() { stop(); }

    template <class F, class... Args> void post(F && f, Args &&... args) {
        enqueue_io(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    }

    template <class F, class... Args>
    auto submit(F && f, Args &&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;
        std::packaged_task<R()> task(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        auto                    task_future = task.get_future();
        auto                    task_ptr    = std::make_shared<std::packaged_task<R()>>(std::move(task));
        enqueue([task_ptr]() { (*task_ptr)(); });
        return task_future;
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (stop_flag_) {
                return;
            }
            stop_flag_ = true;
            if (drop_) {
                tasks_.clear();
                io_tasks_.clear();
            }
        }
        cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    std::thread             worker_;
    std::mutex              mtx_;
    std::condition_variable cv_;

    std::deque<std::function<void()>> tasks_;
    std::deque<std::function<void()>> io_tasks_;

    const bool drop_;
    bool       stop_flag_;

    void enqueue(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (stop_flag_) {
                throw std::runtime_error("executor stopped");
            }
            tasks_.push_back(std::move(fn));
        }
        cv_.notify_one();
    }

    void enqueue_io(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (stop_flag_) {
                throw std::runtime_error("executor stopped");
            }
            io_tasks_.push_back(std::move(fn));
        }
        cv_.notify_one();
    }

    void loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this] { return stop_flag_ || !tasks_.empty() || !io_tasks_.empty(); });

                if (stop_flag_ && ((tasks_.empty() && io_tasks_.empty()) || drop_)) {
                    break;
                }

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
            task();
        }
    }
};
