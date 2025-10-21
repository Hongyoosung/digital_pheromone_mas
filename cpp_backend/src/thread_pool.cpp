#include "thread_pool.hpp"

namespace pheromone {

ThreadPool::ThreadPool(size_t num_threads)
    : stop_(false), pending_count_(0) {

    // Create worker threads
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);

                    // Wait for a task or stop signal
                    condition_.wait(lock, [this] {
                        return stop_.load() || !tasks_.empty();
                    });

                    // Exit if stopped and no tasks remain
                    if (stop_.load() && tasks_.empty()) {
                        return;
                    }

                    // Get next task
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }

                // Execute task outside the lock
                task();
                pending_count_.fetch_sub(1);
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    stop_.store(true);
    condition_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

size_t ThreadPool::pending_tasks() const {
    return pending_count_.load();
}

} // namespace pheromone
