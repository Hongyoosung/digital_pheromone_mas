#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <stdexcept>

namespace pheromone {

/**
 * Thread pool for parallel task execution
 * Based on the implementation guide in CPP_IMPLEMENTATION_GUIDE.md
 */
class ThreadPool {
public:
    /**
     * Constructor: Creates a thread pool with specified number of threads
     * @param num_threads Number of worker threads (default: hardware concurrency)
     */
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());

    /**
     * Destructor: Waits for all tasks to complete and joins all threads
     */
    ~ThreadPool();

    /**
     * Enqueue a task for execution
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return Future that will hold the result
     */
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

    /**
     * Get the number of threads in the pool
     */
    size_t size() const { return workers_.size(); }

    /**
     * Get the number of pending tasks
     */
    size_t pending_tasks() const;

private:
    // Worker threads
    std::vector<std::thread> workers_;

    // Task queue
    std::queue<std::function<void()>> tasks_;

    // Synchronization
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;

    // Task counter for metrics
    std::atomic<size_t> pending_count_;
};

// Template implementation must be in header
template<typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {

    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        if (stop_.load()) {
            throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
        }

        tasks_.emplace([task]() { (*task)(); });
        pending_count_.fetch_add(1);
    }

    condition_.notify_one();
    return res;
}

} // namespace pheromone
