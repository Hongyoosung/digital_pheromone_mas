# C++ Implementation Guide for Digital Pheromone MAS Performance Optimization

## Executive Summary

This guide provides a comprehensive roadmap for implementing C++ modules to optimize the most critical performance bottlenecks in the Digital Pheromone Multi-Agent System (MAS). Based on the architecture analysis in `ARCHITECTURE_AND_BOTTLENECKS.md`, we target **2-5x overall speedup** through strategic C++ multithreading optimization.

### Key Performance Targets

| Component | Current (Python) | Target (C++) | Speedup | Priority |
|-----------|-----------------|--------------|---------|----------|
| Communication Codec | 200-500 ms/timestep | 50-100 ms | 4-10x | **CRITICAL** |
| Field Operations | 70-200 ms/timestep | 20-50 ms | 2-5x | **HIGH** |
| Spatial Queries | 5-50 ms/timestep | 1-10 ms | 3-10x | **MEDIUM** |

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Phase 1: Communication Codec (CRITICAL)](#phase-1-communication-codec)
3. [Phase 2: Field Operations (HIGH)](#phase-2-field-operations)
4. [Phase 3: Spatial Indexing (MEDIUM)](#phase-3-spatial-indexing)
5. [Integration with Python (Pybind11)](#integration-with-python)
6. [Build System (CMake)](#build-system)
7. [Testing Strategy](#testing-strategy)
8. [Performance Benchmarking](#performance-benchmarking)

---

## Project Structure

```
digital_pheromone_mas/
├── cpp_backend/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── message_codec.hpp
│   │   ├── field_operations.hpp
│   │   ├── spatial_index.hpp
│   │   └── thread_pool.hpp
│   ├── src/
│   │   ├── message_codec.cpp
│   │   ├── field_operations.cpp
│   │   ├── spatial_index.cpp
│   │   └── thread_pool.cpp
│   ├── bindings/
│   │   └── pybind_module.cpp
│   └── tests/
│       ├── test_message_codec.cpp
│       ├── test_field_operations.cpp
│       └── test_spatial_index.cpp
├── src/
│   └── core/
│       └── cpp_accelerators.py  # Python wrapper
└── benchmarks/
    └── cpp_vs_python_benchmark.py
```

---

## Phase 1: Communication Codec (CRITICAL)

### Problem Analysis

**Current Bottleneck:**
- JSON serialization in Python: ~1-5ms per message
- 200+ messages per timestep (50 agents × 4 targets)
- Total overhead: **200-500ms per timestep** (30-50% of total time)

**Root Cause:**
- Python's `json.dumps()` is single-threaded and CPU-bound
- Unicode encoding overhead
- Dictionary iteration overhead

### C++ Solution: Multi-threaded Message Codec

#### Header File: `include/message_codec.hpp`

```cpp
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

namespace pheromone {

// 4D Pheromone Data Structure
struct PheromoneData {
    std::vector<float> behavior;           // 4 dims
    std::vector<float> emotion;            // 5 dims
    std::unordered_map<int, float> social_relations;  // 10 dims
    struct {
        std::vector<float> position;
        float local_resources;
        float danger_level;
        std::vector<std::pair<float, float>> exploration_map;
        std::unordered_map<int, float> territory_info;
    } environmental_context;
};

// Complete Message Structure
struct PheromoneMessage {
    std::string type;
    int sender_id;
    double timestamp;
    PheromoneData pheromone_data;

    // Agent status
    struct {
        float health;
        float energy;
        std::vector<std::string> recent_actions;
        std::unordered_map<int, float> cooperation_history;
    } agent_status;

    // Metadata
    struct {
        std::string protocol_version;
        std::string compression_method;
        std::string priority;
        bool expected_response;
        std::vector<int> routing_path;
        std::string security_token;
    } metadata;
};

// Thread Pool for Parallel Encoding
class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};

// High-Performance Message Codec
class MessageCodec {
public:
    explicit MessageCodec(size_t num_threads = std::thread::hardware_concurrency());

    // Parallel batch encoding
    std::vector<std::string> encode_batch(const std::vector<PheromoneMessage>& messages);

    // Parallel batch decoding
    std::vector<PheromoneMessage> decode_batch(const std::vector<std::string>& json_strings);

    // Single message operations (for compatibility)
    std::string encode(const PheromoneMessage& message);
    PheromoneMessage decode(const std::string& json_string);

    // Performance metrics
    struct EncodingMetrics {
        double total_time_ms;
        double avg_time_per_message_us;
        size_t total_bytes;
        size_t num_messages;
    };

    EncodingMetrics get_last_encoding_metrics() const { return last_metrics_; }

private:
    std::unique_ptr<ThreadPool> thread_pool_;
    EncodingMetrics last_metrics_;

    // Optimized JSON encoding using nlohmann/json or RapidJSON
    std::string encode_fast(const PheromoneMessage& message);
    PheromoneMessage decode_fast(const std::string& json_string);
};

} // namespace pheromone
```

#### Implementation: `src/message_codec.cpp`

```cpp
#include "message_codec.hpp"
#include <nlohmann/json.hpp>  // Fast C++ JSON library
#include <chrono>
#include <sstream>

using json = nlohmann::json;

namespace pheromone {

// Thread Pool Implementation
ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    condition.wait(lock, [this] {
                        return stop.load() || !tasks.empty();
                    });

                    if (stop.load() && tasks.empty()) return;

                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    stop.store(true);
    condition.notify_all();
    for (auto& worker : workers) {
        if (worker.joinable()) worker.join();
    }
}

template<typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop.load()) throw std::runtime_error("ThreadPool is stopped");
        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

// MessageCodec Implementation
MessageCodec::MessageCodec(size_t num_threads)
    : thread_pool_(std::make_unique<ThreadPool>(num_threads)) {}

std::string MessageCodec::encode_fast(const PheromoneMessage& msg) {
    json j;

    // Type and metadata
    j["type"] = msg.type;
    j["sender_id"] = msg.sender_id;
    j["timestamp"] = msg.timestamp;

    // Pheromone data (4D vector)
    j["pheromone_data"]["behavior"] = msg.pheromone_data.behavior;
    j["pheromone_data"]["emotion"] = msg.pheromone_data.emotion;
    j["pheromone_data"]["social_relations"] = msg.pheromone_data.social_relations;

    // Environmental context
    auto& env = msg.pheromone_data.environmental_context;
    j["pheromone_data"]["environmental_context"]["position"] = env.position;
    j["pheromone_data"]["environmental_context"]["local_resources"] = env.local_resources;
    j["pheromone_data"]["environmental_context"]["danger_level"] = env.danger_level;
    j["pheromone_data"]["environmental_context"]["exploration_map"] = env.exploration_map;
    j["pheromone_data"]["environmental_context"]["territory_info"] = env.territory_info;

    // Agent status
    j["agent_status"]["health"] = msg.agent_status.health;
    j["agent_status"]["energy"] = msg.agent_status.energy;
    j["agent_status"]["recent_actions"] = msg.agent_status.recent_actions;
    j["agent_status"]["cooperation_history"] = msg.agent_status.cooperation_history;

    // Metadata
    j["metadata"]["protocol_version"] = msg.metadata.protocol_version;
    j["metadata"]["compression_method"] = msg.metadata.compression_method;
    j["metadata"]["priority"] = msg.metadata.priority;
    j["metadata"]["expected_response"] = msg.metadata.expected_response;
    j["metadata"]["routing_path"] = msg.metadata.routing_path;
    j["metadata"]["security_token"] = msg.metadata.security_token;

    return j.dump();  // Fast serialization
}

std::vector<std::string> MessageCodec::encode_batch(
    const std::vector<PheromoneMessage>& messages) {

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::future<std::string>> futures;
    futures.reserve(messages.size());

    // Submit all encoding tasks to thread pool
    for (const auto& msg : messages) {
        futures.push_back(thread_pool_->enqueue([this, &msg]() {
            return encode_fast(msg);
        }));
    }

    // Collect results
    std::vector<std::string> results;
    results.reserve(messages.size());
    size_t total_bytes = 0;

    for (auto& future : futures) {
        std::string encoded = future.get();
        total_bytes += encoded.size();
        results.push_back(std::move(encoded));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update metrics
    last_metrics_.total_time_ms = duration.count() / 1000.0;
    last_metrics_.avg_time_per_message_us = duration.count() / static_cast<double>(messages.size());
    last_metrics_.total_bytes = total_bytes;
    last_metrics_.num_messages = messages.size();

    return results;
}

PheromoneMessage MessageCodec::decode_fast(const std::string& json_string) {
    json j = json::parse(json_string);

    PheromoneMessage msg;
    msg.type = j["type"];
    msg.sender_id = j["sender_id"];
    msg.timestamp = j["timestamp"];

    // Decode pheromone data
    msg.pheromone_data.behavior = j["pheromone_data"]["behavior"].get<std::vector<float>>();
    msg.pheromone_data.emotion = j["pheromone_data"]["emotion"].get<std::vector<float>>();
    msg.pheromone_data.social_relations = j["pheromone_data"]["social_relations"]
        .get<std::unordered_map<int, float>>();

    // Decode environmental context
    auto& env_json = j["pheromone_data"]["environmental_context"];
    msg.pheromone_data.environmental_context.position = env_json["position"];
    msg.pheromone_data.environmental_context.local_resources = env_json["local_resources"];
    msg.pheromone_data.environmental_context.danger_level = env_json["danger_level"];
    msg.pheromone_data.environmental_context.exploration_map =
        env_json["exploration_map"].get<std::vector<std::pair<float, float>>>();
    msg.pheromone_data.environmental_context.territory_info =
        env_json["territory_info"].get<std::unordered_map<int, float>>();

    // Decode agent status
    msg.agent_status.health = j["agent_status"]["health"];
    msg.agent_status.energy = j["agent_status"]["energy"];
    msg.agent_status.recent_actions = j["agent_status"]["recent_actions"];
    msg.agent_status.cooperation_history =
        j["agent_status"]["cooperation_history"].get<std::unordered_map<int, float>>();

    // Decode metadata
    msg.metadata.protocol_version = j["metadata"]["protocol_version"];
    msg.metadata.compression_method = j["metadata"]["compression_method"];
    msg.metadata.priority = j["metadata"]["priority"];
    msg.metadata.expected_response = j["metadata"]["expected_response"];
    msg.metadata.routing_path = j["metadata"]["routing_path"];
    msg.metadata.security_token = j["metadata"]["security_token"];

    return msg;
}

std::vector<PheromoneMessage> MessageCodec::decode_batch(
    const std::vector<std::string>& json_strings) {

    std::vector<std::future<PheromoneMessage>> futures;
    futures.reserve(json_strings.size());

    for (const auto& json_str : json_strings) {
        futures.push_back(thread_pool_->enqueue([this, &json_str]() {
            return decode_fast(json_str);
        }));
    }

    std::vector<PheromoneMessage> results;
    results.reserve(json_strings.size());

    for (auto& future : futures) {
        results.push_back(future.get());
    }

    return results;
}

} // namespace pheromone
```

### Expected Performance Improvement

| Metric | Python | C++ (Single-threaded) | C++ (8 threads) | Speedup |
|--------|--------|----------------------|-----------------|---------|
| Encoding time (200 msgs) | 400-1000ms | 100-200ms | 50-100ms | **4-10x** |
| Per-message time | 2-5ms | 0.5-1ms | 0.25-0.5ms | **4-10x** |
| CPU usage | 100% (1 core) | 100% (1 core) | 800% (8 cores) | 8x parallelism |

---

## Phase 2: Field Operations (HIGH)

### Problem Analysis

**Current Bottleneck:**
- Pheromone decay: 20-50ms (Python loops)
- Pheromone aggregation: 50ms (list comprehension)
- CPU fallback diffusion: 100-200ms (nested loops)

### C++ Solution: SIMD-Vectorized Field Operations

#### Header File: `include/field_operations.hpp`

```cpp
#pragma once

#include <vector>
#include <unordered_map>
#include <immintrin.h>  // AVX/AVX2 SIMD intrinsics
#include <memory>

namespace pheromone {

// 4D Pheromone Vector (aligned for SIMD)
struct alignas(32) PheromoneVector4D {
    float behavior[4];
    float emotion[5];
    float social[10];
    float context[5];
    double timestamp;
    int agent_id;

    // SIMD-optimized magnitude calculation
    float magnitude() const;

    // SIMD-optimized decay
    void decay(float rate);

    // SIMD-optimized addition
    PheromoneVector4D operator+(const PheromoneVector4D& other) const;
};

// Pheromone Field with SIMD Operations
class PheromoneFieldCPP {
public:
    PheromoneFieldCPP(int width, int height, float decay_rate);

    // Multi-threaded decay with SIMD
    void decay_all_parallel(float min_magnitude, double max_lifetime_seconds, int num_threads = 8);

    // SIMD-vectorized aggregation
    std::vector<PheromoneVector4D> aggregate_pheromones_simd(
        const std::vector<std::vector<PheromoneVector4D>>& pheromones_by_position
    );

    // Multi-threaded diffusion (CPU fallback)
    void diffuse_parallel(int radius, int num_threads = 8);

    // Performance metrics
    struct FieldMetrics {
        double decay_time_ms;
        double aggregation_time_ms;
        double diffusion_time_ms;
        size_t num_positions;
        size_t num_pheromones;
    };

    FieldMetrics get_last_metrics() const { return last_metrics_; }

private:
    int width_, height_;
    float decay_rate_;
    std::unordered_map<int, std::vector<PheromoneVector4D>> field_;  // position_key -> pheromones
    FieldMetrics last_metrics_;

    // SIMD helper functions
    void decay_vector_simd(PheromoneVector4D& vec, float rate);
    PheromoneVector4D aggregate_simd(const std::vector<PheromoneVector4D>& vectors);
};

} // namespace pheromone
```

#### Implementation: `src/field_operations.cpp`

```cpp
#include "field_operations.hpp"
#include <thread>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace pheromone {

// SIMD-optimized magnitude using AVX2
float PheromoneVector4D::magnitude() const {
    __m256 sum_vec = _mm256_setzero_ps();

    // Process behavior (4 floats) + emotion (4/5 floats)
    __m256 vec1 = _mm256_loadu_ps(behavior);  // loads 8 floats
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vec1, vec1));

    // Process remaining emotion + social (10 floats)
    __m256 vec2 = _mm256_loadu_ps(&emotion[1]);
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vec2, vec2));

    __m256 vec3 = _mm256_loadu_ps(&social[2]);
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vec3, vec3));

    // Process context (5 floats)
    __m128 vec4 = _mm_loadu_ps(context);
    __m128 sq4 = _mm_mul_ps(vec4, vec4);

    // Horizontal sum
    __m256 sum_256 = sum_vec;
    __m128 sum_high = _mm256_extractf128_ps(sum_256, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_256);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
    sum_128 = _mm_add_ps(sum_128, sq4);

    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);

    return std::sqrt(_mm_cvtss_f32(sum_128));
}

// SIMD-optimized decay
void PheromoneVector4D::decay(float rate) {
    __m256 rate_vec = _mm256_set1_ps(rate);
    __m256 min_threshold = _mm256_set1_ps(1e-6f);

    // Decay behavior and emotion
    __m256 vec1 = _mm256_loadu_ps(behavior);
    vec1 = _mm256_mul_ps(vec1, rate_vec);
    vec1 = _mm256_max_ps(vec1, min_threshold);
    _mm256_storeu_ps(behavior, vec1);

    // Decay social
    __m256 vec2 = _mm256_loadu_ps(social);
    vec2 = _mm256_mul_ps(vec2, rate_vec);
    vec2 = _mm256_max_ps(vec2, min_threshold);
    _mm256_storeu_ps(social, vec2);

    __m256 vec3 = _mm256_loadu_ps(&social[2]);
    vec3 = _mm256_mul_ps(vec3, rate_vec);
    vec3 = _mm256_max_ps(vec3, min_threshold);
    _mm256_storeu_ps(&social[2], vec3);

    // Decay context
    __m128 vec4 = _mm_loadu_ps(context);
    __m128 rate_128 = _mm_set1_ps(rate);
    __m128 min_128 = _mm_set1_ps(1e-6f);
    vec4 = _mm_mul_ps(vec4, rate_128);
    vec4 = _mm_max_ps(vec4, min_128);
    _mm_storeu_ps(context, vec4);
}

// Multi-threaded decay
void PheromoneFieldCPP::decay_all_parallel(
    float min_magnitude, double max_lifetime_seconds, int num_threads) {

    auto start = std::chrono::high_resolution_clock::now();

    double current_time = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;

    // Partition field into chunks
    std::vector<int> position_keys;
    position_keys.reserve(field_.size());
    for (const auto& [key, _] : field_) {
        position_keys.push_back(key);
    }

    size_t chunk_size = (position_keys.size() + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    std::vector<std::vector<int>> positions_to_remove(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, position_keys.size());

        threads.emplace_back([this, &position_keys, &positions_to_remove, t,
                             start_idx, end_idx, min_magnitude, max_lifetime_seconds, current_time]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                int pos_key = position_keys[i];
                auto& pheromones = field_[pos_key];

                std::vector<PheromoneVector4D> survivors;
                for (auto& p : pheromones) {
                    p.decay(decay_rate_);

                    bool strong_enough = p.magnitude() > min_magnitude * 0.5f;
                    bool not_expired = (current_time - p.timestamp) < max_lifetime_seconds * 2.0;

                    if (strong_enough && not_expired) {
                        survivors.push_back(p);
                    }
                }

                if (survivors.empty()) {
                    positions_to_remove[t].push_back(pos_key);
                } else {
                    pheromones = std::move(survivors);
                }
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Remove empty positions
    for (const auto& remove_list : positions_to_remove) {
        for (int pos_key : remove_list) {
            field_.erase(pos_key);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    last_metrics_.decay_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

// SIMD-optimized aggregation
PheromoneVector4D PheromoneFieldCPP::aggregate_simd(
    const std::vector<PheromoneVector4D>& vectors) {

    if (vectors.empty()) {
        PheromoneVector4D zero{};
        std::fill_n(zero.behavior, 4, 0.0f);
        std::fill_n(zero.emotion, 5, 0.0f);
        std::fill_n(zero.social, 10, 0.0f);
        std::fill_n(zero.context, 5, 0.0f);
        return zero;
    }

    PheromoneVector4D result = vectors[0];

    for (size_t i = 1; i < vectors.size(); ++i) {
        const auto& v = vectors[i];

        // SIMD addition
        __m256 res1 = _mm256_loadu_ps(result.behavior);
        __m256 vec1 = _mm256_loadu_ps(v.behavior);
        res1 = _mm256_add_ps(res1, vec1);
        _mm256_storeu_ps(result.behavior, res1);

        __m256 res2 = _mm256_loadu_ps(result.social);
        __m256 vec2 = _mm256_loadu_ps(v.social);
        res2 = _mm256_add_ps(res2, vec2);
        _mm256_storeu_ps(result.social, res2);

        __m128 res3 = _mm_loadu_ps(result.context);
        __m128 vec3 = _mm_loadu_ps(v.context);
        res3 = _mm_add_ps(res3, vec3);
        _mm_storeu_ps(result.context, res3);
    }

    return result;
}

} // namespace pheromone
```

### Expected Performance Improvement

| Operation | Python | C++ (Single) | C++ (SIMD + 8 threads) | Speedup |
|-----------|--------|--------------|------------------------|---------|
| Decay (1000 positions) | 20-50ms | 10-15ms | 5-10ms | **2-5x** |
| Aggregation (100 ops) | 50ms | 20ms | 5ms | **5-10x** |
| Diffusion (CPU) | 100-200ms | 50-80ms | 20-40ms | **2-5x** |

---

## Phase 3: Spatial Indexing (MEDIUM)

### Problem Analysis

**Current Bottleneck:**
- Linear search through 500+ resources: O(n) per query
- 50 agents × multiple queries per timestep = 5-50ms overhead

### C++ Solution: R-tree Spatial Index

#### Header File: `include/spatial_index.hpp`

```cpp
#pragma once

#include <vector>
#include <memory>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace pheromone {

struct ResourcePoint {
    float x, y;
    int resource_id;
    float value;
};

using Point = bg::model::point<float, 2, bg::cs::cartesian>;
using Value = std::pair<Point, ResourcePoint>;

// Thread-safe R-tree spatial index
class SpatialIndex {
public:
    SpatialIndex();

    // Insert resource
    void insert(float x, float y, int resource_id, float value);

    // Range query (find all resources within radius)
    std::vector<ResourcePoint> query_radius(float x, float y, float radius);

    // K-nearest neighbors query
    std::vector<ResourcePoint> query_knn(float x, float y, int k);

    // Remove resource
    void remove(int resource_id);

    // Batch operations
    void insert_batch(const std::vector<ResourcePoint>& resources);
    std::vector<std::vector<ResourcePoint>> query_radius_batch(
        const std::vector<std::pair<float, float>>& positions, float radius, int num_threads = 8);

    size_t size() const;

private:
    std::unique_ptr<bgi::rtree<Value, bgi::quadratic<16>>> rtree_;
    std::mutex mutex_;
};

} // namespace pheromone
```

### Expected Performance Improvement

| Query Type | Python (Linear) | C++ (R-tree) | Speedup |
|------------|----------------|--------------|---------|
| Single radius query | 1-5ms | 0.1-0.5ms | **3-10x** |
| 50 agent queries | 50-250ms | 5-25ms | **10x** |

---

## Integration with Python (Pybind11)

### Python Bindings: `bindings/pybind_module.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "message_codec.hpp"
#include "field_operations.hpp"
#include "spatial_index.hpp"

namespace py = pybind11;
using namespace pheromone;

PYBIND11_MODULE(cpp_accelerators, m) {
    m.doc() = "C++ Performance Accelerators for Digital Pheromone MAS";

    // MessageCodec binding
    py::class_<MessageCodec>(m, "MessageCodec")
        .def(py::init<size_t>(), py::arg("num_threads") = std::thread::hardware_concurrency())
        .def("encode_batch", &MessageCodec::encode_batch)
        .def("decode_batch", &MessageCodec::decode_batch)
        .def("get_last_encoding_metrics", &MessageCodec::get_last_encoding_metrics);

    // PheromoneFieldCPP binding
    py::class_<PheromoneFieldCPP>(m, "PheromoneFieldCPP")
        .def(py::init<int, int, float>())
        .def("decay_all_parallel", &PheromoneFieldCPP::decay_all_parallel,
             py::arg("min_magnitude"), py::arg("max_lifetime_seconds"), py::arg("num_threads") = 8)
        .def("get_last_metrics", &PheromoneFieldCPP::get_last_metrics);

    // SpatialIndex binding
    py::class_<SpatialIndex>(m, "SpatialIndex")
        .def(py::init<>())
        .def("insert", &SpatialIndex::insert)
        .def("query_radius", &SpatialIndex::query_radius)
        .def("query_knn", &SpatialIndex::query_knn)
        .def("insert_batch", &SpatialIndex::insert_batch)
        .def("query_radius_batch", &SpatialIndex::query_radius_batch,
             py::arg("positions"), py::arg("radius"), py::arg("num_threads") = 8)
        .def("size", &SpatialIndex::size);
}
```

### Python Wrapper: `src/core/cpp_accelerators.py`

```python
"""Python wrapper for C++ accelerators with fallback to pure Python"""

try:
    from cpp_accelerators import MessageCodec, PheromoneFieldCPP, SpatialIndex
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ accelerators not available, using pure Python")

class MessageCodecWrapper:
    """Wrapper with automatic fallback"""
    def __init__(self, num_threads=8):
        if CPP_AVAILABLE:
            self.codec = MessageCodec(num_threads)
            self.backend = 'cpp'
        else:
            self.codec = None
            self.backend = 'python'

    def encode_batch(self, messages):
        if self.backend == 'cpp':
            return self.codec.encode_batch(messages)
        else:
            # Fallback to Python
            import json
            return [json.dumps(msg, default=str) for msg in messages]

    def get_metrics(self):
        if self.backend == 'cpp':
            return self.codec.get_last_encoding_metrics()
        else:
            return {'backend': 'python', 'cpp_available': False}
```

---

## Build System (CMake)

### Root CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(pheromone_cpp_backend VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -pthread")

# Find dependencies
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED COMPONENTS geometry)
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${Boost_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/message_codec.cpp
    src/field_operations.cpp
    src/spatial_index.cpp
    src/thread_pool.cpp
)

# Create Python module
pybind11_add_module(cpp_accelerators bindings/pybind_module.cpp ${SOURCES})
target_link_libraries(cpp_accelerators PRIVATE nlohmann_json::nlohmann_json ${Boost_LIBRARIES})

# Install
install(TARGETS cpp_accelerators LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../src/core)
```

### Build Instructions

```bash
# Install dependencies
pip install pybind11
sudo apt-get install libboost-dev libnlohmann-json3-dev

# Build C++ module
cd cpp_backend
mkdir build && cd build
cmake ..
make -j$(nproc)
make install

# Verify installation
python -c "import cpp_accelerators; print('C++ module loaded successfully!')"
```

---

## Testing Strategy

### Unit Tests: `tests/test_message_codec.cpp`

```cpp
#include <gtest/gtest.h>
#include "message_codec.hpp"

using namespace pheromone;

TEST(MessageCodecTest, BatchEncodingPerformance) {
    MessageCodec codec(8);  // 8 threads

    // Create 200 test messages
    std::vector<PheromoneMessage> messages(200);
    for (auto& msg : messages) {
        msg.type = "test";
        msg.sender_id = 0;
        msg.timestamp = 1234567890.0;
        msg.pheromone_data.behavior = {0.1f, 0.2f, 0.3f, 0.4f};
        msg.pheromone_data.emotion = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    }

    auto encoded = codec.encode_batch(messages);
    auto metrics = codec.get_last_encoding_metrics();

    EXPECT_EQ(encoded.size(), 200);
    EXPECT_LT(metrics.total_time_ms, 100.0);  // Should be < 100ms
    EXPECT_GT(metrics.avg_time_per_message_us, 0.0);
}

TEST(MessageCodecTest, RoundTripConsistency) {
    MessageCodec codec(1);

    PheromoneMessage original;
    original.type = "test";
    original.sender_id = 42;
    original.timestamp = 12345.0;
    original.pheromone_data.behavior = {0.1f, 0.2f, 0.3f, 0.4f};

    std::string encoded = codec.encode(original);
    PheromoneMessage decoded = codec.decode(encoded);

    EXPECT_EQ(decoded.sender_id, original.sender_id);
    EXPECT_FLOAT_EQ(decoded.timestamp, original.timestamp);
    EXPECT_EQ(decoded.pheromone_data.behavior, original.pheromone_data.behavior);
}
```

---

## Performance Benchmarking

### Benchmark Script: `benchmarks/cpp_vs_python_benchmark.py`

```python
"""Comprehensive benchmark comparing Python vs C++ implementations"""

import time
import numpy as np
import json
from typing import List, Dict

try:
    from cpp_accelerators import MessageCodec, PheromoneFieldCPP
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

def benchmark_message_codec(num_messages: int = 200, num_iterations: int = 10):
    """Benchmark message encoding/decoding"""

    # Create sample messages
    messages = []
    for i in range(num_messages):
        messages.append({
            'type': 'test',
            'sender_id': i,
            'timestamp': time.time(),
            'pheromone_data': {
                'behavior': [0.1, 0.2, 0.3, 0.4],
                'emotion': [0.1, 0.2, 0.3, 0.4, 0.5],
                'social_relations': {str(j): 0.5 for j in range(10)},
                'environmental_context': {
                    'position': [10.0, 20.0],
                    'local_resources': 50.0,
                    'danger_level': 0.3
                }
            }
        })

    # Python baseline
    python_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        encoded = [json.dumps(msg, default=str) for msg in messages]
        python_times.append(time.perf_counter() - start)

    python_avg = np.mean(python_times) * 1000  # ms

    # C++ implementation
    if CPP_AVAILABLE:
        codec = MessageCodec(8)
        cpp_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            encoded = codec.encode_batch(messages)
            cpp_times.append(time.perf_counter() - start)

        cpp_avg = np.mean(cpp_times) * 1000  # ms
        speedup = python_avg / cpp_avg

        print(f"\n{'='*60}")
        print(f"Message Codec Benchmark ({num_messages} messages)")
        print(f"{'='*60}")
        print(f"Python (single-threaded): {python_avg:.2f} ms")
        print(f"C++ (8 threads):          {cpp_avg:.2f} ms")
        print(f"Speedup:                  {speedup:.2f}x")
        print(f"Per-message (Python):     {python_avg/num_messages:.3f} ms")
        print(f"Per-message (C++):        {cpp_avg/num_messages:.3f} ms")
    else:
        print(f"Python baseline: {python_avg:.2f} ms")
        print("C++ module not available for comparison")

if __name__ == '__main__':
    benchmark_message_codec(num_messages=200, num_iterations=50)
```

---

## Implementation Timeline

### Week 1-2: Phase 1 (Communication Codec)
- **Days 1-3**: Implement `ThreadPool` and `MessageCodec` core
- **Days 4-5**: Add Pybind11 bindings and Python wrapper
- **Days 6-7**: Unit tests and benchmarking

### Week 3-4: Phase 2 (Field Operations)
- **Days 1-4**: Implement SIMD-vectorized decay and aggregation
- **Days 5-6**: Multi-threaded diffusion
- **Days 7**: Integration and testing

### Week 5: Phase 3 (Spatial Indexing)
- **Days 1-3**: Implement R-tree spatial index
- **Days 4-5**: Batch query optimization
- **Days 6-7**: Integration and testing

### Week 6: Final Integration and Validation
- **Days 1-2**: Full system integration
- **Days 3-4**: Performance regression testing
- **Days 5-7**: Documentation and deployment

---

## Expected Overall Performance Improvement

### Per-Timestep Breakdown (50 agents)

| Component | Python | C++ | Improvement |
|-----------|--------|-----|-------------|
| Communication | 200-500ms | 50-100ms | **4-10x** |
| Field Ops | 70-200ms | 20-50ms | **2-5x** |
| Spatial Queries | 5-50ms | 1-10ms | **3-10x** |
| **Total** | **275-750ms** | **71-160ms** | **3.9-4.7x** |

### Full Simulation (1000 timesteps, 50 agents)

| Metric | Python | C++ | Improvement |
|--------|--------|-----|-------------|
| Total time | 30-60 seconds | 6-12 seconds | **5x** |
| Timestep rate | 16-33 steps/sec | 83-167 steps/sec | **5x** |

---

## Troubleshooting

### Common Issues

**1. Compilation errors with AVX2**
```bash
# Verify CPU supports AVX2
cat /proc/cpuinfo | grep avx2

# If not available, remove -mavx2 from CMakeLists.txt
```

**2. Pybind11 import errors**
```python
# Check Python path
import sys
print(sys.path)

# Manually add module path
sys.path.insert(0, '/path/to/cpp_backend/build')
```

**3. Performance not improving**
- Verify multi-threading: Check CPU usage during encoding (should be 800% for 8 threads)
- Profile with `perf`: `perf stat -d python performance_profiler.py`
- Check memory alignment for SIMD operations

---

## Conclusion

This implementation guide provides a complete roadmap for achieving **2-5x overall speedup** through targeted C++ optimization of the three critical bottlenecks:

1. **Communication serialization** (4-10x speedup)
2. **Pheromone field operations** (2-5x speedup)
3. **Spatial queries** (3-10x speedup)

By following this phased approach with proper testing and benchmarking at each stage, you can systematically validate performance improvements and ensure seamless integration with the existing Python codebase.

For questions or issues, refer to the performance profiling results from `performance_profiler.py` to identify specific bottlenecks in your deployment environment.
