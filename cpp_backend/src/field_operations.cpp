#include "field_operations.hpp"
#include <thread>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace pheromone {

// Constructor
PheromoneVector4D::PheromoneVector4D() : timestamp(0.0), agent_id(-1) {
    std::fill_n(behavior, 4, 0.0f);
    std::fill_n(emotion, 5, 0.0f);
    std::fill_n(social, 10, 0.0f);
    std::fill_n(context, 5, 0.0f);
}

// SIMD-optimized magnitude using AVX2
float PheromoneVector4D::magnitude() const {
    __m256 sum_vec = _mm256_setzero_ps();

    // Process behavior (4 floats) + emotion (4/5 floats) = 8 floats
    __m256 vec1 = _mm256_loadu_ps(behavior);  // loads behavior[0-3] + emotion[0-3]
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vec1, vec1));

    // Process remaining emotion[4] + social[0-6] = 8 floats
    __m256 vec2 = _mm256_loadu_ps(&emotion[1]);
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vec2, vec2));

    // Process social[2-9] = 8 floats
    __m256 vec3 = _mm256_loadu_ps(&social[2]);
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vec3, vec3));

    // Process context (5 floats) - use SSE for remainder
    __m128 vec4 = _mm_loadu_ps(context);
    __m128 sq4 = _mm_mul_ps(vec4, vec4);

    // Add remaining element
    float last_elem = context[4] * context[4];

    // Horizontal sum of AVX256
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
    sum_128 = _mm_add_ps(sum_128, sq4);

    // Horizontal add
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);

    float total = _mm_cvtss_f32(sum_128) + last_elem;
    return std::sqrt(total);
}

// SIMD-optimized decay
void PheromoneVector4D::decay(float rate) {
    __m256 rate_vec = _mm256_set1_ps(rate);
    __m256 min_threshold = _mm256_set1_ps(1e-6f);

    // Decay behavior and emotion (8 floats)
    __m256 vec1 = _mm256_loadu_ps(behavior);
    vec1 = _mm256_mul_ps(vec1, rate_vec);
    vec1 = _mm256_max_ps(vec1, min_threshold);
    _mm256_storeu_ps(behavior, vec1);

    // Decay social (8 floats starting from social[0])
    __m256 vec2 = _mm256_loadu_ps(social);
    vec2 = _mm256_mul_ps(vec2, rate_vec);
    vec2 = _mm256_max_ps(vec2, min_threshold);
    _mm256_storeu_ps(social, vec2);

    // Decay remaining social (2 floats) and context (5 floats) - use SSE
    __m128 rate_128 = _mm_set1_ps(rate);
    __m128 min_128 = _mm_set1_ps(1e-6f);

    // Process social[8-9]
    for (int i = 8; i < 10; ++i) {
        social[i] *= rate;
        social[i] = std::max(social[i], 1e-6f);
    }

    // Process context
    __m128 vec3 = _mm_loadu_ps(context);
    vec3 = _mm_mul_ps(vec3, rate_128);
    vec3 = _mm_max_ps(vec3, min_128);
    _mm_storeu_ps(context, vec3);

    // Process context[4]
    context[4] *= rate;
    context[4] = std::max(context[4], 1e-6f);
}

// SIMD-optimized addition
PheromoneVector4D PheromoneVector4D::operator+(const PheromoneVector4D& other) const {
    PheromoneVector4D result = *this;

    // Add behavior and emotion (8 floats)
    __m256 res1 = _mm256_loadu_ps(result.behavior);
    __m256 vec1 = _mm256_loadu_ps(other.behavior);
    res1 = _mm256_add_ps(res1, vec1);
    _mm256_storeu_ps(result.behavior, res1);

    // Add social (8 floats)
    __m256 res2 = _mm256_loadu_ps(result.social);
    __m256 vec2 = _mm256_loadu_ps(other.social);
    res2 = _mm256_add_ps(res2, vec2);
    _mm256_storeu_ps(result.social, res2);

    // Add remaining social and context
    for (int i = 8; i < 10; ++i) {
        result.social[i] += other.social[i];
    }

    __m128 res3 = _mm_loadu_ps(result.context);
    __m128 vec3 = _mm_loadu_ps(other.context);
    res3 = _mm_add_ps(res3, vec3);
    _mm_storeu_ps(result.context, res3);

    result.context[4] += other.context[4];

    return result;
}

// PheromoneFieldCPP Constructor
PheromoneFieldCPP::PheromoneFieldCPP(int width, int height, float decay_rate)
    : width_(width), height_(height), decay_rate_(decay_rate) {
    std::memset(&last_metrics_, 0, sizeof(last_metrics_));
}

int PheromoneFieldCPP::get_position_key(int x, int y) const {
    return y * width_ + x;
}

void PheromoneFieldCPP::add_pheromone(int x, int y, const PheromoneVector4D& pheromone) {
    std::lock_guard<std::mutex> lock(mutex_);
    int key = get_position_key(x, y);
    field_[key].push_back(pheromone);
}

std::vector<PheromoneVector4D> PheromoneFieldCPP::get_pheromones_at(int x, int y) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int key = get_position_key(x, y);
    auto it = field_.find(key);
    if (it != field_.end()) {
        return it->second;
    }
    return {};
}

void PheromoneFieldCPP::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    field_.clear();
}

size_t PheromoneFieldCPP::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return field_.size();
}

// Multi-threaded decay
void PheromoneFieldCPP::decay_all_parallel(
    float min_magnitude, double max_lifetime_seconds, int num_threads) {

    auto start = std::chrono::high_resolution_clock::now();

    auto current_time_point = std::chrono::system_clock::now();
    double current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time_point.time_since_epoch()).count() / 1000.0;

    // Partition field into chunks
    std::vector<int> position_keys;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        position_keys.reserve(field_.size());
        for (const auto& [key, _] : field_) {
            position_keys.push_back(key);
        }
    }

    if (position_keys.empty()) {
        last_metrics_.decay_time_ms = 0.0;
        last_metrics_.num_positions = 0;
        last_metrics_.num_pheromones = 0;
        return;
    }

    size_t chunk_size = (position_keys.size() + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    std::vector<std::vector<int>> positions_to_remove(num_threads);
    size_t total_pheromones = 0;

    for (int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, position_keys.size());

        threads.emplace_back([this, &position_keys, &positions_to_remove, t,
                             start_idx, end_idx, min_magnitude, max_lifetime_seconds, current_time]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                int pos_key = position_keys[i];

                // Lock only for accessing this specific position
                std::vector<PheromoneVector4D> pheromones;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto it = field_.find(pos_key);
                    if (it != field_.end()) {
                        pheromones = it->second;
                    }
                }

                if (pheromones.empty()) continue;

                std::vector<PheromoneVector4D> survivors;
                for (auto& p : pheromones) {
                    p.decay(decay_rate_);

                    bool strong_enough = p.magnitude() > min_magnitude * 0.5f;
                    bool not_expired = (current_time - p.timestamp) < max_lifetime_seconds * 2.0;

                    if (strong_enough && not_expired) {
                        survivors.push_back(p);
                    }
                }

                // Update field
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (survivors.empty()) {
                        positions_to_remove[t].push_back(pos_key);
                    } else {
                        field_[pos_key] = std::move(survivors);
                    }
                }
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Remove empty positions
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& remove_list : positions_to_remove) {
            for (int pos_key : remove_list) {
                field_.erase(pos_key);
            }
        }

        // Count total pheromones
        for (const auto& [key, pheromones] : field_) {
            total_pheromones += pheromones.size();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    last_metrics_.decay_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    last_metrics_.num_positions = field_.size();
    last_metrics_.num_pheromones = total_pheromones;
}

// SIMD-optimized aggregation helper
PheromoneVector4D PheromoneFieldCPP::aggregate_simd(
    const std::vector<PheromoneVector4D>& vectors) {

    if (vectors.empty()) {
        return PheromoneVector4D();
    }

    PheromoneVector4D result = vectors[0];

    for (size_t i = 1; i < vectors.size(); ++i) {
        const auto& v = vectors[i];

        // SIMD addition for behavior and emotion (8 floats)
        __m256 res1 = _mm256_loadu_ps(result.behavior);
        __m256 vec1 = _mm256_loadu_ps(v.behavior);
        res1 = _mm256_add_ps(res1, vec1);
        _mm256_storeu_ps(result.behavior, res1);

        // SIMD addition for social (8 floats)
        __m256 res2 = _mm256_loadu_ps(result.social);
        __m256 vec2 = _mm256_loadu_ps(v.social);
        res2 = _mm256_add_ps(res2, vec2);
        _mm256_storeu_ps(result.social, res2);

        // Scalar addition for remaining
        for (int j = 8; j < 10; ++j) {
            result.social[j] += v.social[j];
        }

        __m128 res3 = _mm_loadu_ps(result.context);
        __m128 vec3 = _mm_loadu_ps(v.context);
        res3 = _mm_add_ps(res3, vec3);
        _mm_storeu_ps(result.context, res3);

        result.context[4] += v.context[4];
    }

    return result;
}

// Batch aggregation
std::vector<PheromoneVector4D> PheromoneFieldCPP::aggregate_pheromones_simd(
    const std::vector<std::vector<PheromoneVector4D>>& pheromones_by_position) {

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<PheromoneVector4D> results;
    results.reserve(pheromones_by_position.size());

    for (const auto& pheromones : pheromones_by_position) {
        results.push_back(aggregate_simd(pheromones));
    }

    auto end = std::chrono::high_resolution_clock::now();
    last_metrics_.aggregation_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    return results;
}

// Multi-threaded diffusion (CPU fallback)
void PheromoneFieldCPP::diffuse_parallel(int radius, int num_threads) {
    auto start = std::chrono::high_resolution_clock::now();

    // Get all positions
    std::vector<int> position_keys;
    std::unordered_map<int, std::vector<PheromoneVector4D>> field_copy;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        position_keys.reserve(field_.size());
        for (const auto& [key, pheromones] : field_) {
            position_keys.push_back(key);
            field_copy[key] = pheromones;
        }
    }

    if (position_keys.empty()) {
        last_metrics_.diffusion_time_ms = 0.0;
        return;
    }

    // Create diffusion result map
    std::unordered_map<int, std::vector<PheromoneVector4D>> diffused_field;
    std::mutex diffused_mutex;

    size_t chunk_size = (position_keys.size() + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, position_keys.size());

        threads.emplace_back([this, &position_keys, &field_copy, &diffused_field, &diffused_mutex,
                             start_idx, end_idx, radius]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                int pos_key = position_keys[i];
                int x = pos_key % width_;
                int y = pos_key / width_;

                const auto& pheromones = field_copy[pos_key];

                // Diffuse to neighbors
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (dx == 0 && dy == 0) continue;

                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
                            int neighbor_key = get_position_key(nx, ny);

                            // Calculate diffusion factor based on distance
                            float dist = std::sqrt(dx * dx + dy * dy);
                            float diffusion_factor = 0.1f / (1.0f + dist);

                            // Add diffused pheromones to neighbor
                            for (const auto& p : pheromones) {
                                PheromoneVector4D diffused = p;
                                diffused.decay(1.0f - diffusion_factor);

                                std::lock_guard<std::mutex> lock(diffused_mutex);
                                diffused_field[neighbor_key].push_back(diffused);
                            }
                        }
                    }
                }
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Merge diffused pheromones back to field
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [key, diffused_pheromones] : diffused_field) {
            field_[key].insert(field_[key].end(),
                              diffused_pheromones.begin(),
                              diffused_pheromones.end());
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    last_metrics_.diffusion_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

} // namespace pheromone
