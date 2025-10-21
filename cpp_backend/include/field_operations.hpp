#pragma once

#include <vector>
#include <unordered_map>
#include <immintrin.h>  // AVX/AVX2 SIMD intrinsics
#include <memory>
#include <mutex>

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

    // Constructor
    PheromoneVector4D();
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

    // Add pheromone to field
    void add_pheromone(int x, int y, const PheromoneVector4D& pheromone);

    // Get pheromones at position
    std::vector<PheromoneVector4D> get_pheromones_at(int x, int y) const;

    // Clear all pheromones
    void clear();

    // Get field size
    size_t size() const;

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
    mutable std::mutex mutex_;

    // SIMD helper functions
    void decay_vector_simd(PheromoneVector4D& vec, float rate);
    PheromoneVector4D aggregate_simd(const std::vector<PheromoneVector4D>& vectors);

    // Position key calculation
    int get_position_key(int x, int y) const;
};

} // namespace pheromone
