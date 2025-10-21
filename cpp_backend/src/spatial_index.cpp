#include "spatial_index.hpp"
#include <thread>
#include <algorithm>
#include <cmath>

namespace pheromone {

// ========== Constructor/Destructor ==========

SpatialIndex::SpatialIndex()
    : rtree_(std::make_unique<bgi::rtree<Value, bgi::quadratic<16>>>()) {
}

SpatialIndex::~SpatialIndex() {
    // unique_ptr handles cleanup automatically
}

// ========== Helper Methods ==========

Value SpatialIndex::to_value(const ResourcePoint& rp) const {
    Point pt(rp.x, rp.y);
    return std::make_pair(pt, rp);
}

ResourcePoint SpatialIndex::from_value(const Value& val) const {
    return val.second;
}

// ========== Single Insert/Remove Operations ==========

void SpatialIndex::insert(float x, float y, int resource_id, float value) {
    std::lock_guard<std::mutex> lock(mutex_);
    ResourcePoint rp(x, y, resource_id, value);
    rtree_->insert(to_value(rp));
}

void SpatialIndex::remove(int resource_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find and collect all values to remove
    std::vector<Value> to_remove;
    for (auto it = rtree_->qbegin(bgi::satisfies([resource_id](const Value& v) {
        return v.second.resource_id == resource_id;
    })); it != rtree_->qend(); ++it) {
        to_remove.push_back(*it);
    }

    // Remove them
    for (const auto& val : to_remove) {
        rtree_->remove(val);
    }
}

void SpatialIndex::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    rtree_->clear();
}

// ========== Query Operations ==========

std::vector<ResourcePoint> SpatialIndex::query_radius(float x, float y, float radius) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<ResourcePoint> results;
    Point query_point(x, y);

    // Use R-tree's spatial query with within() predicate
    // This creates a bounding box and efficiently finds candidates
    for (auto it = rtree_->qbegin(bgi::satisfies([x, y, radius](const Value& v) {
        float dx = v.second.x - x;
        float dy = v.second.y - y;
        float dist_sq = dx * dx + dy * dy;
        return dist_sq <= radius * radius;
    })); it != rtree_->qend(); ++it) {
        results.push_back(from_value(*it));
    }

    return results;
}

std::vector<ResourcePoint> SpatialIndex::query_knn(float x, float y, int k) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<ResourcePoint> results;
    Point query_point(x, y);

    // Use R-tree's built-in KNN query
    std::vector<Value> knn_results;
    rtree_->query(bgi::nearest(query_point, k), std::back_inserter(knn_results));

    // Convert to ResourcePoint
    results.reserve(knn_results.size());
    for (const auto& val : knn_results) {
        results.push_back(from_value(val));
    }

    return results;
}

// ========== Batch Operations ==========

void SpatialIndex::insert_batch(const std::vector<ResourcePoint>& resources) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Batch insert is more efficient than individual inserts
    std::vector<Value> values;
    values.reserve(resources.size());

    for (const auto& rp : resources) {
        values.push_back(to_value(rp));
    }

    // Bulk insert into R-tree
    for (const auto& val : values) {
        rtree_->insert(val);
    }
}

std::vector<std::vector<ResourcePoint>> SpatialIndex::query_radius_batch(
    const std::vector<std::pair<float, float>>& positions,
    float radius,
    int num_threads) {

    // Prepare result vector
    std::vector<std::vector<ResourcePoint>> results(positions.size());

    // Handle edge cases
    if (positions.empty()) {
        return results;
    }

    // Ensure num_threads is reasonable
    num_threads = std::max(1, std::min(num_threads, static_cast<int>(positions.size())));

    // Lambda for processing a range of queries
    auto process_range = [this, &positions, &results, radius](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            results[i] = this->query_radius(positions[i].first, positions[i].second, radius);
        }
    };

    // Multi-threaded batch processing
    std::vector<std::thread> threads;
    size_t chunk_size = (positions.size() + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, positions.size());

        if (start < end) {
            threads.emplace_back(process_range, start, end);
        }
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    return results;
}

std::vector<std::vector<ResourcePoint>> SpatialIndex::query_knn_batch(
    const std::vector<std::pair<float, float>>& positions,
    int k,
    int num_threads) {

    // Prepare result vector
    std::vector<std::vector<ResourcePoint>> results(positions.size());

    // Handle edge cases
    if (positions.empty()) {
        return results;
    }

    // Ensure num_threads is reasonable
    num_threads = std::max(1, std::min(num_threads, static_cast<int>(positions.size())));

    // Lambda for processing a range of queries
    auto process_range = [this, &positions, &results, k](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            results[i] = this->query_knn(positions[i].first, positions[i].second, k);
        }
    };

    // Multi-threaded batch processing
    std::vector<std::thread> threads;
    size_t chunk_size = (positions.size() + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, positions.size());

        if (start < end) {
            threads.emplace_back(process_range, start, end);
        }
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    return results;
}

// ========== Utility Methods ==========

size_t SpatialIndex::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return rtree_->size();
}

bool SpatialIndex::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return rtree_->empty();
}

} // namespace pheromone
