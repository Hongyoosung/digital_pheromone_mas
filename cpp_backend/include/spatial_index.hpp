#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace pheromone {

/**
 * Represents a resource point in 2D space with associated metadata
 */
struct ResourcePoint {
    float x, y;           // Position in 2D space
    int resource_id;      // Unique identifier for the resource
    float value;          // Value/intensity of the resource

    ResourcePoint() : x(0), y(0), resource_id(-1), value(0) {}
    ResourcePoint(float x_, float y_, int id, float val)
        : x(x_), y(y_), resource_id(id), value(val) {}

    // Equality operator needed by Boost R-tree for remove operations
    bool operator==(const ResourcePoint& other) const {
        return resource_id == other.resource_id &&
               x == other.x &&
               y == other.y &&
               value == other.value;
    }

    bool operator!=(const ResourcePoint& other) const {
        return !(*this == other);
    }
};

// Boost.Geometry types for spatial indexing
using Point = bg::model::point<float, 2, bg::cs::cartesian>;
using Value = std::pair<Point, ResourcePoint>;

/**
 * High-performance spatial index using R-tree for efficient spatial queries
 *
 * Features:
 * - O(log n) query time for range and KNN searches (vs O(n) linear search)
 * - Thread-safe operations with mutex protection
 * - Batch operations with multi-threading support
 * - Quadratic node splitting strategy for balanced tree
 */
class SpatialIndex {
public:
    /**
     * Constructor - initializes empty R-tree with quadratic splitting (max 16 entries per node)
     */
    SpatialIndex();

    /**
     * Destructor
     */
    ~SpatialIndex();

    // ========== Single Insert/Remove Operations ==========

    /**
     * Insert a single resource point into the spatial index
     * Thread-safe operation
     *
     * @param x X-coordinate
     * @param y Y-coordinate
     * @param resource_id Unique identifier for the resource
     * @param value Value/intensity of the resource
     */
    void insert(float x, float y, int resource_id, float value);

    /**
     * Remove a resource by its ID
     * Note: This operation requires rebuilding the tree, so use sparingly
     * Thread-safe operation
     *
     * @param resource_id The ID of the resource to remove
     */
    void remove(int resource_id);

    /**
     * Clear all resources from the index
     * Thread-safe operation
     */
    void clear();

    // ========== Query Operations ==========

    /**
     * Find all resources within a given radius from a point
     * Uses R-tree spatial indexing for O(log n) average performance
     *
     * @param x X-coordinate of query point
     * @param y Y-coordinate of query point
     * @param radius Search radius
     * @return Vector of ResourcePoint objects within the radius
     */
    std::vector<ResourcePoint> query_radius(float x, float y, float radius);

    /**
     * Find k nearest neighbors to a given point
     * Uses R-tree KNN algorithm for efficient nearest neighbor search
     *
     * @param x X-coordinate of query point
     * @param y Y-coordinate of query point
     * @param k Number of nearest neighbors to find
     * @return Vector of k nearest ResourcePoint objects (or fewer if index has less than k items)
     */
    std::vector<ResourcePoint> query_knn(float x, float y, int k);

    // ========== Batch Operations (Multi-threaded) ==========

    /**
     * Insert multiple resources at once
     * More efficient than individual inserts for large batches
     * Thread-safe operation
     *
     * @param resources Vector of ResourcePoint objects to insert
     */
    void insert_batch(const std::vector<ResourcePoint>& resources);

    /**
     * Perform radius queries for multiple positions in parallel
     * Uses thread pool for parallel processing
     *
     * @param positions Vector of (x, y) query positions
     * @param radius Search radius (same for all queries)
     * @param num_threads Number of threads to use (default: 8)
     * @return Vector of result vectors, one per query position
     */
    std::vector<std::vector<ResourcePoint>> query_radius_batch(
        const std::vector<std::pair<float, float>>& positions,
        float radius,
        int num_threads = 8);

    /**
     * Perform KNN queries for multiple positions in parallel
     * Uses thread pool for parallel processing
     *
     * @param positions Vector of (x, y) query positions
     * @param k Number of nearest neighbors per query
     * @param num_threads Number of threads to use (default: 8)
     * @return Vector of result vectors, one per query position
     */
    std::vector<std::vector<ResourcePoint>> query_knn_batch(
        const std::vector<std::pair<float, float>>& positions,
        int k,
        int num_threads = 8);

    // ========== Utility Methods ==========

    /**
     * Get the number of resources in the index
     * Thread-safe operation
     *
     * @return Number of resources currently in the index
     */
    size_t size() const;

    /**
     * Check if the index is empty
     * Thread-safe operation
     *
     * @return true if empty, false otherwise
     */
    bool empty() const;

private:
    // R-tree with quadratic splitting (max 16 entries per node)
    // Using unique_ptr for proper RAII and avoiding header pollution
    std::unique_ptr<bgi::rtree<Value, bgi::quadratic<16>>> rtree_;

    // Mutex for thread-safe operations
    mutable std::mutex mutex_;

    // Helper function to convert ResourcePoint to Value (Point + data pair)
    Value to_value(const ResourcePoint& rp) const;

    // Helper function to extract ResourcePoint from Value
    ResourcePoint from_value(const Value& val) const;
};

} // namespace pheromone
