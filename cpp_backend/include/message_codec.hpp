#pragma once

#include "thread_pool.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>

namespace pheromone {

/**
 * 4D Pheromone Data Structure
 * Matches the Python PheromoneVector structure
 */
struct PheromoneData {
    std::vector<float> behavior;           // 4 dimensions
    std::vector<float> emotion;            // 5 dimensions
    std::unordered_map<std::string, float> social_relations;  // 10 dimensions

    struct EnvironmentalContext {
        std::vector<float> position;       // [x, y]
        float local_resources;
        float danger_level;
        std::vector<std::pair<float, float>> exploration_map;
        std::unordered_map<std::string, float> territory_info;
    } environmental_context;
};

/**
 * Complete Message Structure
 * Matches the Python message format for pheromone communication
 */
struct PheromoneMessage {
    std::string type;
    int sender_id;
    double timestamp;
    PheromoneData pheromone_data;

    // Agent status
    struct AgentStatus {
        float health;
        float energy;
        std::vector<std::string> recent_actions;
        std::unordered_map<std::string, float> cooperation_history;
    } agent_status;

    // Metadata
    struct Metadata {
        std::string protocol_version;
        std::string compression_method;
        std::string priority;
        bool expected_response;
        std::vector<int> routing_path;
        std::string security_token;
    } metadata;
};

/**
 * High-Performance Message Codec
 * Provides multi-threaded JSON encoding/decoding for pheromone messages
 * Target: 4-10x speedup over Python's json.dumps()
 */
class MessageCodec {
public:
    /**
     * Constructor
     * @param num_threads Number of threads for parallel encoding (default: hardware concurrency)
     */
    explicit MessageCodec(size_t num_threads = std::thread::hardware_concurrency());

    /**
     * Parallel batch encoding of messages to JSON strings
     * @param messages Vector of PheromoneMessage objects
     * @return Vector of JSON strings
     */
    std::vector<std::string> encode_batch(const std::vector<PheromoneMessage>& messages);

    /**
     * Parallel batch decoding of JSON strings to messages
     * @param json_strings Vector of JSON strings
     * @return Vector of PheromoneMessage objects
     */
    std::vector<PheromoneMessage> decode_batch(const std::vector<std::string>& json_strings);

    /**
     * Single message encoding (for compatibility)
     * @param message PheromoneMessage to encode
     * @return JSON string
     */
    std::string encode(const PheromoneMessage& message);

    /**
     * Single message decoding (for compatibility)
     * @param json_string JSON string to decode
     * @return PheromoneMessage object
     */
    PheromoneMessage decode(const std::string& json_string);

    /**
     * Performance metrics for the last batch operation
     */
    struct EncodingMetrics {
        double total_time_ms;
        double avg_time_per_message_us;
        size_t total_bytes;
        size_t num_messages;
        size_t num_threads;
    };

    /**
     * Get metrics from the last encoding operation
     */
    EncodingMetrics get_last_encoding_metrics() const { return last_metrics_; }

    /**
     * Get metrics from the last decoding operation
     */
    EncodingMetrics get_last_decoding_metrics() const { return last_decode_metrics_; }

    /**
     * Direct encoding from JSON strings (bypasses struct conversion)
     * This is used when Python already has dict objects - we just serialize them
     * @param json_strings Pre-serialized JSON strings from Python
     * @return Same JSON strings (pass-through for compatibility)
     */
    std::vector<std::string> encode_batch_from_json(const std::vector<std::string>& json_strings);

    /**
     * Set the batching threshold for parallel processing
     * Below this threshold, messages are processed serially to avoid thread overhead
     * @param threshold Number of messages (default: 100)
     */
    void set_batch_threshold(size_t threshold) { batch_threshold_ = threshold; }

    /**
     * Get the current batching threshold
     */
    size_t get_batch_threshold() const { return batch_threshold_; }

private:
    std::unique_ptr<ThreadPool> thread_pool_;
    EncodingMetrics last_metrics_;
    EncodingMetrics last_decode_metrics_;
    size_t batch_threshold_;  // Threshold for using parallel processing

    // Fast JSON encoding using nlohmann/json
    std::string encode_fast(const PheromoneMessage& message);

    // Fast JSON decoding using nlohmann/json
    PheromoneMessage decode_fast(const std::string& json_string);
};

} // namespace pheromone
