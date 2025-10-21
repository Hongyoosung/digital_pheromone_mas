#include "message_codec.hpp"

// Use nlohmann/json for fast JSON parsing
#ifdef __has_include
#  if __has_include(<nlohmann/json.hpp>)
#    include <nlohmann/json.hpp>
#  else
#    include "../include/nlohmann/json.hpp"
#  endif
#endif

#include <sstream>
#include <chrono>

using json = nlohmann::json;

namespace pheromone {

MessageCodec::MessageCodec(size_t num_threads)
    : thread_pool_(std::make_unique<ThreadPool>(num_threads)),
      batch_threshold_(100) {  // Default: use threading for 100+ messages
    last_metrics_ = {0, 0, 0, 0, num_threads};
    last_decode_metrics_ = {0, 0, 0, 0, num_threads};
}

std::string MessageCodec::encode_fast(const PheromoneMessage& msg) {
    json j;

    // Basic message fields
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

    return j.dump();
}

std::string MessageCodec::encode(const PheromoneMessage& message) {
    return encode_fast(message);
}

std::vector<std::string> MessageCodec::encode_batch(
    const std::vector<PheromoneMessage>& messages) {

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::string> results;
    results.reserve(messages.size());
    size_t total_bytes = 0;

    // For small batches, use serial processing to avoid thread overhead
    if (messages.size() < batch_threshold_) {
        for (const auto& msg : messages) {
            std::string encoded = encode_fast(msg);
            total_bytes += encoded.size();
            results.push_back(std::move(encoded));
        }
    } else {
        // For large batches, use parallel processing
        std::vector<std::future<std::string>> futures;
        futures.reserve(messages.size());

        // Submit all encoding tasks to thread pool
        for (const auto& msg : messages) {
            futures.push_back(thread_pool_->enqueue([this, msg]() {
                return encode_fast(msg);
            }));
        }

        // Collect results
        for (auto& future : futures) {
            std::string encoded = future.get();
            total_bytes += encoded.size();
            results.push_back(std::move(encoded));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update metrics
    last_metrics_.total_time_ms = duration.count() / 1000.0;
    last_metrics_.avg_time_per_message_us =
        messages.empty() ? 0.0 : duration.count() / static_cast<double>(messages.size());
    last_metrics_.total_bytes = total_bytes;
    last_metrics_.num_messages = messages.size();
    last_metrics_.num_threads = thread_pool_->size();

    return results;
}

PheromoneMessage MessageCodec::decode_fast(const std::string& json_string) {
    json j = json::parse(json_string);

    PheromoneMessage msg;

    // Basic fields
    msg.type = j.value("type", "");
    msg.sender_id = j.value("sender_id", 0);
    msg.timestamp = j.value("timestamp", 0.0);

    // Pheromone data
    if (j.contains("pheromone_data")) {
        auto& pd = j["pheromone_data"];

        if (pd.contains("behavior") && pd["behavior"].is_array()) {
            msg.pheromone_data.behavior = pd["behavior"].get<std::vector<float>>();
        }

        if (pd.contains("emotion") && pd["emotion"].is_array()) {
            msg.pheromone_data.emotion = pd["emotion"].get<std::vector<float>>();
        }

        if (pd.contains("social_relations") && pd["social_relations"].is_object()) {
            msg.pheromone_data.social_relations =
                pd["social_relations"].get<std::unordered_map<std::string, float>>();
        }

        // Environmental context
        if (pd.contains("environmental_context")) {
            auto& env_json = pd["environmental_context"];

            if (env_json.contains("position") && env_json["position"].is_array()) {
                msg.pheromone_data.environmental_context.position =
                    env_json["position"].get<std::vector<float>>();
            }

            msg.pheromone_data.environmental_context.local_resources =
                env_json.value("local_resources", 0.0f);

            msg.pheromone_data.environmental_context.danger_level =
                env_json.value("danger_level", 0.0f);

            if (env_json.contains("exploration_map") && env_json["exploration_map"].is_array()) {
                msg.pheromone_data.environmental_context.exploration_map =
                    env_json["exploration_map"].get<std::vector<std::pair<float, float>>>();
            }

            if (env_json.contains("territory_info") && env_json["territory_info"].is_object()) {
                msg.pheromone_data.environmental_context.territory_info =
                    env_json["territory_info"].get<std::unordered_map<std::string, float>>();
            }
        }
    }

    // Agent status
    if (j.contains("agent_status")) {
        auto& as = j["agent_status"];

        msg.agent_status.health = as.value("health", 0.0f);
        msg.agent_status.energy = as.value("energy", 0.0f);

        if (as.contains("recent_actions") && as["recent_actions"].is_array()) {
            msg.agent_status.recent_actions =
                as["recent_actions"].get<std::vector<std::string>>();
        }

        if (as.contains("cooperation_history") && as["cooperation_history"].is_object()) {
            msg.agent_status.cooperation_history =
                as["cooperation_history"].get<std::unordered_map<std::string, float>>();
        }
    }

    // Metadata
    if (j.contains("metadata")) {
        auto& md = j["metadata"];

        msg.metadata.protocol_version = md.value("protocol_version", "");
        msg.metadata.compression_method = md.value("compression_method", "");
        msg.metadata.priority = md.value("priority", "");
        msg.metadata.expected_response = md.value("expected_response", false);

        if (md.contains("routing_path") && md["routing_path"].is_array()) {
            msg.metadata.routing_path = md["routing_path"].get<std::vector<int>>();
        }

        msg.metadata.security_token = md.value("security_token", "");
    }

    return msg;
}

PheromoneMessage MessageCodec::decode(const std::string& json_string) {
    return decode_fast(json_string);
}

std::vector<PheromoneMessage> MessageCodec::decode_batch(
    const std::vector<std::string>& json_strings) {

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<PheromoneMessage> results;
    results.reserve(json_strings.size());

    // For small batches, use serial processing to avoid thread overhead
    if (json_strings.size() < batch_threshold_) {
        for (const auto& json_str : json_strings) {
            results.push_back(decode_fast(json_str));
        }
    } else {
        // For large batches, use parallel processing
        std::vector<std::future<PheromoneMessage>> futures;
        futures.reserve(json_strings.size());

        // Submit all decoding tasks to thread pool
        for (const auto& json_str : json_strings) {
            futures.push_back(thread_pool_->enqueue([this, json_str]() {
                return decode_fast(json_str);
            }));
        }

        // Collect results
        for (auto& future : futures) {
            results.push_back(future.get());
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update metrics
    last_decode_metrics_.total_time_ms = duration.count() / 1000.0;
    last_decode_metrics_.avg_time_per_message_us =
        json_strings.empty() ? 0.0 : duration.count() / static_cast<double>(json_strings.size());
    last_decode_metrics_.total_bytes = 0;  // Not tracked for decoding
    last_decode_metrics_.num_messages = json_strings.size();
    last_decode_metrics_.num_threads = thread_pool_->size();

    return results;
}

std::vector<std::string> MessageCodec::encode_batch_from_json(
    const std::vector<std::string>& json_strings) {
    // This is a fast path for when Python has already created JSON strings
    // We just pass them through after validation (optional)
    // This avoids the Python->C++ struct conversion overhead

    auto start = std::chrono::high_resolution_clock::now();

    // For now, just pass through - in future we could add:
    // - JSON validation
    // - Compression
    // - Format normalization
    std::vector<std::string> results = json_strings;

    size_t total_bytes = 0;
    for (const auto& s : results) {
        total_bytes += s.size();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update metrics
    last_metrics_.total_time_ms = duration.count() / 1000.0;
    last_metrics_.avg_time_per_message_us =
        json_strings.empty() ? 0.0 : duration.count() / static_cast<double>(json_strings.size());
    last_metrics_.total_bytes = total_bytes;
    last_metrics_.num_messages = json_strings.size();
    last_metrics_.num_threads = 0;  // No threading used

    return results;
}

} // namespace pheromone
