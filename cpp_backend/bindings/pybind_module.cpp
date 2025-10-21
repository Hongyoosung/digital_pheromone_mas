#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "../include/message_codec.hpp"
#include "../include/thread_pool.hpp"
#include "../include/field_operations.hpp"
#include "../include/spatial_index.hpp"

namespace py = pybind11;
using namespace pheromone;

PYBIND11_MODULE(cpp_accelerators, m) {
    m.doc() = "C++ Performance Accelerators for Digital Pheromone MAS\n"
              "Provides high-performance message encoding/decoding with multi-threading\n"
              "Target speedup: 4-10x over pure Python implementation";

    // EnvironmentalContext struct
    py::class_<PheromoneData::EnvironmentalContext>(m, "EnvironmentalContext")
        .def(py::init<>())
        .def_readwrite("position", &PheromoneData::EnvironmentalContext::position)
        .def_readwrite("local_resources", &PheromoneData::EnvironmentalContext::local_resources)
        .def_readwrite("danger_level", &PheromoneData::EnvironmentalContext::danger_level)
        .def_readwrite("exploration_map", &PheromoneData::EnvironmentalContext::exploration_map)
        .def_readwrite("territory_info", &PheromoneData::EnvironmentalContext::territory_info);

    // PheromoneData struct
    py::class_<PheromoneData>(m, "PheromoneData")
        .def(py::init<>())
        .def_readwrite("behavior", &PheromoneData::behavior)
        .def_readwrite("emotion", &PheromoneData::emotion)
        .def_readwrite("social_relations", &PheromoneData::social_relations)
        .def_readwrite("environmental_context", &PheromoneData::environmental_context);

    // AgentStatus struct
    py::class_<PheromoneMessage::AgentStatus>(m, "AgentStatus")
        .def(py::init<>())
        .def_readwrite("health", &PheromoneMessage::AgentStatus::health)
        .def_readwrite("energy", &PheromoneMessage::AgentStatus::energy)
        .def_readwrite("recent_actions", &PheromoneMessage::AgentStatus::recent_actions)
        .def_readwrite("cooperation_history", &PheromoneMessage::AgentStatus::cooperation_history);

    // Metadata struct
    py::class_<PheromoneMessage::Metadata>(m, "Metadata")
        .def(py::init<>())
        .def_readwrite("protocol_version", &PheromoneMessage::Metadata::protocol_version)
        .def_readwrite("compression_method", &PheromoneMessage::Metadata::compression_method)
        .def_readwrite("priority", &PheromoneMessage::Metadata::priority)
        .def_readwrite("expected_response", &PheromoneMessage::Metadata::expected_response)
        .def_readwrite("routing_path", &PheromoneMessage::Metadata::routing_path)
        .def_readwrite("security_token", &PheromoneMessage::Metadata::security_token);

    // PheromoneMessage struct
    py::class_<PheromoneMessage>(m, "PheromoneMessage")
        .def(py::init<>())
        .def_readwrite("type", &PheromoneMessage::type)
        .def_readwrite("sender_id", &PheromoneMessage::sender_id)
        .def_readwrite("timestamp", &PheromoneMessage::timestamp)
        .def_readwrite("pheromone_data", &PheromoneMessage::pheromone_data)
        .def_readwrite("agent_status", &PheromoneMessage::agent_status)
        .def_readwrite("metadata", &PheromoneMessage::metadata);

    // EncodingMetrics struct
    py::class_<MessageCodec::EncodingMetrics>(m, "EncodingMetrics")
        .def(py::init<>())
        .def_readwrite("total_time_ms", &MessageCodec::EncodingMetrics::total_time_ms)
        .def_readwrite("avg_time_per_message_us", &MessageCodec::EncodingMetrics::avg_time_per_message_us)
        .def_readwrite("total_bytes", &MessageCodec::EncodingMetrics::total_bytes)
        .def_readwrite("num_messages", &MessageCodec::EncodingMetrics::num_messages)
        .def_readwrite("num_threads", &MessageCodec::EncodingMetrics::num_threads)
        .def("__repr__", [](const MessageCodec::EncodingMetrics& m) {
            return "EncodingMetrics(total_time=" + std::to_string(m.total_time_ms) + "ms, "
                   "avg_per_msg=" + std::to_string(m.avg_time_per_message_us) + "us, "
                   "messages=" + std::to_string(m.num_messages) + ", "
                   "threads=" + std::to_string(m.num_threads) + ")";
        });

    // MessageCodec class - main interface
    py::class_<MessageCodec>(m, "MessageCodec")
        .def(py::init<size_t>(),
             py::arg("num_threads") = std::thread::hardware_concurrency(),
             "Create a message codec with specified number of threads\n\n"
             "Args:\n"
             "    num_threads: Number of worker threads (default: CPU core count)")
        .def("encode_batch", &MessageCodec::encode_batch,
             py::arg("messages"),
             "Encode a batch of messages to JSON strings in parallel\n\n"
             "Args:\n"
             "    messages: List of PheromoneMessage objects\n\n"
             "Returns:\n"
             "    List of JSON strings\n\n"
             "Performance: 4-10x faster than Python's json.dumps()")
        .def("decode_batch", &MessageCodec::decode_batch,
             py::arg("json_strings"),
             "Decode a batch of JSON strings to messages in parallel\n\n"
             "Args:\n"
             "    json_strings: List of JSON strings\n\n"
             "Returns:\n"
             "    List of PheromoneMessage objects\n\n"
             "Performance: 4-10x faster than Python's json.loads()")
        .def("encode", &MessageCodec::encode,
             py::arg("message"),
             "Encode a single message to JSON string\n\n"
             "Args:\n"
             "    message: PheromoneMessage object\n\n"
             "Returns:\n"
             "    JSON string")
        .def("decode", &MessageCodec::decode,
             py::arg("json_string"),
             "Decode a single JSON string to message\n\n"
             "Args:\n"
             "    json_string: JSON string\n\n"
             "Returns:\n"
             "    PheromoneMessage object")
        .def("get_last_encoding_metrics", &MessageCodec::get_last_encoding_metrics,
             "Get performance metrics from the last encode_batch() call\n\n"
             "Returns:\n"
             "    EncodingMetrics object with timing and throughput data")
        .def("get_last_decoding_metrics", &MessageCodec::get_last_decoding_metrics,
             "Get performance metrics from the last decode_batch() call\n\n"
             "Returns:\n"
             "    EncodingMetrics object with timing and throughput data")
        .def("encode_batch_from_json", &MessageCodec::encode_batch_from_json,
             py::arg("json_strings"),
             "Fast path: encode batch from pre-serialized JSON strings\n\n"
             "This bypasses the Python->C++ struct conversion overhead.\n"
             "Use this when you already have JSON strings from json.dumps().\n\n"
             "Args:\n"
             "    json_strings: List of JSON strings\n\n"
             "Returns:\n"
             "    List of JSON strings (pass-through with optional validation)")
        .def("set_batch_threshold", &MessageCodec::set_batch_threshold,
             py::arg("threshold"),
             "Set the batching threshold for parallel processing\n\n"
             "Below this threshold, messages are processed serially to avoid\n"
             "thread overhead. Above it, parallel processing is used.\n\n"
             "Args:\n"
             "    threshold: Number of messages (default: 100)")
        .def("get_batch_threshold", &MessageCodec::get_batch_threshold,
             "Get the current batching threshold\n\n"
             "Returns:\n"
             "    Current threshold value");

    // Module-level convenience functions
    m.def("get_hardware_concurrency", []() {
        return std::thread::hardware_concurrency();
    }, "Get the number of CPU cores available for parallel processing");

    m.def("version", []() {
        return "1.0.0";
    }, "Get the version of the C++ accelerators module");

    // Add version info
    m.attr("__version__") = "1.0.0";

    // ===== PHASE 2: FIELD OPERATIONS =====

    // PheromoneVector4D struct
    py::class_<PheromoneVector4D>(m, "PheromoneVector4D")
        .def(py::init<>())
        .def_readwrite("timestamp", &PheromoneVector4D::timestamp)
        .def_readwrite("agent_id", &PheromoneVector4D::agent_id)
        .def("magnitude", &PheromoneVector4D::magnitude,
             "Calculate the magnitude of the pheromone vector using SIMD (AVX2)\n\n"
             "Returns:\n"
             "    float: Euclidean magnitude of all vector components")
        .def("decay", &PheromoneVector4D::decay,
             py::arg("rate"),
             "Apply SIMD-optimized decay to all vector components\n\n"
             "Args:\n"
             "    rate: Decay factor (0.0 to 1.0)\n\n"
             "Performance: 4-10x faster than Python loops")
        .def("__add__", &PheromoneVector4D::operator+,
             "SIMD-optimized vector addition")
        .def_property("behavior",
            [](const PheromoneVector4D& p) { return std::vector<float>(p.behavior, p.behavior + 4); },
            [](PheromoneVector4D& p, const std::vector<float>& v) {
                if (v.size() >= 4) std::copy(v.begin(), v.begin() + 4, p.behavior);
            })
        .def_property("emotion",
            [](const PheromoneVector4D& p) { return std::vector<float>(p.emotion, p.emotion + 5); },
            [](PheromoneVector4D& p, const std::vector<float>& v) {
                if (v.size() >= 5) std::copy(v.begin(), v.begin() + 5, p.emotion);
            })
        .def_property("social",
            [](const PheromoneVector4D& p) { return std::vector<float>(p.social, p.social + 10); },
            [](PheromoneVector4D& p, const std::vector<float>& v) {
                if (v.size() >= 10) std::copy(v.begin(), v.begin() + 10, p.social);
            })
        .def_property("context",
            [](const PheromoneVector4D& p) { return std::vector<float>(p.context, p.context + 5); },
            [](PheromoneVector4D& p, const std::vector<float>& v) {
                if (v.size() >= 5) std::copy(v.begin(), v.begin() + 5, p.context);
            })
        .def("__repr__", [](const PheromoneVector4D& p) {
            return "PheromoneVector4D(agent_id=" + std::to_string(p.agent_id) +
                   ", magnitude=" + std::to_string(p.magnitude()) + ")";
        });

    // FieldMetrics struct
    py::class_<PheromoneFieldCPP::FieldMetrics>(m, "FieldMetrics")
        .def(py::init<>())
        .def_readwrite("decay_time_ms", &PheromoneFieldCPP::FieldMetrics::decay_time_ms)
        .def_readwrite("aggregation_time_ms", &PheromoneFieldCPP::FieldMetrics::aggregation_time_ms)
        .def_readwrite("diffusion_time_ms", &PheromoneFieldCPP::FieldMetrics::diffusion_time_ms)
        .def_readwrite("num_positions", &PheromoneFieldCPP::FieldMetrics::num_positions)
        .def_readwrite("num_pheromones", &PheromoneFieldCPP::FieldMetrics::num_pheromones)
        .def("__repr__", [](const PheromoneFieldCPP::FieldMetrics& m) {
            return "FieldMetrics(decay=" + std::to_string(m.decay_time_ms) + "ms, "
                   "aggregation=" + std::to_string(m.aggregation_time_ms) + "ms, "
                   "diffusion=" + std::to_string(m.diffusion_time_ms) + "ms, "
                   "positions=" + std::to_string(m.num_positions) + ", "
                   "pheromones=" + std::to_string(m.num_pheromones) + ")";
        });

    // PheromoneFieldCPP class - SIMD-optimized field operations
    py::class_<PheromoneFieldCPP>(m, "PheromoneFieldCPP")
        .def(py::init<int, int, float>(),
             py::arg("width"),
             py::arg("height"),
             py::arg("decay_rate"),
             "Create a pheromone field with SIMD-optimized operations\n\n"
             "Args:\n"
             "    width: Field width in cells\n"
             "    height: Field height in cells\n"
             "    decay_rate: Decay factor per timestep (0.0 to 1.0)")
        .def("decay_all_parallel", &PheromoneFieldCPP::decay_all_parallel,
             py::arg("min_magnitude"),
             py::arg("max_lifetime_seconds"),
             py::arg("num_threads") = 8,
             "Apply SIMD-optimized decay to all pheromones in parallel\n\n"
             "Args:\n"
             "    min_magnitude: Minimum magnitude threshold for removal\n"
             "    max_lifetime_seconds: Maximum lifetime in seconds\n"
             "    num_threads: Number of parallel threads (default: 8)\n\n"
             "Performance: 2-5x faster than Python with SIMD vectorization")
        .def("aggregate_pheromones_simd", &PheromoneFieldCPP::aggregate_pheromones_simd,
             py::arg("pheromones_by_position"),
             "Aggregate pheromones using SIMD vectorization\n\n"
             "Args:\n"
             "    pheromones_by_position: List of lists of PheromoneVector4D\n\n"
             "Returns:\n"
             "    List of aggregated PheromoneVector4D objects\n\n"
             "Performance: 2.5-5x faster than Python loops")
        .def("diffuse_parallel", &PheromoneFieldCPP::diffuse_parallel,
             py::arg("radius"),
             py::arg("num_threads") = 8,
             "Apply parallel diffusion (CPU fallback)\n\n"
             "Args:\n"
             "    radius: Diffusion radius in cells\n"
             "    num_threads: Number of parallel threads (default: 8)\n\n"
             "Performance: 2-5x faster than Python with multi-threading")
        .def("add_pheromone", &PheromoneFieldCPP::add_pheromone,
             py::arg("x"), py::arg("y"), py::arg("pheromone"),
             "Add a pheromone at the specified position")
        .def("get_pheromones_at", &PheromoneFieldCPP::get_pheromones_at,
             py::arg("x"), py::arg("y"),
             "Get all pheromones at the specified position")
        .def("clear", &PheromoneFieldCPP::clear,
             "Clear all pheromones from the field")
        .def("size", &PheromoneFieldCPP::size,
             "Get the number of occupied positions in the field")
        .def("get_last_metrics", &PheromoneFieldCPP::get_last_metrics,
             "Get performance metrics from the last field operation\n\n"
             "Returns:\n"
             "    FieldMetrics object with timing data");

    // ===== PHASE 3: SPATIAL INDEXING =====

    // ResourcePoint struct
    py::class_<ResourcePoint>(m, "ResourcePoint")
        .def(py::init<>())
        .def(py::init<float, float, int, float>(),
             py::arg("x"), py::arg("y"), py::arg("resource_id"), py::arg("value"),
             "Create a resource point\n\n"
             "Args:\n"
             "    x: X-coordinate\n"
             "    y: Y-coordinate\n"
             "    resource_id: Unique identifier\n"
             "    value: Resource value/intensity")
        .def_readwrite("x", &ResourcePoint::x)
        .def_readwrite("y", &ResourcePoint::y)
        .def_readwrite("resource_id", &ResourcePoint::resource_id)
        .def_readwrite("value", &ResourcePoint::value)
        .def("__repr__", [](const ResourcePoint& rp) {
            return "ResourcePoint(x=" + std::to_string(rp.x) +
                   ", y=" + std::to_string(rp.y) +
                   ", id=" + std::to_string(rp.resource_id) +
                   ", value=" + std::to_string(rp.value) + ")";
        });

    // SpatialIndex class - R-tree spatial indexing
    py::class_<SpatialIndex>(m, "SpatialIndex")
        .def(py::init<>(),
             "Create an R-tree spatial index for efficient spatial queries\n\n"
             "Features:\n"
             "  - O(log n) query time vs O(n) linear search\n"
             "  - Thread-safe operations\n"
             "  - Batch processing with multi-threading\n\n"
             "Performance: 3-10x faster than Python linear search")
        .def("insert", &SpatialIndex::insert,
             py::arg("x"), py::arg("y"), py::arg("resource_id"), py::arg("value"),
             "Insert a resource into the spatial index\n\n"
             "Args:\n"
             "    x: X-coordinate\n"
             "    y: Y-coordinate\n"
             "    resource_id: Unique identifier\n"
             "    value: Resource value/intensity")
        .def("remove", &SpatialIndex::remove,
             py::arg("resource_id"),
             "Remove a resource by ID\n\n"
             "Args:\n"
             "    resource_id: The ID of the resource to remove")
        .def("clear", &SpatialIndex::clear,
             "Clear all resources from the index")
        .def("query_radius", &SpatialIndex::query_radius,
             py::arg("x"), py::arg("y"), py::arg("radius"),
             "Find all resources within a radius\n\n"
             "Args:\n"
             "    x: X-coordinate of query point\n"
             "    y: Y-coordinate of query point\n"
             "    radius: Search radius\n\n"
             "Returns:\n"
             "    List of ResourcePoint objects within the radius\n\n"
             "Performance: 3-10x faster than Python linear search")
        .def("query_knn", &SpatialIndex::query_knn,
             py::arg("x"), py::arg("y"), py::arg("k"),
             "Find k nearest neighbors\n\n"
             "Args:\n"
             "    x: X-coordinate of query point\n"
             "    y: Y-coordinate of query point\n"
             "    k: Number of nearest neighbors\n\n"
             "Returns:\n"
             "    List of k nearest ResourcePoint objects\n\n"
             "Performance: 5-15x faster than Python sorting")
        .def("insert_batch", &SpatialIndex::insert_batch,
             py::arg("resources"),
             "Insert multiple resources at once\n\n"
             "Args:\n"
             "    resources: List of ResourcePoint objects")
        .def("query_radius_batch", &SpatialIndex::query_radius_batch,
             py::arg("positions"), py::arg("radius"), py::arg("num_threads") = 8,
             "Perform radius queries for multiple positions in parallel\n\n"
             "Args:\n"
             "    positions: List of (x, y) tuples\n"
             "    radius: Search radius (same for all queries)\n"
             "    num_threads: Number of threads (default: 8)\n\n"
             "Returns:\n"
             "    List of result lists, one per query position\n\n"
             "Performance: 10x faster for 50 agent batch queries")
        .def("query_knn_batch", &SpatialIndex::query_knn_batch,
             py::arg("positions"), py::arg("k"), py::arg("num_threads") = 8,
             "Perform KNN queries for multiple positions in parallel\n\n"
             "Args:\n"
             "    positions: List of (x, y) tuples\n"
             "    k: Number of nearest neighbors per query\n"
             "    num_threads: Number of threads (default: 8)\n\n"
             "Returns:\n"
             "    List of result lists, one per query position\n\n"
             "Performance: 10x faster for 50 agent batch queries")
        .def("size", &SpatialIndex::size,
             "Get the number of resources in the index\n\n"
             "Returns:\n"
             "    Number of resources")
        .def("empty", &SpatialIndex::empty,
             "Check if the index is empty\n\n"
             "Returns:\n"
             "    True if empty, False otherwise")
        .def("__len__", &SpatialIndex::size,
             "Get the number of resources (Python len() support)")
        .def("__bool__", [](const SpatialIndex& si) { return !si.empty(); },
             "Check if index is non-empty (Python bool() support)");
}
