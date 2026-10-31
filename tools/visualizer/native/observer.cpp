#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "common.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;
static volatile std::sig_atomic_t cancelled = 0;
static void cancel_signal(int) { cancelled = 1; }
static bool abort_eval(void *) { return cancelled != 0; }

struct observer {
    std::string model_id, run_id, phase = "inspection", capture = "routing";
    int step = -1;
    int token_offset = 0;
    int graph_epoch = 0;
    std::unordered_set<const ggml_tensor *> asked;
    uint64_t sequence = 0;
    int64_t step_start = 0;
    bool enabled = true;
    bool failed = false;
    std::unordered_map<const ggml_tensor *, std::string> ids;
    std::unordered_map<const ggml_tensor *, int64_t> starts;
    std::map<int, std::vector<int32_t>> expert_ids;
    json nodes = json::array();
    std::string component = "decoder";

    void emit(const std::string & kind, json data = json::object()) {
        std::cout << json{{"schema_version", 1}, {"sequence", ++sequence}, {"model_id", model_id},
            {"run_id", run_id}, {"kind", kind}, {"time_us", ggml_time_us()}, {"data", std::move(data)}}.dump(-1, ' ', false, json::error_handler_t::replace) << '\n' << std::flush;
    }

    static int layer_of(const std::string & name) {
        std::smatch m;
        static const std::regex weight("(?:blk|layers|block)\\.(\\d+)\\.");
        static const std::regex suffix("-(\\d+)(?:$|[^0-9])");
        if (std::regex_search(name, m, weight) || std::regex_search(name, m, suffix)) {
            return std::stoi(m[1].str());
        }
        return -1;
    }

    std::string discover(const ggml_tensor * t, int depth = 0) {
        if (!t) { return ""; }
        const auto found = ids.find(t);
        if (found != ids.end()) { return found->second; }
        if (depth > 1024 || ids.size() >= 100000) { throw std::runtime_error("runtime graph exceeds observation limits"); }
        const std::string id = "e" + std::to_string(graph_epoch) + "n" + std::to_string(ids.size());
        ids[t] = id;
        json sources = json::array();
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (t->src[i]) { sources.push_back(discover(t->src[i], depth + 1)); }
        }
        std::string device = "unassigned", buffer = "none";
        const ggml_tensor * owner = t;
        for (int i = 0; !owner->buffer && owner->view_src && i < 32; ++i) { owner = owner->view_src; }
        if (owner->buffer) {
            auto buft = ggml_backend_buffer_get_type(owner->buffer);
            buffer = ggml_backend_buft_name(buft);
            auto dev = ggml_backend_buft_get_device(buft);
            if (dev) { device = ggml_backend_dev_name(dev); }
        }
        json node{{"id", id}, {"name", t->name}, {"op", ggml_op_name(t->op)},
            {"dtype", ggml_type_name(t->type)}, {"shape", std::vector<int64_t>(t->ne, t->ne + GGML_MAX_DIMS)},
            {"strides", std::vector<size_t>(t->nb, t->nb + GGML_MAX_DIMS)}, {"bytes", ggml_nbytes(t)},
            {"elements", ggml_nelements(t)}, {"sources", sources}, {"layer", layer_of(t->name)},
            {"component", component}, {"device", device}, {"buffer", buffer}, {"flags", t->flags},
            {"kind", t->op == GGML_OP_NONE ? (owner->buffer && ggml_backend_buffer_get_usage(owner->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS ? "weight" : "input") : "operation"}};
        nodes.push_back(node);
        emit("node", {{"step", step}, {"phase", phase}, {"node", std::move(node)}});
        return id;
    }

    void begin(const std::string & next_phase, int offset, int count) {
        phase = next_phase;
        token_offset = offset;
        ++step;
        ids.clear(); starts.clear(); asked.clear(); expert_ids.clear(); graph_epoch = 0; nodes = json::array();
        step_start = ggml_time_us();
        emit("step_begin", {{"step", step}, {"phase", phase}, {"token_offset", offset}, {"input_tokens", count}});
    }

    void end(int status) {
        emit("step_end", {{"step", step}, {"phase", phase}, {"duration_us", ggml_time_us() - step_start},
            {"node_count", nodes.size()}, {"status", status}, {"complete", status == 0 && !failed}});
    }

    json stats(const ggml_tensor * t) {
        const bool supported = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_I32;
        if (!supported || !t->buffer || ggml_nelements(t) <= 0) { return nullptr; }
        const size_t element_size = ggml_type_size(t->type);
        const size_t count = std::min<int64_t>(256, ggml_is_contiguous(t) ? ggml_nelements(t) : t->ne[0]);
        const size_t span = count ? (count - 1) * t->nb[0] + element_size : 0;
        if (!span || span > 16384 || span > ggml_nbytes(t)) { return nullptr; }
        std::vector<uint8_t> data(span);
        ggml_backend_tensor_get(t, data.data(), 0, span);
        double sum = 0, squares = 0, lo = INFINITY, hi = -INFINITY;
        int finite = 0, nonfinite = 0;
        json values = json::array();
        for (size_t i = 0; i < count; ++i) {
            float value = 0;
            const uint8_t * p = data.data() + i * t->nb[0];
            if (t->type == GGML_TYPE_F32) { std::memcpy(&value, p, 4); }
            else if (t->type == GGML_TYPE_I32) { int32_t v; std::memcpy(&v, p, 4); value = float(v); }
            else { uint16_t v; std::memcpy(&v, p, 2); value = t->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(v) : ggml_bf16_to_fp32({v}); }
            if (std::isfinite(value)) {
                ++finite; sum += value; squares += double(value) * value; lo = std::min(lo, double(value)); hi = std::max(hi, double(value));
                if (i < 16) { values.push_back(value); }
            } else { ++nonfinite; if (i < 16) { values.push_back(nullptr); } }
        }
        return {{"sampled_elements", count}, {"total_elements", ggml_nelements(t)}, {"complete", count == size_t(ggml_nelements(t))},
            {"nonfinite", nonfinite}, {"min", finite ? json(lo) : json(nullptr)}, {"max", finite ? json(hi) : json(nullptr)},
            {"mean", finite ? json(sum / finite) : json(nullptr)}, {"rms", finite ? json(std::sqrt(squares / finite)) : json(nullptr)}, {"sample", values}};
    }

    json neuron_values(const ggml_tensor * t) {
        const std::string name = t->name;
        const bool hidden = name.find("ffn") != std::string::npos &&
            (name.find("swiglu") != std::string::npos || name.find("gelu") != std::string::npos || name.find("ffn_act-") == 0);
        const bool boundary = name.find("l_out-") == 0;
        if ((!hidden && !boundary) || name.find("(") != std::string::npos || !t->buffer) { return nullptr; }
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) { return nullptr; }
        const bool moe = hidden && name.find("moe") != std::string::npos;
        const int64_t channels = t->ne[0], slots = moe ? t->ne[1] : 1;
        if (channels <= 0 || channels > 65536 || slots <= 0 || slots > 256 || channels > 262144 / slots) { return nullptr; }
        const size_t element_size = ggml_type_size(t->type);
        if (t->nb[0] > 4 * 1024 * 1024 / size_t(channels) || t->nb[1] > 4 * 1024 * 1024 / size_t(slots)) { return nullptr; }
        const size_t span = size_t(slots - 1) * t->nb[1] + size_t(channels - 1) * t->nb[0] + element_size;
        if (span > 4 * 1024 * 1024 || span > ggml_nbytes(t)) { return nullptr; }
        std::vector<uint8_t> bytes(span);
        ggml_backend_tensor_get(t, bytes.data(), 0, span);
        json values = json::array();
        for (int64_t slot = 0; slot < slots; ++slot) {
            for (int64_t channel = 0; channel < channels; ++channel) {
                const auto * p = bytes.data() + size_t(slot) * t->nb[1] + size_t(channel) * t->nb[0];
                float value;
                if (t->type == GGML_TYPE_F32) { std::memcpy(&value, p, sizeof(value)); }
                else { uint16_t bits; std::memcpy(&bits, p, sizeof(bits)); value = t->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(bits) : ggml_bf16_to_fp32({bits}); }
                values.push_back(std::isfinite(value) ? json(value) : json(nullptr));
            }
        }
        json selected = json::array();
        if (moe) {
            auto found = expert_ids.find(layer_of(name));
            if (found != expert_ids.end()) { selected = found->second; }
        }
        return {{"channels", channels}, {"slots", slots}, {"token_index", token_offset}, {"expert_ids", selected},
            {"values", std::move(values)}, {"scope", "all channels at the first token of this execution step"},
            {"meaning", hidden ? "FFN hidden activation after nonlinearity" : "layer output feature channels"}};
    }

    void route(const ggml_tensor * op) {
        const ggml_tensor * t = op->src[2];
        if (!t || t->type != GGML_TYPE_I32 || !t->buffer || t->ne[0] <= 0 || t->ne[1] <= 0) { return; }
        const size_t span = ggml_nbytes(t);
        if (span > 16 * 1024 * 1024 || t->ne[0] > 4096 || t->ne[1] > 65536) {
            failed = true; emit("observation_error", {{"message", "expert routing tensor exceeds capture limit"}}); return;
        }
        std::vector<uint8_t> bytes(span);
        ggml_backend_tensor_get(t, bytes.data(), 0, span);
        json selections = json::array();
        for (int64_t col = 0; col < t->ne[1]; ++col) {
            json experts = json::array();
            for (int64_t row = 0; row < t->ne[0]; ++row) {
                const size_t offset = size_t(col) * t->nb[1] + size_t(row) * t->nb[0];
                if (offset > span || span - offset < sizeof(int32_t)) { throw std::runtime_error("invalid routing strides"); }
                int32_t id; std::memcpy(&id, bytes.data() + offset, sizeof(id)); experts.push_back(id);
            }
            if (col == 0) { expert_ids[layer_of(op->name)] = experts.get<std::vector<int32_t>>(); }
            selections.push_back({{"token_index", token_offset + col}, {"experts", experts}});
        }
        emit("routing", {{"step", step}, {"phase", phase}, {"node_id", ids.at(op)}, {"layer", layer_of(op->name)},
            {"component", component}, {"source", "MUL_MAT_ID.src[2]"}, {"selections", selections}});
    }

    static bool eval_encoder(ggml_tensor * t, bool ask, void * userdata) {
        auto & self = *static_cast<observer *>(userdata);
        const std::string previous = self.component;
        self.component = "multimodal_encoder";
        const bool result = eval(t, ask, userdata);
        self.component = previous;
        return result;
    }

    static bool eval(ggml_tensor * t, bool ask, void * userdata) {
        auto & self = *static_cast<observer *>(userdata);
        if (!self.enabled) { return false; }
        try {
            if (ask) {
                if (self.asked.count(t)) {
                    ++self.graph_epoch; self.ids.clear(); self.starts.clear(); self.asked.clear();
                }
                self.asked.insert(t);
                self.discover(t);
                const bool observe = self.capture == "full" || t->op == GGML_OP_MUL_MAT_ID ||
                    std::string(t->name).find("l_out") == 0 || std::string(t->name).find("result_") == 0;
                if (observe) { self.starts[t] = ggml_time_us(); }
                return observe;
            }
            const auto it = self.starts.find(t);
            json data{{"step", self.step}, {"phase", self.phase}, {"node_id", self.ids.at(t)}, {"measured", true},
                {"wall_us", it == self.starts.end() ? 0 : ggml_time_us() - it->second}};
            if (self.capture != "routing") { data["stats"] = self.stats(t); }
            if (self.capture == "full") {
                auto neurons = self.neuron_values(t);
                if (!neurons.is_null()) { data["neurons"] = std::move(neurons); }
            }
            self.emit("activation", std::move(data));
            if (t->op == GGML_OP_MUL_MAT_ID) { self.route(t); }
        } catch (const std::exception & e) {
            self.failed = true; self.emit("observation_error", {{"message", e.what()}});
        }
        return true;
    }
};

static json file_details(const std::string & path) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_from_file(path.c_str(), {true, nullptr}), gguf_free);
    if (!file) { throw std::runtime_error("cannot inspect GGUF metadata"); }
    json metadata = json::object(), tensors = json::array();
    for (int64_t i = 0; i < gguf_get_n_kv(file.get()); ++i) {
        const auto type = gguf_get_kv_type(file.get(), i);
        json value;
        if (type == GGUF_TYPE_ARRAY) {
            value = {{"array_type", gguf_type_name(gguf_get_arr_type(file.get(), i))}, {"count", gguf_get_arr_n(file.get(), i)}, {"paged", true}};
        } else if (type == GGUF_TYPE_STRING) {
            std::string s = gguf_get_val_str(file.get(), i);
            value = s.size() <= 32768 ? json(s) : json{{"preview", s.substr(0, 32768)}, {"characters", s.size()}, {"paged", true}};
        } else {
            switch (type) {
                case GGUF_TYPE_UINT8: value = gguf_get_val_u8(file.get(), i); break;
                case GGUF_TYPE_INT8: value = gguf_get_val_i8(file.get(), i); break;
                case GGUF_TYPE_UINT16: value = gguf_get_val_u16(file.get(), i); break;
                case GGUF_TYPE_INT16: value = gguf_get_val_i16(file.get(), i); break;
                case GGUF_TYPE_UINT32: value = gguf_get_val_u32(file.get(), i); break;
                case GGUF_TYPE_INT32: value = gguf_get_val_i32(file.get(), i); break;
                case GGUF_TYPE_UINT64: value = gguf_get_val_u64(file.get(), i); break;
                case GGUF_TYPE_INT64: value = gguf_get_val_i64(file.get(), i); break;
                case GGUF_TYPE_FLOAT32: value = gguf_get_val_f32(file.get(), i); break;
                case GGUF_TYPE_FLOAT64: value = gguf_get_val_f64(file.get(), i); break;
                case GGUF_TYPE_BOOL: value = gguf_get_val_bool(file.get(), i); break;
                default: value = nullptr;
            }
        }
        metadata[gguf_get_key(file.get(), i)] = value;
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(file.get()); ++i) {
        const auto * ne = gguf_get_tensor_ne(file.get(), i);
        tensors.push_back({{"name", gguf_get_tensor_name(file.get(), i)}, {"dtype", ggml_type_name(gguf_get_tensor_type(file.get(), i))},
            {"shape", std::vector<int64_t>(ne, ne + GGML_MAX_DIMS)}, {"bytes", gguf_get_tensor_size(file.get(), i)}, {"offset", gguf_get_tensor_offset(file.get(), i)}});
    }
    return {{"metadata", metadata}, {"tensors", tensors}};
}

struct runner {
    observer trace;
    llama_model_ptr model;
    llama_context_ptr ctx;
    mtmd::context_ptr mm;
    common_chat_templates_ptr templates;
    const llama_vocab * vocab = nullptr;
    int context_size = 2048;
    int batch_size = 256;


    int process(llama_batch_ext * batch, llama_process_type type, const std::string & phase, int offset, int count) {
        trace.component = type == LLAMA_PROCESS_TYPE_ENCODE ? "encoder" : "decoder";
        trace.begin(phase, offset, count);
        int result = llama_process(ctx.get(), type, batch);
        llama_synchronize(ctx.get());
        trace.end(result);
        if (trace.failed) { throw std::runtime_error("observation failed; trace is incomplete"); }
        return result;
    }

    void fill(llama_batch_ext * batch, const std::vector<llama_token> & tokens, size_t start, size_t count, int position) {
        llama_batch_ext_clear(batch);
        for (size_t i = 0; i < count; ++i) {
            int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[start + i]);
            llama_pos pos = position + int(i);
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, int32_t(count) - 1, true);
    }

    void load(const json & command) {
        cancelled = 0;
        templates.reset(); mm.reset(); ctx.reset(); model.reset();
        trace.model_id = command.at("model_id"); trace.run_id = command.value("inspection_id", "inspection:" + trace.model_id);
        trace.step = -1; trace.failed = false; trace.capture = command.value("capture", "routing"); trace.enabled = true;
        const std::string path = command.at("path");
        trace.emit("load_started", {{"path", path}});
        auto params = llama_model_default_params(); params.n_gpu_layers = command.value("gpu_layers", 99);
        model.reset(llama_model_load_from_file(path.c_str(), params));
        if (!model) { throw std::runtime_error("model loading failed; see engine log"); }
        vocab = llama_model_get_vocab(model.get());
        context_size = command.value("context_size", 2048);
        auto cp = llama_context_default_params(); cp.n_ctx = context_size; cp.n_batch = batch_size; cp.n_ubatch = batch_size;
        cp.n_threads = command.value("threads", 4); cp.n_threads_batch = cp.n_threads;
        cp.cb_eval = command.value("observe", true) ? observer::eval : nullptr; cp.cb_eval_user_data = &trace; cp.abort_callback = abort_eval;
        cp.embeddings = !llama_model_has_decoder(model.get()); cp.no_perf = false;
        ctx.reset(llama_init_from_model(model.get(), cp));
        if (!ctx) { throw std::runtime_error("context creation failed"); }
        const std::string mmproj = command.value("mmproj", "");
        if (!mmproj.empty()) {
            auto mp = mtmd_context_params_default(); mp.cb_eval = observer::eval_encoder; mp.cb_eval_user_data = &trace; mp.warmup = false; mp.n_threads = cp.n_threads;
            mm.reset(mtmd_init_from_file(mmproj.c_str(), model.get(), mp));
            if (!mm) { throw std::runtime_error("multimodal projector loading failed"); }
        }
        if (llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_NONE && llama_model_chat_template(model.get(), nullptr)) {
            templates = common_chat_templates_init(model.get(), "");
        }
        json details = file_details(path);
        details["path"] = path; details["parameters"] = llama_model_n_params(model.get()); details["weight_bytes"] = llama_model_size(model.get());
        details["layers"] = llama_model_n_layer(model.get()); details["embedding_size"] = llama_model_n_embd(model.get());
        details["vocab_size"] = llama_vocab_n_tokens(vocab); details["context_train"] = llama_model_n_ctx_train(model.get());
        details["context_size"] = llama_n_ctx(ctx.get()); details["has_encoder"] = llama_model_has_encoder(model.get());
        details["has_decoder"] = llama_model_has_decoder(model.get()); details["recurrent"] = llama_model_is_recurrent(model.get());
        details["hybrid"] = llama_model_is_hybrid(model.get()); details["multimodal"] = bool(mm); details["runtime_version"] = llama_version();
        if (mm) { details["projector"] = file_details(mmproj); }
        trace.emit("model_loaded", std::move(details));
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        llama_token token = llama_vocab_bos(vocab); if (token == LLAMA_TOKEN_NULL) { token = 0; }
        fill(batch.get(), {token}, 0, 1, 0);
        int status = process(batch.get(), llama_model_has_encoder(model.get()) ? LLAMA_PROCESS_TYPE_ENCODE : LLAMA_PROCESS_TYPE_DECODE, "inspection", 0, 1);
        if (status) { throw std::runtime_error("runtime inspection failed: " + std::to_string(status)); }
        if (llama_get_memory(ctx.get())) { llama_memory_clear(llama_get_memory(ctx.get()), true); }
        trace.emit("model_ready", {{"capture", trace.capture}});
    }

    void run(const json & command) {
        if (!ctx) { throw std::runtime_error("load a model first"); }
        cancelled = 0; trace.run_id = command.at("run_id"); trace.step = -1; trace.failed = false;
        trace.capture = command.value("capture", "routing"); trace.enabled = command.value("observe", true);
        const int max_tokens = command.value("max_tokens", 32);
        std::string prompt = command.value("prompt", "");
        const std::string media_path = command.value("media_path", "");
        if (!media_path.empty()) {
            if (!mm) { throw std::runtime_error("Attach a compatible multimodal projector before sending media"); }
            prompt = std::string(mtmd_get_marker(mm.get())) + "\n" + prompt;
        }
        common_chat_prompt rendered;
        bool has_template = false;
        if (templates && command.value("chat", true)) {
            common_chat_templates_inputs input;
            common_chat_msg msg; msg.role = "user"; msg.content = prompt; input.messages.push_back(msg);
            input.enable_thinking = false;
            rendered = common_chat_templates_apply(templates.get(), input).prompt;
            prompt = rendered.str(); has_template = true;
        }
        std::vector<llama_token> tokens;
        if (command.contains("token_ids")) { tokens = command.at("token_ids").get<std::vector<llama_token>>(); }
        else if (has_template) {
            bool first = true;
            for (const auto & part : rendered.parts) {
                auto part_tokens = common_tokenize(vocab, part.text, first, !part.is_input);
                tokens.insert(tokens.end(), part_tokens.begin(), part_tokens.end()); first = false;
            }
        } else { tokens = common_tokenize(vocab, prompt, true, false); }
        if (tokens.empty()) { throw std::runtime_error("the prompt contains no tokens"); }
        if (tokens.size() + size_t(max_tokens) > llama_n_ctx(ctx.get())) { throw std::runtime_error("prompt plus output exceeds the loaded context"); }
        for (const auto token : tokens) { if (token < 0 || token >= llama_vocab_n_tokens(vocab)) { throw std::runtime_error("token ID outside vocabulary"); } }
        if (llama_get_memory(ctx.get())) { llama_memory_clear(llama_get_memory(ctx.get()), true); }
        trace.emit("run_started", {{"prompt_tokens", tokens.size()}, {"capture", trace.enabled ? trace.capture : "off"}, {"token_ids", tokens}});
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        int position = 0;
        int status = 0;
        bool media_evaluated = false;
        if (!media_path.empty()) {
            auto loaded = mtmd_helper_bitmap_init_from_file(mm.get(), media_path.c_str(), false, mtmd_helper_init_opt_default());
            mtmd::bitmap_ptr bitmap(loaded.bitmap);
            if (!bitmap || loaded.video_ctx) { throw std::runtime_error("Cannot decode media; image and audio inputs are supported"); }
            mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
            const mtmd_bitmap * bitmap_ptr = bitmap.get();
            mtmd_input_text text{prompt.c_str(), prompt.size(), true, true};
            if (mtmd_tokenize(mm.get(), chunks.get(), &text, &bitmap_ptr, 1) != 0) { throw std::runtime_error("Multimodal preprocessing failed"); }
            const auto n_pos = mtmd_helper_get_n_pos(chunks.get());
            if (uint64_t(n_pos) + max_tokens > llama_n_ctx(ctx.get())) { throw std::runtime_error("Media plus output exceeds the context window"); }
            trace.begin("multimodal_prefill", 0, int(mtmd_helper_get_n_tokens(chunks.get())));
            llama_pos next = 0;
            status = mtmd_helper_eval_chunks(mm.get(), ctx.get(), chunks.get(), 0, 0, batch_size, true, &next);
            llama_synchronize(ctx.get()); trace.end(status); position = next;
            if (trace.failed) { throw std::runtime_error("Multimodal observation failed; trace is incomplete"); }
            media_evaluated = true;
        }
        if (!media_evaluated && llama_model_has_encoder(model.get())) {
            if (tokens.size() > size_t(batch_size)) { throw std::runtime_error("encoder prompt exceeds the adapter's batch size"); }
            fill(batch.get(), tokens, 0, tokens.size(), 0);
            status = process(batch.get(), LLAMA_PROCESS_TYPE_ENCODE, "encoder", 0, int(tokens.size()));
            llama_token first = llama_model_decoder_start_token(model.get()); if (first == LLAMA_TOKEN_NULL) { first = llama_vocab_bos(vocab); }
            if (llama_model_has_decoder(model.get())) { tokens = {first}; }
        }
        if (!media_evaluated && llama_model_has_decoder(model.get()) && status == 0) {
            for (size_t off = 0; off < tokens.size() && !cancelled; off += batch_size) {
                const size_t n = std::min<size_t>(batch_size, tokens.size() - off);
                fill(batch.get(), tokens, off, n, int(off));
                status = process(batch.get(), LLAMA_PROCESS_TYPE_DECODE, "prefill", int(off), int(n));
                position += int(n); if (status) { break; }
            }
        }
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
        std::string output;
        int generated = 0;
        if (llama_model_has_decoder(model.get())) {
            for (int i = 0; i < max_tokens && !status && !cancelled; ++i) {
                const llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
                const std::string piece = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE ? "" : common_token_to_piece(vocab, token, true);
                const bool eog = llama_vocab_is_eog(vocab, token);
                trace.emit("token", {{"index", i}, {"id", token}, {"text", piece}, {"eog", eog}, {"step", trace.step}});
                if (eog) { break; }
                output += piece; ++generated;
                if (i + 1 < max_tokens) {
                    fill(batch.get(), {token}, 0, 1, position);
                    status = process(batch.get(), LLAMA_PROCESS_TYPE_DECODE, "decode", position++, 1);
                }
            }
        }
        if (status && !cancelled) { throw std::runtime_error("runtime returned status " + std::to_string(status)); }
        trace.emit("run_finished", {{"status", cancelled ? "cancelled" : "completed"}, {"output", output}, {"generated_tokens", generated},
            {"steps", trace.step + 1}, {"observed", trace.enabled}});
        trace.enabled = true;
    }
};

int main() {
    std::setlocale(LC_NUMERIC, "C");
#ifdef SIGUSR1
    std::signal(SIGUSR1, cancel_signal);
#endif
    llama_log_set([](ggml_log_level, const char * text, void *) { std::cerr << text << std::flush; }, nullptr);
    llama_backend_init();
    runner engine;
    engine.trace.emit("hello", {{"runtime_version", llama_version()}, {"adapter_version", 1}});
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            if (line.size() > 2 * 1024 * 1024) { throw std::runtime_error("command exceeds 2 MiB"); }
            const json command = json::parse(line);
            const auto kind = command.at("kind").get<std::string>();
            if (kind == "shutdown") { break; }
            if (kind == "load") { engine.load(command); }
            else if (kind == "run") { engine.run(command); }
            else { throw std::runtime_error("unknown engine command"); }
        } catch (const std::exception & e) {
            engine.trace.enabled = true;
            engine.trace.emit("error", {{"message", e.what()}, {"status", cancelled ? "cancelled" : "failed"}});
        }
    }
    return 0;
}
