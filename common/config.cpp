#include "config.h"
#include "log.h"

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

static std::set<std::string> get_valid_keys() {
    return {
        "model.path", "model.url", "model.hf_repo", "model.hf_file",
        "model_alias", "hf_token", "prompt", "system_prompt", "prompt_file",
        "n_predict", "n_ctx", "n_batch", "n_ubatch", "n_keep", "n_chunks",
        "n_parallel", "n_sequences", "grp_attn_n", "grp_attn_w", "n_print",
        "rope_freq_base", "rope_freq_scale", "yarn_ext_factor", "yarn_attn_factor",
        "yarn_beta_fast", "yarn_beta_slow", "yarn_orig_ctx",
        "n_gpu_layers", "main_gpu", "split_mode", "pooling_type", "attention_type",
        "flash_attn_type", "numa", "use_mmap", "use_mlock", "verbose_prompt",
        "display_prompt", "no_kv_offload", "warmup", "check_tensors", "no_op_offload",
        "no_extra_bufts", "cache_type_k", "cache_type_v", "conversation_mode",
        "simple_io", "interactive", "interactive_first", "input_prefix", "input_suffix",
        "logits_file", "path_prompt_cache", "antiprompt", "in_files", "kv_overrides",
        "tensor_buft_overrides", "lora_adapters", "control_vectors", "image", "seed",
        "sampling.seed", "sampling.n_prev", "sampling.n_probs", "sampling.min_keep",
        "sampling.top_k", "sampling.top_p", "sampling.min_p", "sampling.xtc_probability",
        "sampling.xtc_threshold", "sampling.typ_p", "sampling.temp", "sampling.dynatemp_range",
        "sampling.dynatemp_exponent", "sampling.penalty_last_n", "sampling.penalty_repeat",
        "sampling.penalty_freq", "sampling.penalty_present", "sampling.dry_multiplier",
        "sampling.dry_base", "sampling.dry_allowed_length", "sampling.dry_penalty_last_n",
        "sampling.mirostat", "sampling.mirostat_tau", "sampling.mirostat_eta",
        "sampling.top_n_sigma", "sampling.ignore_eos", "sampling.no_perf",
        "sampling.timing_per_token", "sampling.dry_sequence_breakers", "sampling.samplers",
        "sampling.grammar", "sampling.grammar_lazy", "sampling.grammar_triggers",
        "speculative.devices", "speculative.n_ctx", "speculative.n_max", "speculative.n_min",
        "speculative.n_gpu_layers", "speculative.p_split", "speculative.p_min",
        "speculative.model.path", "speculative.model.url", "speculative.model.hf_repo",
        "speculative.model.hf_file", "speculative.tensor_buft_overrides",
        "speculative.cpuparams", "speculative.cpuparams_batch",
        "vocoder.model.path", "vocoder.model.url", "vocoder.model.hf_repo",
        "vocoder.model.hf_file", "vocoder.speaker_file", "vocoder.use_guide_tokens"
    };
}

std::string common_yaml_valid_keys_help() {
    const auto keys = get_valid_keys();
    std::ostringstream ss;
    bool first = true;
    for (const auto & key : keys) {
        if (!first) ss << ", ";
        ss << key;
        first = false;
    }
    return ss.str();
}

static std::string resolve_path(const std::string & path, const fs::path & yaml_dir) {
    fs::path p(path);
    if (p.is_absolute()) {
        return path;
    }
    return fs::weakly_canonical(yaml_dir / p).string();
}

static void collect_keys(const YAML::Node & node, const std::string & prefix, std::set<std::string> & found_keys) {
    if (node.IsMap()) {
        for (const auto & kv : node) {
            std::string key = kv.first.as<std::string>();
            std::string full_key = prefix.empty() ? key : prefix + "." + key;
            found_keys.insert(full_key);
            collect_keys(kv.second, full_key, found_keys);
        }
    }
}

static void validate_keys(const YAML::Node & root) {
    std::set<std::string> found_keys;
    collect_keys(root, "", found_keys);
    
    const auto valid_keys = get_valid_keys();
    std::vector<std::string> unknown_keys;
    
    for (const auto & key : found_keys) {
        if (valid_keys.find(key) == valid_keys.end()) {
            bool is_parent = false;
            for (const auto & valid_key : valid_keys) {
                if (valid_key.find(key + ".") == 0) {
                    is_parent = true;
                    break;
                }
            }
            if (!is_parent) {
                unknown_keys.push_back(key);
            }
        }
    }
    
    if (!unknown_keys.empty()) {
        std::ostringstream ss;
        ss << "Unknown YAML keys: ";
        for (size_t i = 0; i < unknown_keys.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << unknown_keys[i];
        }
        ss << "; valid keys are: " << common_yaml_valid_keys_help();
        throw std::invalid_argument(ss.str());
    }
}

static ggml_type parse_ggml_type(const std::string & type_str) {
    if (type_str == "f32") return GGML_TYPE_F32;
    if (type_str == "f16") return GGML_TYPE_F16;
    if (type_str == "bf16") return GGML_TYPE_BF16;
    if (type_str == "q8_0") return GGML_TYPE_Q8_0;
    if (type_str == "q4_0") return GGML_TYPE_Q4_0;
    if (type_str == "q4_1") return GGML_TYPE_Q4_1;
    if (type_str == "iq4_nl") return GGML_TYPE_IQ4_NL;
    if (type_str == "q5_0") return GGML_TYPE_Q5_0;
    if (type_str == "q5_1") return GGML_TYPE_Q5_1;
    throw std::invalid_argument("Unknown ggml_type: " + type_str);
}

static enum llama_split_mode parse_split_mode(const std::string & mode_str) {
    if (mode_str == "none") return LLAMA_SPLIT_MODE_NONE;
    if (mode_str == "layer") return LLAMA_SPLIT_MODE_LAYER;
    if (mode_str == "row") return LLAMA_SPLIT_MODE_ROW;
    throw std::invalid_argument("Unknown split_mode: " + mode_str);
}

static enum llama_pooling_type parse_pooling_type(const std::string & type_str) {
    if (type_str == "unspecified") return LLAMA_POOLING_TYPE_UNSPECIFIED;
    if (type_str == "none") return LLAMA_POOLING_TYPE_NONE;
    if (type_str == "mean") return LLAMA_POOLING_TYPE_MEAN;
    if (type_str == "cls") return LLAMA_POOLING_TYPE_CLS;
    if (type_str == "last") return LLAMA_POOLING_TYPE_LAST;
    if (type_str == "rank") return LLAMA_POOLING_TYPE_RANK;
    throw std::invalid_argument("Unknown pooling_type: " + type_str);
}

static enum llama_attention_type parse_attention_type(const std::string & type_str) {
    if (type_str == "unspecified") return LLAMA_ATTENTION_TYPE_UNSPECIFIED;
    if (type_str == "causal") return LLAMA_ATTENTION_TYPE_CAUSAL;
    if (type_str == "non_causal") return LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    throw std::invalid_argument("Unknown attention_type: " + type_str);
}

static enum llama_flash_attn_type parse_flash_attn_type(const std::string & type_str) {
    if (type_str == "auto") return LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (type_str == "disabled") return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (type_str == "enabled") return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    throw std::invalid_argument("Unknown flash_attn_type: " + type_str);
}

static ggml_numa_strategy parse_numa_strategy(const std::string & strategy_str) {
    if (strategy_str == "disabled") return GGML_NUMA_STRATEGY_DISABLED;
    if (strategy_str == "distribute") return GGML_NUMA_STRATEGY_DISTRIBUTE;
    if (strategy_str == "isolate") return GGML_NUMA_STRATEGY_ISOLATE;
    if (strategy_str == "numactl") return GGML_NUMA_STRATEGY_NUMACTL;
    if (strategy_str == "mirror") return GGML_NUMA_STRATEGY_MIRROR;
    throw std::invalid_argument("Unknown numa_strategy: " + strategy_str);
}

static common_conversation_mode parse_conversation_mode(const std::string & mode_str) {
    if (mode_str == "auto") return COMMON_CONVERSATION_MODE_AUTO;
    if (mode_str == "enabled") return COMMON_CONVERSATION_MODE_ENABLED;
    if (mode_str == "disabled") return COMMON_CONVERSATION_MODE_DISABLED;
    throw std::invalid_argument("Unknown conversation_mode: " + mode_str);
}

bool common_load_yaml_config(const std::string & path, common_params & params) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        
        validate_keys(root);
        
        fs::path yaml_dir = fs::absolute(path).parent_path();
        
        if (root["model"]) {
            auto model = root["model"];
            if (model["path"]) {
                params.model.path = resolve_path(model["path"].as<std::string>(), yaml_dir);
            }
            if (model["url"]) {
                params.model.url = model["url"].as<std::string>();
            }
            if (model["hf_repo"]) {
                params.model.hf_repo = model["hf_repo"].as<std::string>();
            }
            if (model["hf_file"]) {
                params.model.hf_file = model["hf_file"].as<std::string>();
            }
        }
        
        if (root["model_alias"]) params.model_alias = root["model_alias"].as<std::string>();
        if (root["hf_token"]) params.hf_token = root["hf_token"].as<std::string>();
        if (root["prompt"]) params.prompt = root["prompt"].as<std::string>();
        if (root["system_prompt"]) params.system_prompt = root["system_prompt"].as<std::string>();
        if (root["prompt_file"]) {
            params.prompt_file = resolve_path(root["prompt_file"].as<std::string>(), yaml_dir);
        }
        
        if (root["n_predict"]) params.n_predict = root["n_predict"].as<int32_t>();
        if (root["n_ctx"]) params.n_ctx = root["n_ctx"].as<int32_t>();
        if (root["n_batch"]) params.n_batch = root["n_batch"].as<int32_t>();
        if (root["n_ubatch"]) params.n_ubatch = root["n_ubatch"].as<int32_t>();
        if (root["n_keep"]) params.n_keep = root["n_keep"].as<int32_t>();
        if (root["n_chunks"]) params.n_chunks = root["n_chunks"].as<int32_t>();
        if (root["n_parallel"]) params.n_parallel = root["n_parallel"].as<int32_t>();
        if (root["n_sequences"]) params.n_sequences = root["n_sequences"].as<int32_t>();
        if (root["grp_attn_n"]) params.grp_attn_n = root["grp_attn_n"].as<int32_t>();
        if (root["grp_attn_w"]) params.grp_attn_w = root["grp_attn_w"].as<int32_t>();
        if (root["n_print"]) params.n_print = root["n_print"].as<int32_t>();
        
        if (root["rope_freq_base"]) params.rope_freq_base = root["rope_freq_base"].as<float>();
        if (root["rope_freq_scale"]) params.rope_freq_scale = root["rope_freq_scale"].as<float>();
        if (root["yarn_ext_factor"]) params.yarn_ext_factor = root["yarn_ext_factor"].as<float>();
        if (root["yarn_attn_factor"]) params.yarn_attn_factor = root["yarn_attn_factor"].as<float>();
        if (root["yarn_beta_fast"]) params.yarn_beta_fast = root["yarn_beta_fast"].as<float>();
        if (root["yarn_beta_slow"]) params.yarn_beta_slow = root["yarn_beta_slow"].as<float>();
        if (root["yarn_orig_ctx"]) params.yarn_orig_ctx = root["yarn_orig_ctx"].as<int32_t>();
        
        if (root["n_gpu_layers"]) params.n_gpu_layers = root["n_gpu_layers"].as<int32_t>();
        if (root["main_gpu"]) params.main_gpu = root["main_gpu"].as<int32_t>();
        
        if (root["split_mode"]) {
            params.split_mode = parse_split_mode(root["split_mode"].as<std::string>());
        }
        if (root["pooling_type"]) {
            params.pooling_type = parse_pooling_type(root["pooling_type"].as<std::string>());
        }
        if (root["attention_type"]) {
            params.attention_type = parse_attention_type(root["attention_type"].as<std::string>());
        }
        if (root["flash_attn_type"]) {
            params.flash_attn_type = parse_flash_attn_type(root["flash_attn_type"].as<std::string>());
        }
        if (root["numa"]) {
            params.numa = parse_numa_strategy(root["numa"].as<std::string>());
        }
        if (root["conversation_mode"]) {
            params.conversation_mode = parse_conversation_mode(root["conversation_mode"].as<std::string>());
        }
        
        if (root["use_mmap"]) params.use_mmap = root["use_mmap"].as<bool>();
        if (root["use_mlock"]) params.use_mlock = root["use_mlock"].as<bool>();
        if (root["verbose_prompt"]) params.verbose_prompt = root["verbose_prompt"].as<bool>();
        if (root["display_prompt"]) params.display_prompt = root["display_prompt"].as<bool>();
        if (root["no_kv_offload"]) params.no_kv_offload = root["no_kv_offload"].as<bool>();
        if (root["warmup"]) params.warmup = root["warmup"].as<bool>();
        if (root["check_tensors"]) params.check_tensors = root["check_tensors"].as<bool>();
        if (root["no_op_offload"]) params.no_op_offload = root["no_op_offload"].as<bool>();
        if (root["no_extra_bufts"]) params.no_extra_bufts = root["no_extra_bufts"].as<bool>();
        if (root["simple_io"]) params.simple_io = root["simple_io"].as<bool>();
        if (root["interactive"]) params.interactive = root["interactive"].as<bool>();
        if (root["interactive_first"]) params.interactive_first = root["interactive_first"].as<bool>();
        
        if (root["input_prefix"]) params.input_prefix = root["input_prefix"].as<std::string>();
        if (root["input_suffix"]) params.input_suffix = root["input_suffix"].as<std::string>();
        if (root["logits_file"]) {
            params.logits_file = resolve_path(root["logits_file"].as<std::string>(), yaml_dir);
        }
        if (root["path_prompt_cache"]) {
            params.path_prompt_cache = resolve_path(root["path_prompt_cache"].as<std::string>(), yaml_dir);
        }
        
        if (root["cache_type_k"]) {
            params.cache_type_k = parse_ggml_type(root["cache_type_k"].as<std::string>());
        }
        if (root["cache_type_v"]) {
            params.cache_type_v = parse_ggml_type(root["cache_type_v"].as<std::string>());
        }
        
        if (root["antiprompt"]) {
            params.antiprompt.clear();
            for (const auto & item : root["antiprompt"]) {
                params.antiprompt.push_back(item.as<std::string>());
            }
        }
        
        if (root["in_files"]) {
            params.in_files.clear();
            for (const auto & item : root["in_files"]) {
                params.in_files.push_back(resolve_path(item.as<std::string>(), yaml_dir));
            }
        }
        
        if (root["image"]) {
            params.image.clear();
            for (const auto & item : root["image"]) {
                params.image.push_back(resolve_path(item.as<std::string>(), yaml_dir));
            }
        }
        
        if (root["seed"]) {
            params.sampling.seed = root["seed"].as<uint32_t>();
        }
        
        if (root["sampling"]) {
            auto sampling = root["sampling"];
            if (sampling["seed"]) params.sampling.seed = sampling["seed"].as<uint32_t>();
            if (sampling["n_prev"]) params.sampling.n_prev = sampling["n_prev"].as<int32_t>();
            if (sampling["n_probs"]) params.sampling.n_probs = sampling["n_probs"].as<int32_t>();
            if (sampling["min_keep"]) params.sampling.min_keep = sampling["min_keep"].as<int32_t>();
            if (sampling["top_k"]) params.sampling.top_k = sampling["top_k"].as<int32_t>();
            if (sampling["top_p"]) params.sampling.top_p = sampling["top_p"].as<float>();
            if (sampling["min_p"]) params.sampling.min_p = sampling["min_p"].as<float>();
            if (sampling["xtc_probability"]) params.sampling.xtc_probability = sampling["xtc_probability"].as<float>();
            if (sampling["xtc_threshold"]) params.sampling.xtc_threshold = sampling["xtc_threshold"].as<float>();
            if (sampling["typ_p"]) params.sampling.typ_p = sampling["typ_p"].as<float>();
            if (sampling["temp"]) params.sampling.temp = sampling["temp"].as<float>();
            if (sampling["dynatemp_range"]) params.sampling.dynatemp_range = sampling["dynatemp_range"].as<float>();
            if (sampling["dynatemp_exponent"]) params.sampling.dynatemp_exponent = sampling["dynatemp_exponent"].as<float>();
            if (sampling["penalty_last_n"]) params.sampling.penalty_last_n = sampling["penalty_last_n"].as<int32_t>();
            if (sampling["penalty_repeat"]) params.sampling.penalty_repeat = sampling["penalty_repeat"].as<float>();
            if (sampling["penalty_freq"]) params.sampling.penalty_freq = sampling["penalty_freq"].as<float>();
            if (sampling["penalty_present"]) params.sampling.penalty_present = sampling["penalty_present"].as<float>();
            if (sampling["dry_multiplier"]) params.sampling.dry_multiplier = sampling["dry_multiplier"].as<float>();
            if (sampling["dry_base"]) params.sampling.dry_base = sampling["dry_base"].as<float>();
            if (sampling["dry_allowed_length"]) params.sampling.dry_allowed_length = sampling["dry_allowed_length"].as<int32_t>();
            if (sampling["dry_penalty_last_n"]) params.sampling.dry_penalty_last_n = sampling["dry_penalty_last_n"].as<int32_t>();
            if (sampling["mirostat"]) params.sampling.mirostat = sampling["mirostat"].as<int32_t>();
            if (sampling["mirostat_tau"]) params.sampling.mirostat_tau = sampling["mirostat_tau"].as<float>();
            if (sampling["mirostat_eta"]) params.sampling.mirostat_eta = sampling["mirostat_eta"].as<float>();
            if (sampling["top_n_sigma"]) params.sampling.top_n_sigma = sampling["top_n_sigma"].as<float>();
            if (sampling["ignore_eos"]) params.sampling.ignore_eos = sampling["ignore_eos"].as<bool>();
            if (sampling["no_perf"]) params.sampling.no_perf = sampling["no_perf"].as<bool>();
            if (sampling["timing_per_token"]) params.sampling.timing_per_token = sampling["timing_per_token"].as<bool>();
            if (sampling["grammar"]) params.sampling.grammar = sampling["grammar"].as<std::string>();
            if (sampling["grammar_lazy"]) params.sampling.grammar_lazy = sampling["grammar_lazy"].as<bool>();
        }
        
        return true;
    } catch (const YAML::Exception & e) {
        throw std::invalid_argument("YAML parsing error: " + std::string(e.what()));
    } catch (const std::exception & e) {
        throw std::invalid_argument("Config loading error: " + std::string(e.what()));
    }
}
