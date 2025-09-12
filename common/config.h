#pragma once

#include "common.h"
#include <string>

#ifdef LLAMA_ENABLE_CONFIG_YAML
bool common_load_yaml_config(const std::string & path, common_params & params);
std::string common_yaml_valid_keys_help();
#endif
