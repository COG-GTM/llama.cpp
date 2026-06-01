#include "arch-registry.h"
#include "llama-impl.h"

#include <stdexcept>

namespace llama {
namespace model {

ArchitectureRegistry & ArchitectureRegistry::instance() {
    static ArchitectureRegistry registry;
    return registry;
}

void ArchitectureRegistry::register_builder(llm_arch arch, std::unique_ptr<ArchitectureBuilder> builder) {
    if (builder == nullptr) {
        throw std::invalid_argument("Cannot register null builder");
    }
    builders_[arch] = std::move(builder);
}

void ArchitectureRegistry::register_builder_factory(
    llm_arch arch,
    std::function<std::unique_ptr<ArchitectureBuilder>()> factory) {
    if (!factory) {
        throw std::invalid_argument("Cannot register null factory");
    }
    factories_[arch] = std::move(factory);
}

ArchitectureBuilder * ArchitectureRegistry::get_builder(llm_arch arch) {
    auto it = builders_.find(arch);
    if (it != builders_.end()) {
        return it->second.get();
    }

    auto factory_it = factories_.find(arch);
    if (factory_it != factories_.end()) {
        auto builder = factory_it->second();
        auto * builder_ptr = builder.get();
        builders_[arch] = std::move(builder);
        return builder_ptr;
    }

    LLAMA_LOG_WARN("%s: no builder registered for architecture %d\n", __func__, static_cast<int>(arch));
    return nullptr;
}

bool ArchitectureRegistry::has_builder(llm_arch arch) const {
    return builders_.find(arch) != builders_.end() ||
           factories_.find(arch) != factories_.end();
}

void ArchitectureRegistry::clear() {
    builders_.clear();
    factories_.clear();
}

}
}
