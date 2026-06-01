#pragma once

#include "llama-arch.h"
#include "../interfaces/model-interfaces.h"

#include <memory>
#include <unordered_map>
#include <functional>

struct llama_model;
struct llama_model_loader;
struct llm_graph_params;
struct ggml_cgraph;

namespace llama {
namespace model {

class ArchitectureBuilder {
public:
    virtual ~ArchitectureBuilder() = default;

    virtual void load_hparams(llama_model & model, llama_model_loader & ml) = 0;

    virtual bool load_tensors(llama_model & model, llama_model_loader & ml) = 0;

    virtual ggml_cgraph * build_graph(const llama_model & model, const llm_graph_params & params) = 0;

    virtual llm_arch get_architecture() const = 0;
};

class ArchitectureRegistry {
public:
    static ArchitectureRegistry & instance();

    void register_builder(llm_arch arch, std::unique_ptr<ArchitectureBuilder> builder);

    void register_builder_factory(llm_arch arch, std::function<std::unique_ptr<ArchitectureBuilder>()> factory);

    ArchitectureBuilder * get_builder(llm_arch arch);

    bool has_builder(llm_arch arch) const;

    void clear();

private:
    ArchitectureRegistry() = default;
    ~ArchitectureRegistry() = default;

    ArchitectureRegistry(const ArchitectureRegistry &) = delete;
    ArchitectureRegistry & operator=(const ArchitectureRegistry &) = delete;

    std::unordered_map<llm_arch, std::unique_ptr<ArchitectureBuilder>> builders_;
    std::unordered_map<llm_arch, std::function<std::unique_ptr<ArchitectureBuilder>()>> factories_;
};

template<typename BuilderType>
class ArchitectureRegistrar {
public:
    explicit ArchitectureRegistrar(llm_arch arch) {
        ArchitectureRegistry::instance().register_builder_factory(
            arch,
            []() -> std::unique_ptr<ArchitectureBuilder> {
                return std::make_unique<BuilderType>();
            }
        );
    }
};

}
}

#define REGISTER_ARCHITECTURE(arch, builder_class) \
    namespace { \
        static ::llama::model::ArchitectureRegistrar<builder_class> \
            registrar_##arch##_##__LINE__(arch); \
    }
