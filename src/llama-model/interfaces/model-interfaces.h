#pragma once

#include "llama-arch.h"
#include "llama-hparams.h"

#include <memory>

struct ggml_cgraph;
struct llama_model_loader;
struct llama_model;
struct llm_graph_params;

namespace llama {
namespace model {

class IModelLoader {
public:
    virtual ~IModelLoader() = default;

    virtual void load_stats(llama_model_loader & ml) = 0;

    virtual void load_arch(llama_model_loader & ml) = 0;

    virtual void load_hparams(llama_model_loader & ml) = 0;

    virtual void load_vocab(llama_model_loader & ml) = 0;

    virtual bool load_tensors(llama_model_loader & ml) = 0;
};

class IGraphBuilder {
public:
    virtual ~IGraphBuilder() = default;

    virtual ggml_cgraph * build_graph(const llm_graph_params & params) const = 0;
};

class IArchitectureHandler : public IModelLoader, public IGraphBuilder {
public:
    virtual ~IArchitectureHandler() = default;

    virtual llm_arch get_architecture() const = 0;
};

}
}
