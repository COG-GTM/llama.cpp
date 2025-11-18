#include "../registry/arch-registry.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-graph.h"
#include "llama-impl.h"

namespace llama {
namespace model {

class LlamaArchitectureBuilder : public ArchitectureBuilder {
public:
    LlamaArchitectureBuilder() = default;
    ~LlamaArchitectureBuilder() override = default;

    void load_hparams(llama_model & model, llama_model_loader & ml) override {
        model.load_hparams(ml);
    }

    bool load_tensors(llama_model & model, llama_model_loader & ml) override {
        return model.load_tensors(ml);
    }

    ggml_cgraph * build_graph(const llama_model & model, const llm_graph_params & params) override {
        return model.build_graph(params);
    }

    llm_arch get_architecture() const override {
        return LLM_ARCH_LLAMA;
    }
};

}
}

REGISTER_ARCHITECTURE(LLM_ARCH_LLAMA, llama::model::LlamaArchitectureBuilder)
