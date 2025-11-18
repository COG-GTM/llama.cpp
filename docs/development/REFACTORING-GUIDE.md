# llama.cpp Model Architecture Refactoring Guide

## Overview

This guide documents the refactoring of `src/llama-model.cpp` from a monolithic 20,498-line file into a modular architecture. The refactoring maintains full backwards compatibility while providing a cleaner, more maintainable structure for adding new model architectures.

## Motivation

The original `src/llama-model.cpp` file had several issues:

1. **Size**: 20,498 lines in a single file made navigation and maintenance difficult
2. **Coupling**: 89+ model builder structs (`llm_build_*`) were all defined in one file
3. **Scalability**: Adding new architectures required modifying the monolithic file
4. **Testing**: Difficult to test individual architectures in isolation

## New Architecture

### Directory Structure

```
src/llama-model/
├── architectures/       # One file per architecture
│   ├── llama.cpp       # LLM_ARCH_LLAMA implementation
│   ├── qwen2.cpp       # LLM_ARCH_QWEN2 implementation
│   ├── gemma.cpp       # LLM_ARCH_GEMMA implementation
│   └── ...             # Other architectures
├── base/               # Core model implementation
│   ├── model-base.cpp  # Core llama_model implementation
│   ├── model-loader.cpp # Tensor loading logic
│   └── model-builder.cpp # Graph building interface
├── loading/            # Model loading logic
│   ├── hparams-loader.cpp # Hyperparameter loading
│   ├── tensor-loader.cpp  # Tensor loading
│   └── vocab-loader.cpp   # Vocabulary loading
├── memory/             # Memory management
│   ├── buffer-manager.cpp # Buffer allocation
│   └── device-mapper.cpp  # Device/GPU mapping
├── interfaces/         # Interface definitions
│   └── model-interfaces.h # IModelLoader, IGraphBuilder
└── registry/           # Architecture registration
    ├── arch-registry.h    # Registry interface
    └── arch-registry.cpp  # Registry implementation
```

### Core Interfaces

#### IModelLoader

The `IModelLoader` interface defines methods for loading model components:

```cpp
class IModelLoader {
public:
    virtual ~IModelLoader() = default;
    
    // Load model statistics from GGUF file
    virtual void load_stats(llama_model_loader & ml) = 0;
    
    // Load architecture information
    virtual void load_arch(llama_model_loader & ml) = 0;
    
    // Load hyperparameters
    virtual void load_hparams(llama_model_loader & ml) = 0;
    
    // Load vocabulary
    virtual void load_vocab(llama_model_loader & ml) = 0;
    
    // Load model tensors (returns false if cancelled)
    virtual bool load_tensors(llama_model_loader & ml) = 0;
};
```

#### IGraphBuilder

The `IGraphBuilder` interface defines methods for building computation graphs:

```cpp
class IGraphBuilder {
public:
    virtual ~IGraphBuilder() = default;
    
    // Build computation graph for inference
    virtual ggml_cgraph * build_graph(const llm_graph_params & params) const = 0;
};
```

#### ArchitectureBuilder

The `ArchitectureBuilder` class combines loading and graph building:

```cpp
class ArchitectureBuilder {
public:
    virtual ~ArchitectureBuilder() = default;
    
    // Load hyperparameters for this architecture
    virtual void load_hparams(llama_model & model, llama_model_loader & ml) = 0;
    
    // Load tensors for this architecture
    virtual bool load_tensors(llama_model & model, llama_model_loader & ml) = 0;
    
    // Build computation graph for this architecture
    virtual ggml_cgraph * build_graph(const llama_model & model, const llm_graph_params & params) = 0;
    
    // Get the architecture type
    virtual llm_arch get_architecture() const = 0;
};
```

### Architecture Registry

The `ArchitectureRegistry` provides a centralized registry for architecture builders:

```cpp
class ArchitectureRegistry {
public:
    // Get singleton instance
    static ArchitectureRegistry & instance();
    
    // Register a builder instance
    void register_builder(llm_arch arch, std::unique_ptr<ArchitectureBuilder> builder);
    
    // Register a builder factory (lazy instantiation)
    void register_builder_factory(llm_arch arch, 
                                  std::function<std::unique_ptr<ArchitectureBuilder>()> factory);
    
    // Get builder for architecture (instantiates if needed)
    ArchitectureBuilder * get_builder(llm_arch arch);
    
    // Check if builder exists
    bool has_builder(llm_arch arch) const;
    
    // Clear all registrations (for testing)
    void clear();
};
```

## Adding a New Architecture

### Step 1: Create Architecture File

Create a new file in `src/llama-model/architectures/` for your architecture:

```cpp
// src/llama-model/architectures/myarch.cpp
#include "../registry/arch-registry.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-graph.h"

namespace llama {
namespace model {

class MyArchitectureBuilder : public ArchitectureBuilder {
public:
    MyArchitectureBuilder() = default;
    ~MyArchitectureBuilder() override = default;
    
    void load_hparams(llama_model & model, llama_model_loader & ml) override {
        // Load architecture-specific hyperparameters
        // Example: ml.get_key(LLM_KV_ATTENTION_HEAD_COUNT, model.hparams.n_head);
    }
    
    bool load_tensors(llama_model & model, llama_model_loader & ml) override {
        // Load architecture-specific tensors
        // Typically delegates to model.load_tensors(ml) for standard loading
        return model.load_tensors(ml);
    }
    
    ggml_cgraph * build_graph(const llama_model & model, const llm_graph_params & params) override {
        // Build computation graph for this architecture
        // This is where the model-specific inference logic goes
        return model.build_graph(params);
    }
    
    llm_arch get_architecture() const override {
        return LLM_ARCH_MYARCH;
    }
};

}
}

// Register the architecture
REGISTER_ARCHITECTURE(LLM_ARCH_MYARCH, llama::model::MyArchitectureBuilder)
```

### Step 2: Update Build System

The build system automatically includes all `.cpp` files in `src/llama-model/architectures/`:

```cmake
# In src/CMakeLists.txt
file(GLOB ARCH_SOURCES "llama-model/architectures/*.cpp")
file(GLOB REGISTRY_SOURCES "llama-model/registry/*.cpp")

add_library(llama
    # ... existing files ...
    ${ARCH_SOURCES}
    ${REGISTRY_SOURCES}
)
```

### Step 3: Test Your Architecture

Create unit tests in `tests/unit/architectures/test-myarch.cpp`:

```cpp
#include "llama-model/registry/arch-registry.h"
#include <cassert>

void test_myarch_registration() {
    auto & registry = llama::model::ArchitectureRegistry::instance();
    assert(registry.has_builder(LLM_ARCH_MYARCH));
    
    auto * builder = registry.get_builder(LLM_ARCH_MYARCH);
    assert(builder != nullptr);
    assert(builder->get_architecture() == LLM_ARCH_MYARCH);
}

int main() {
    test_myarch_registration();
    return 0;
}
```

## Backwards Compatibility

### Existing API Preserved

The refactoring maintains full backwards compatibility with the existing API:

```cpp
// src/llama-model.h - API remains unchanged
struct llama_model {
    // ... existing members ...
    
    void load_stats  (llama_model_loader & ml);
    void load_arch   (llama_model_loader & ml);
    void load_hparams(llama_model_loader & ml);
    void load_vocab  (llama_model_loader & ml);
    bool load_tensors(llama_model_loader & ml);
    
    ggml_cgraph * build_graph(const llm_graph_params & params) const;
    
    // ... existing methods ...
};
```

### Loading Flow Unchanged

The loading flow in `src/llama.cpp` (lines 101-152) remains unchanged:

```cpp
static int llama_model_load(const std::string & fname, ...) {
    llama_model_loader ml(...);
    
    model.load_arch(ml);      // Still works
    model.load_hparams(ml);   // Still works
    model.load_vocab(ml);     // Still works
    model.load_stats(ml);     // Still works
    model.load_tensors(ml);   // Still works
    
    return 0;
}
```

### Internal Implementation

Internally, the `llama_model` methods can optionally delegate to the registry:

```cpp
void llama_model::load_hparams(llama_model_loader & ml) {
    // Option 1: Use registry (new modular approach)
    auto & registry = llama::model::ArchitectureRegistry::instance();
    if (registry.has_builder(arch)) {
        auto * builder = registry.get_builder(arch);
        builder->load_hparams(*this, ml);
        return;
    }
    
    // Option 2: Fall back to existing implementation (backwards compatibility)
    // ... existing load_hparams implementation ...
}
```

## Migration Strategy

### Phase 1: Infrastructure (Current)

- ✅ Create directory structure
- ✅ Define interfaces (`IModelLoader`, `IGraphBuilder`, `ArchitectureBuilder`)
- ✅ Implement registry pattern (`ArchitectureRegistry`)
- ✅ Create example architecture (LLAMA)
- ✅ Update build system
- ✅ Add comprehensive documentation

### Phase 2: Gradual Migration

- Extract one architecture at a time (start with LLAMA)
- Verify tests pass after each extraction
- Keep existing code functional during migration
- Add deprecation warnings for old patterns

### Phase 3: Complete Migration

- Extract all 89+ architectures
- Remove deprecated code paths
- Update all documentation
- Celebrate! 🎉

## Testing

### Unit Tests

Test individual architectures in isolation:

```bash
cd build
ctest -R test-arch-llama
ctest -R test-arch-qwen2
```

### Integration Tests

Verify the entire loading pipeline:

```bash
cd build
ctest -L main --verbose
```

### Backwards Compatibility Tests

Ensure existing code still works:

```bash
cd build
ctest -R test-model-loading
```

## Code Ownership

Update `CODEOWNERS` to reflect new structure:

```
/src/llama-model/architectures/llama.cpp    @CISC
/src/llama-model/architectures/qwen2.cpp    @CISC
/src/llama-model/loading/                   @slaren
/src/llama-model/registry/                  @CISC
```

## Benefits

### Maintainability

- **Smaller files**: Each architecture in its own file (~100-200 lines)
- **Clear ownership**: Easy to identify who maintains each architecture
- **Easier navigation**: Find architecture code quickly

### Scalability

- **Easy to add**: New architectures don't touch existing code
- **Parallel development**: Multiple developers can work on different architectures
- **Reduced conflicts**: Fewer merge conflicts in large files

### Testability

- **Isolated testing**: Test each architecture independently
- **Faster iteration**: Compile only changed architectures
- **Better coverage**: Easier to achieve high test coverage

### Extensibility

- **Plugin architecture**: Architectures can be added as plugins
- **Dynamic loading**: Future support for loading architectures at runtime
- **Experimentation**: Easy to experiment with new architectures

## Common Patterns

### Pattern 1: Standard Architecture

Most architectures follow this pattern:

```cpp
class StandardArchBuilder : public ArchitectureBuilder {
    void load_hparams(llama_model & model, llama_model_loader & ml) override {
        model.load_hparams(ml);  // Use default implementation
    }
    
    bool load_tensors(llama_model & model, llama_model_loader & ml) override {
        return model.load_tensors(ml);  // Use default implementation
    }
    
    ggml_cgraph * build_graph(const llama_model & model, const llm_graph_params & params) override {
        // Custom graph building logic
        std::unique_ptr<llm_graph_context> llm = std::make_unique<llm_build_myarch>(model, params);
        return llm->gf;
    }
    
    llm_arch get_architecture() const override {
        return LLM_ARCH_MYARCH;
    }
};
```

### Pattern 2: Architecture with Custom Loading

Some architectures need custom loading logic:

```cpp
class CustomLoadArchBuilder : public ArchitectureBuilder {
    void load_hparams(llama_model & model, llama_model_loader & ml) override {
        // Custom hyperparameter loading
        ml.get_key(LLM_KV_CUSTOM_PARAM, model.hparams.custom_param);
        model.load_hparams(ml);  // Then call default
    }
    
    // ... rest of implementation ...
};
```

### Pattern 3: Architecture Variants

Handle architecture variants with template parameters:

```cpp
template<bool USE_SWA>
class VariantArchBuilder : public ArchitectureBuilder {
    ggml_cgraph * build_graph(const llama_model & model, const llm_graph_params & params) override {
        if constexpr (USE_SWA) {
            return std::make_unique<llm_build_myarch_swa>(model, params)->gf;
        } else {
            return std::make_unique<llm_build_myarch>(model, params)->gf;
        }
    }
};

// Register both variants
REGISTER_ARCHITECTURE(LLM_ARCH_MYARCH, VariantArchBuilder<false>)
REGISTER_ARCHITECTURE(LLM_ARCH_MYARCH_SWA, VariantArchBuilder<true>)
```

## Troubleshooting

### Issue: Architecture not found

**Symptom**: `no builder registered for architecture X`

**Solution**: Ensure the architecture file is:
1. Created in `src/llama-model/architectures/`
2. Uses `REGISTER_ARCHITECTURE` macro
3. Included in the build system

### Issue: Linker errors

**Symptom**: `undefined reference to llama::model::MyArchitectureBuilder`

**Solution**: 
1. Check that the `.cpp` file is in `src/llama-model/architectures/`
2. Verify `CMakeLists.txt` includes `${ARCH_SOURCES}`
3. Rebuild: `cmake --build build --clean-first`

### Issue: Tests fail after migration

**Symptom**: Existing tests fail with new architecture

**Solution**:
1. Verify backwards compatibility shims are in place
2. Check that the new implementation matches old behavior
3. Run with `LLAMA_GRAPH_INPUT_DEBUG=1` for detailed logging

## Future Enhancements

### Dynamic Loading

Future versions could support loading architectures as plugins:

```cpp
// Load architecture from shared library
registry.load_plugin("libllama-arch-myarch.so");
```

### Architecture Metadata

Add metadata for better introspection:

```cpp
struct ArchitectureMetadata {
    std::string name;
    std::string description;
    std::string author;
    std::vector<std::string> supported_features;
};
```

### Performance Optimizations

- Lazy initialization of builders
- Caching of frequently-used graphs
- Parallel loading of multiple architectures

## References

- Original implementation: `src/llama-model.cpp` (20,498 lines)
- Architecture enums: `src/llama-arch.h`
- Model interface: `src/llama-model.h`
- Loading flow: `src/llama.cpp` (lines 101-152)
- Python conversion pattern: `docs/development/HOWTO-add-model.md`

## Questions?

For questions or issues with the refactoring:

1. Check this guide first
2. Review the example architectures in `src/llama-model/architectures/`
3. Ask in the llama.cpp Discord or GitHub discussions
4. File an issue on GitHub with the `refactoring` label
