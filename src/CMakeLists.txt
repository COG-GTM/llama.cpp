llama_add_compile_flags()

#
# libraries
#

# llama

# Collect modular architecture sources
file(GLOB ARCH_SOURCES "llama-model/architectures/*.cpp")
file(GLOB LOADER_SOURCES "llama-model/loading/*.cpp")
file(GLOB MEMORY_SOURCES "llama-model/memory/*.cpp")
file(GLOB BASE_SOURCES "llama-model/base/*.cpp")
file(GLOB REGISTRY_SOURCES "llama-model/registry/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${ARCH_SOURCES}
            ${LOADER_SOURCES}
            ${MEMORY_SOURCES}
            ${BASE_SOURCES}
            ${REGISTRY_SOURCES}
            )

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
