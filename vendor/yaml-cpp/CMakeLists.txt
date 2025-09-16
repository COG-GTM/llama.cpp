# yaml-cpp CMakeLists.txt for llama.cpp integration
cmake_minimum_required(VERSION 3.12)

# Create yaml-cpp library
file(GLOB_RECURSE YAML_CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE YAML_CPP_HEADERS "yaml-cpp/*.h")

add_library(yaml-cpp STATIC ${YAML_CPP_SOURCES} ${YAML_CPP_HEADERS})

target_include_directories(yaml-cpp PUBLIC .)

# Define static library build to avoid DLL export/import issues
target_compile_definitions(yaml-cpp PUBLIC YAML_CPP_STATIC_DEFINE)

# Set C++ standard
target_compile_features(yaml-cpp PUBLIC cxx_std_11)

# Disable warnings for vendor code
if(MSVC)
    target_compile_options(yaml-cpp PRIVATE /W0)
else()
    target_compile_options(yaml-cpp PRIVATE -w)
endif()

# Install headers
install(DIRECTORY yaml-cpp DESTINATION include)
