set(TARGET llama-model-hub)

add_executable(${TARGET} model-hub.cpp)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE common)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)
