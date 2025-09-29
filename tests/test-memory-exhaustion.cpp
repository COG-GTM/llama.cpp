#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
    #define setenv_portable(name, value) _putenv_s(name, value)
    #define unsetenv_portable(name) _putenv_s(name, "")
#else
    #define setenv_portable(name, value) setenv(name, value, 1)
    #define unsetenv_portable(name) unsetenv(name)
#endif

enum memory_exhaustion_scenario {
    MEM_EXHAUST_SMALL_ALLOC = 1,
    MEM_EXHAUST_MEDIUM_ALLOC,
    MEM_EXHAUST_LARGE_ALLOC,
    MEM_EXHAUST_MANY_ALLOCS,
    MEM_EXHAUST_FRAGMENTATION,
    MEM_EXHAUST_BUFFER_OVERFLOW,
    MEM_EXHAUST_RECOVERY,
};

static std::string scenario_name(enum memory_exhaustion_scenario scenario) {
    switch (scenario) {
        case MEM_EXHAUST_SMALL_ALLOC:     return "SMALL_ALLOC";
        case MEM_EXHAUST_MEDIUM_ALLOC:    return "MEDIUM_ALLOC";
        case MEM_EXHAUST_LARGE_ALLOC:     return "LARGE_ALLOC";
        case MEM_EXHAUST_MANY_ALLOCS:     return "MANY_ALLOCS";
        case MEM_EXHAUST_FRAGMENTATION:   return "FRAGMENTATION";
        case MEM_EXHAUST_BUFFER_OVERFLOW: return "BUFFER_OVERFLOW";
        case MEM_EXHAUST_RECOVERY:        return "RECOVERY";
    }
    GGML_ABORT("unknown scenario");
}

static bool should_fail(enum memory_exhaustion_scenario scenario) {
    return scenario != MEM_EXHAUST_RECOVERY;
}

static bool test_memory_exhaustion_scenario(ggml_backend_t backend, enum memory_exhaustion_scenario scenario) {
    printf("%s: testing scenario=%s\n", __func__, scenario_name(scenario).c_str());

    switch (scenario) {
        case MEM_EXHAUST_SMALL_ALLOC:
            setenv_portable("GGML_ALLOC_FAIL_THRESHOLD", "1024");
            break;
        case MEM_EXHAUST_MEDIUM_ALLOC:
            setenv_portable("GGML_ALLOC_FAIL_THRESHOLD", "1048576");
            break;
        case MEM_EXHAUST_LARGE_ALLOC:
            setenv_portable("GGML_ALLOC_FAIL_THRESHOLD", "10485760");
            break;
        case MEM_EXHAUST_MANY_ALLOCS:
            setenv_portable("GGML_ALLOC_FAIL_COUNT", "10");
            break;
        case MEM_EXHAUST_BUFFER_OVERFLOW:
            setenv_portable("GGML_ALLOC_FAIL_THRESHOLD", "100");
            break;
        default:
            unsetenv_portable("GGML_ALLOC_FAIL_THRESHOLD");
            unsetenv_portable("GGML_ALLOC_FAIL_COUNT");
            break;
    }

    ggml_init_params params = {
        ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        NULL,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  - failed to create context\n");
        return false;
    }

    ggml_tensor * a = nullptr;

    switch (scenario) {
        case MEM_EXHAUST_SMALL_ALLOC:
            a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
            break;
        case MEM_EXHAUST_MEDIUM_ALLOC:
            a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
            break;
        case MEM_EXHAUST_LARGE_ALLOC:
            a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 2048);
            break;
        case MEM_EXHAUST_MANY_ALLOCS:
            for (int i = 0; i < 15; i++) {
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
            }
            a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
            break;
        default:
            a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
            break;
    }

    if (!a) {
        printf("  - failed to create tensor\n");
        ggml_free(ctx);
        return false;
    }

    ggml_set_name(a, "a");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    bool test_passed = false;
    if (should_fail(scenario)) {
        if (buf == NULL) {
            printf("  - \033[1;32mOK\033[0m: allocation failed as expected\n");
            test_passed = true;
        } else {
            printf("  - \033[1;31mFAIL\033[0m: allocation succeeded when it should have failed\n");
            ggml_backend_buffer_free(buf);
        }
    } else {
        if (buf != NULL) {
            printf("  - \033[1;32mOK\033[0m: allocation succeeded as expected\n");
            test_passed = true;
            ggml_backend_buffer_free(buf);
        } else {
            printf("  - \033[1;31mFAIL\033[0m: allocation failed when it should have succeeded\n");
        }
    }

    ggml_free(ctx);

    unsetenv_portable("GGML_ALLOC_FAIL_THRESHOLD");
    unsetenv_portable("GGML_ALLOC_FAIL_COUNT");

    return test_passed;
}

int main(void) {
    ggml_backend_load_all();

    const std::vector<memory_exhaustion_scenario> scenarios = {
        MEM_EXHAUST_SMALL_ALLOC,
        MEM_EXHAUST_MEDIUM_ALLOC,
        MEM_EXHAUST_LARGE_ALLOC,
        MEM_EXHAUST_MANY_ALLOCS,
        MEM_EXHAUST_BUFFER_OVERFLOW,
        MEM_EXHAUST_RECOVERY,
    };

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        fprintf(stderr, "Failed to initialize CPU backend\n");
        return 1;
    }

    int npass = 0;
    int ntest = 0;

    for (auto scenario : scenarios) {
        if (test_memory_exhaustion_scenario(backend, scenario)) {
            npass++;
        }
        ntest++;
        printf("\n");
    }

    ggml_backend_free(backend);

    printf("Tests passed: %d/%d\n", npass, ntest);
    return npass == ntest ? 0 : 1;
}
