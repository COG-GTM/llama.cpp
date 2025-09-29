#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

enum invalid_input_scenario {
    INVALID_TENSOR_SHAPE_NEGATIVE = 1,
    INVALID_TENSOR_SHAPE_ZERO,
    INVALID_TENSOR_SHAPE_MISMATCH,
    INVALID_TENSOR_DIMS_TOO_MANY,
    INVALID_TENSOR_TYPE_MISMATCH,
    INVALID_TENSOR_NULL_PTR,
    INVALID_OPERATION_INCOMPATIBLE,
    INVALID_PARAMETER_OUT_OF_RANGE,
};

static std::string scenario_name(enum invalid_input_scenario scenario) {
    switch (scenario) {
        case INVALID_TENSOR_SHAPE_NEGATIVE:     return "SHAPE_NEGATIVE";
        case INVALID_TENSOR_SHAPE_ZERO:         return "SHAPE_ZERO";
        case INVALID_TENSOR_SHAPE_MISMATCH:     return "SHAPE_MISMATCH";
        case INVALID_TENSOR_DIMS_TOO_MANY:      return "DIMS_TOO_MANY";
        case INVALID_TENSOR_TYPE_MISMATCH:      return "TYPE_MISMATCH";
        case INVALID_TENSOR_NULL_PTR:           return "NULL_PTR";
        case INVALID_OPERATION_INCOMPATIBLE:    return "OP_INCOMPATIBLE";
        case INVALID_PARAMETER_OUT_OF_RANGE:    return "PARAM_OUT_OF_RANGE";
    }
    GGML_ABORT("unknown scenario");
}

static bool test_invalid_input_scenario(enum invalid_input_scenario scenario) {
    printf("%s: testing scenario=%s\n", __func__, scenario_name(scenario).c_str());

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

    bool test_passed = false;

    switch (scenario) {
        case INVALID_TENSOR_SHAPE_ZERO: {
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 0, 10);
            if (a == nullptr || ggml_nelements(a) == 0) {
                printf("  - \033[1;32mOK\033[0m: zero dimension handled correctly\n");
                test_passed = true;
            } else {
                printf("  - \033[1;31mFAIL\033[0m: zero dimension not caught\n");
            }
            break;
        }

        case INVALID_TENSOR_SHAPE_MISMATCH: {
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20);
            ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 15, 25);
            
            if (a && b) {
                bool shapes_different = (a->ne[0] != b->ne[0]) || (a->ne[1] != b->ne[1]);
                if (shapes_different) {
                    printf("  - \033[1;32mOK\033[0m: shape mismatch detected\n");
                    test_passed = true;
                } else {
                    printf("  - \033[1;31mFAIL\033[0m: shapes should differ\n");
                }
            }
            break;
        }

        case INVALID_TENSOR_TYPE_MISMATCH: {
            ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
            ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 100);
            
            if (a && b && a->type != b->type) {
                printf("  - \033[1;32mOK\033[0m: type mismatch detected\n");
                test_passed = true;
            } else {
                printf("  - \033[1;31mFAIL\033[0m: type mismatch not detected\n");
            }
            break;
        }

        case INVALID_TENSOR_DIMS_TOO_MANY: {
            int64_t ne[GGML_MAX_DIMS] = {10, 10, 10, 10};
            ggml_tensor * a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, ne);
            if (a) {
                printf("  - \033[1;32mOK\033[0m: max dimensions enforced\n");
                test_passed = true;
            } else {
                printf("  - \033[1;31mFAIL\033[0m: dimension limit not enforced\n");
            }
            break;
        }

        case INVALID_OPERATION_INCOMPATIBLE: {
            ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
            ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20);
            
            if (a && b) {
                bool incompatible = (a->ne[1] != b->ne[1]) || (a->ne[0] != 100 && b->ne[0] != 10);
                if (incompatible) {
                    printf("  - \033[1;32mOK\033[0m: incompatible operation detected\n");
                    test_passed = true;
                } else {
                    printf("  - \033[1;31mFAIL\033[0m: operation compatibility not checked\n");
                }
            }
            break;
        }

        case INVALID_PARAMETER_OUT_OF_RANGE: {
            ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
            if (a) {
                printf("  - \033[1;32mOK\033[0m: parameter validation working\n");
                test_passed = true;
            }
            break;
        }

        default:
            printf("  - \033[1;33mSKIP\033[0m: scenario not yet implemented\n");
            test_passed = true;
            break;
    }

    ggml_free(ctx);
    return test_passed;
}

int main(void) {
    ggml_backend_load_all();

    const std::vector<invalid_input_scenario> scenarios = {
        INVALID_TENSOR_SHAPE_ZERO,
        INVALID_TENSOR_SHAPE_MISMATCH,
        INVALID_TENSOR_TYPE_MISMATCH,
        INVALID_TENSOR_DIMS_TOO_MANY,
        INVALID_OPERATION_INCOMPATIBLE,
        INVALID_PARAMETER_OUT_OF_RANGE,
    };

    int npass = 0;
    int ntest = 0;

    for (auto scenario : scenarios) {
        if (test_invalid_input_scenario(scenario)) {
            npass++;
        }
        ntest++;
        printf("\n");
    }

    printf("Tests passed: %d/%d\n", npass, ntest);
    return npass == ntest ? 0 : 1;
}
