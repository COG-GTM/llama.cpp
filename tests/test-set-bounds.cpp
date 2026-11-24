// CPU SET kernel bounds: writes that end exactly at the destination end are valid, unused strides are ignored
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

static ggml_context * make_ctx() {
    ggml_init_params params = { 16*1024*1024, nullptr, false };
    return ggml_init(params);
}

static void fill(ggml_tensor * t, float base) {
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        ggml_set_f32_1d(t, i, base + i);
    }
}

static bool compute(ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    return ggml_graph_compute_with_ctx(ctx, gf, 2) == GGML_STATUS_SUCCESS;
}

int main() {
    int n_failed = 0;

    // F32 2D patch whose last element is the last element of the destination
    {
        ggml_context * ctx = make_ctx();
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
        fill(a, 0.0f);
        fill(b, 100.0f);
        ggml_tensor * out = ggml_set_2d(ctx, a, b, a->nb[1], (2*8 + 5)*sizeof(float));
        bool ok = compute(ctx, out);
        const float * o = (const float *) out->data;
        ok = ok && o[2*8 + 5] == 100.0f && o[2*8 + 7] == 102.0f && o[3*8 + 5] == 103.0f && o[31] == 105.0f && o[2*8 + 4] == 20.0f;
        printf("  %-50s %s\n", "f32 patch ending at the destination end", ok ? "OK" : "FAIL");
        n_failed += !ok;
        ggml_free(ctx);
    }

    // I32 1D write of the final element
    {
        ggml_context * ctx = make_ctx();
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        for (int i = 0; i < 16; i++) {
            ggml_set_i32_1d(a, i, i);
        }
        ggml_set_i32_1d(b, 0, -7);
        ggml_tensor * out = ggml_set_1d(ctx, a, b, 15*sizeof(int32_t));
        bool ok = compute(ctx, out);
        ok = ok && ggml_get_i32_1d(out, 15) == -7 && ggml_get_i32_1d(out, 14) == 14;
        printf("  %-50s %s\n", "i32 write of the final element", ok ? "OK" : "FAIL");
        n_failed += !ok;
        ggml_free(ctx);
    }

    // strides of singleton source dimensions are never used, even if they do not fit in int32
    {
        ggml_context * ctx = make_ctx();
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
        fill(a, 0.0f);
        fill(b, 50.0f);
        ggml_tensor * out = ggml_set_1d(ctx, a, b, 6*sizeof(float));
        int32_t * p = (int32_t *) out->op_params;
        p[0] = p[1] = p[2] = -1;
        bool ok = compute(ctx, out);
        const float * o = (const float *) out->data;
        ok = ok && o[6] == 50.0f && o[7] == 51.0f && o[5] == 5.0f;
        printf("  %-50s %s\n", "unused negative strides are ignored", ok ? "OK" : "FAIL");
        n_failed += !ok;
        ggml_free(ctx);
    }

#ifndef _WIN32
    // a write whose last element starts at the end of dst must abort instead of writing past it
    {
        fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0) {
            setenv("GGML_NO_BACKTRACE", "1", 1);
            if (freopen("/dev/null", "w", stderr) == nullptr) { /* keep going with stderr */ }
            ggml_context * ctx = make_ctx();
            ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
            ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
            fill(a, 0.0f);
            fill(b, 50.0f);
            compute(ctx, ggml_set_1d(ctx, a, b, 7*sizeof(float)));
            _exit(0);
        }
        int status = 0;
        const bool ok = pid > 0 && waitpid(pid, &status, 0) == pid && !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        printf("  %-50s %s\n", "write past the destination end aborts", ok ? "OK" : "FAIL");
        n_failed += !ok;
    }
#endif

    printf("%s\n", n_failed == 0 ? "OK" : "FAILED");
    return n_failed == 0 ? 0 : 1;
}
