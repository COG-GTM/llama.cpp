// Checks that rpc-server rejects malformed tensors in GRAPH_COMPUTE instead of computing on them.
// The server runs in a thread of this process, so a rejected write must leave local memory untouched.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// must match the wire format in ggml/src/ggml-rpc/ggml-rpc.cpp
#pragma pack(push, 1)
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    char padding[4];
};
#pragma pack(pop)

enum {
    CMD_ALLOC_BUFFER    = 0,
    CMD_BUFFER_GET_BASE = 3,
    CMD_SET_TENSOR      = 6,
    CMD_GET_TENSOR      = 8,
    CMD_GRAPH_COMPUTE   = 10,
    CMD_HELLO           = 14,
};

static int g_port = 0;

static bool send_all(int fd, const void * data, size_t size) {
    const char * p = (const char *) data;
    while (size > 0) {
        ssize_t n = send(fd, p, size, MSG_NOSIGNAL);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= (size_t) n;
    }
    return true;
}

static bool recv_all(int fd, void * data, size_t size) {
    char * p = (char *) data;
    while (size > 0) {
        ssize_t n = recv(fd, p, size, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= (size_t) n;
    }
    return true;
}

static bool send_cmd(int fd, uint8_t cmd, const void * payload, uint64_t size) {
    return send_all(fd, &cmd, 1) && send_all(fd, &size, sizeof(size)) && send_all(fd, payload, size);
}

// returns false if the server closed the connection instead of replying
static bool recv_rsp(int fd, void * out, uint64_t size) {
    uint64_t got;
    if (!recv_all(fd, &got, sizeof(got)) || got != size) {
        return false;
    }
    return recv_all(fd, out, size);
}

static int connect_server() {
    for (int attempt = 0; attempt < 100; attempt++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(g_port);
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        if (connect(fd, (sockaddr *) &addr, sizeof(addr)) == 0) {
            uint8_t hello[3];
            if (send_cmd(fd, CMD_HELLO, nullptr, 0) && recv_rsp(fd, hello, sizeof(hello))) {
                return fd;
            }
        }
        close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return -1;
}

struct remote_buffer {
    uint64_t handle;
    uint64_t base;
};

static bool alloc_buffer(int fd, uint64_t size, remote_buffer & buf) {
    #pragma pack(push, 1)
    struct { uint32_t device; uint64_t size; } req = { 0, size };
    #pragma pack(pop)
    uint64_t rsp[2];
    if (!send_cmd(fd, CMD_ALLOC_BUFFER, &req, sizeof(req)) || !recv_rsp(fd, rsp, sizeof(rsp)) || rsp[0] == 0) {
        return false;
    }
    buf.handle = rsp[0];
    return send_cmd(fd, CMD_BUFFER_GET_BASE, &buf.handle, sizeof(buf.handle)) && recv_rsp(fd, &buf.base, sizeof(buf.base));
}

static rpc_tensor make_f32(uint64_t id, uint64_t buffer, uint64_t data, uint32_t n) {
    rpc_tensor t;
    memset(&t, 0, sizeof(t));
    t.id     = id;
    t.type   = GGML_TYPE_F32;
    t.buffer = buffer;
    t.ne[0] = n; t.ne[1] = 1; t.ne[2] = 1; t.ne[3] = 1;
    t.nb[0] = sizeof(float);
    t.nb[1] = t.nb[2] = t.nb[3] = n*sizeof(float);
    t.op   = GGML_OP_NONE;
    t.data = data;
    return t;
}

static bool set_tensor(int fd, const rpc_tensor & t, const float * data, size_t n) {
    std::vector<uint8_t> msg(sizeof(rpc_tensor) + sizeof(uint64_t) + n*sizeof(float), 0);
    memcpy(msg.data(), &t, sizeof(t));
    memcpy(msg.data() + sizeof(t) + sizeof(uint64_t), data, n*sizeof(float));
    return send_cmd(fd, CMD_SET_TENSOR, msg.data(), msg.size());
}

static bool get_tensor(int fd, const rpc_tensor & t, float * out, size_t n) {
    #pragma pack(push, 1)
    struct { rpc_tensor tensor; uint64_t offset; uint64_t size; } req = { t, 0, n*sizeof(float) };
    #pragma pack(pop)
    return send_cmd(fd, CMD_GET_TENSOR, &req, sizeof(req)) && recv_rsp(fd, out, n*sizeof(float));
}

// returns true if the server computed the graph and replied, false if it rejected it and closed the connection
static bool graph_compute(int fd, const std::vector<uint64_t> & nodes, const std::vector<rpc_tensor> & tensors) {
    std::vector<uint8_t> msg;
    auto put = [&](const void * p, size_t sz) { msg.insert(msg.end(), (const uint8_t *) p, (const uint8_t *) p + sz); };
    uint32_t device = 0;
    uint32_t n_nodes = nodes.size();
    uint32_t n_tensors = tensors.size();
    put(&device, sizeof(device));
    put(&n_nodes, sizeof(n_nodes));
    put(nodes.data(), nodes.size()*sizeof(uint64_t));
    put(&n_tensors, sizeof(n_tensors));
    put(tensors.data(), tensors.size()*sizeof(rpc_tensor));
    uint8_t status;
    return send_cmd(fd, CMD_GRAPH_COMPUTE, msg.data(), msg.size()) && recv_rsp(fd, &status, sizeof(status));
}

static const float A[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
static const float B[4] = { 10.0f, 20.0f, 30.0f, 40.0f };

struct session {
    int fd = -1;
    remote_buffer buf = {};
    rpc_tensor a, b;

    bool open() {
        fd = connect_server();
        if (fd < 0 || !alloc_buffer(fd, 4096, buf)) {
            return false;
        }
        a = make_f32(1, buf.handle, buf.base +  0, 4);
        b = make_f32(2, buf.handle, buf.base + 16, 4);
        return set_tensor(fd, a, A, 4) && set_tensor(fd, b, B, 4);
    }

    rpc_tensor add_node(uint64_t buffer, uint64_t data) const {
        rpc_tensor c = make_f32(3, buffer, data, 4);
        c.op = GGML_OP_ADD;
        c.src[0] = 1;
        c.src[1] = 2;
        return c;
    }

    ~session() {
        if (fd >= 0) {
            close(fd);
        }
    }
};

static int n_failed = 0;

static void check(bool ok, const char * name) {
    printf("  %-60s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) {
        n_failed++;
    }
}

// sends one malicious graph on a fresh connection and checks it is rejected without touching local memory
static void expect_rejected(const char * name, float * canary, const std::vector<uint64_t> & nodes,
                            const std::vector<rpc_tensor> & extra, bool add_inputs = true) {
    session s;
    if (!s.open()) {
        check(false, name);
        return;
    }
    std::vector<rpc_tensor> tensors = extra;
    if (add_inputs) {
        tensors.push_back(s.a);
        tensors.push_back(s.b);
    }
    for (auto & t : tensors) {
        if (t.buffer == UINT64_MAX) { t.buffer = s.buf.handle; }
        if (t.data   == UINT64_MAX) { t.data   = s.buf.base + 32; }
    }
    const bool computed = graph_compute(s.fd, nodes, tensors);
    const bool canary_intact = canary == nullptr || (canary[0] == -1.0f && canary[3] == -1.0f);
    check(!computed && canary_intact, name);
}

int main() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        fprintf(stderr, "no CPU device\n");
        return 1;
    }

    {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        socklen_t len = sizeof(addr);
        if (bind(fd, (sockaddr *) &addr, sizeof(addr)) != 0 || getsockname(fd, (sockaddr *) &addr, &len) != 0) {
            fprintf(stderr, "failed to pick a port\n");
            return 1;
        }
        g_port = ntohs(addr.sin_port);
        close(fd);
    }
    const std::string endpoint = "127.0.0.1:" + std::to_string(g_port);
    std::thread([endpoint, cpu]() mutable {
        ggml_backend_rpc_start_server(endpoint.c_str(), nullptr, 1, 1, &cpu);
    }).detach();

    printf("test-rpc-server:\n");

    // valid graph still computes
    {
        session s;
        bool ok = s.open();
        rpc_tensor c = s.add_node(s.buf.handle, s.buf.base + 32);
        float out[4] = {};
        ok = ok && graph_compute(s.fd, { c.id }, { c, s.a, s.b }) && get_tensor(s.fd, c, out, 4);
        ok = ok && out[0] == 11.0f && out[1] == 22.0f && out[2] == 33.0f && out[3] == 44.0f;
        check(ok, "valid ADD graph computes");
    }

    static float canary[4] = { -1.0f, -1.0f, -1.0f, -1.0f };
    const uint64_t canary_addr = (uint64_t) (uintptr_t) canary;

    // output tensor with buffer = 0 and a raw data pointer into this process
    {
        session tmp;
        rpc_tensor c = tmp.add_node(0, canary_addr);
        expect_rejected("buffer=0 with non-zero data is rejected", canary, { c.id }, { c });
    }

    // unknown buffer handle with a raw data pointer
    {
        session tmp;
        rpc_tensor c = tmp.add_node(0xdeadbeef, canary_addr);
        expect_rejected("unknown buffer with non-zero data is rejected", canary, { c.id }, { c });
    }

    // strides whose byte size wraps to 4 bytes so the bounds check would pass
    {
        rpc_tensor c = make_f32(3, UINT64_MAX, UINT64_MAX, 1);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            c.ne[i] = 0x80000001u;
            c.nb[i] = 0x80000000u;
        }
        expect_rejected("wrapping nb is rejected", nullptr, { c.id }, { c }, false);
    }

    // tensor that extends past the end of the buffer
    {
        rpc_tensor c = make_f32(3, UINT64_MAX, UINT64_MAX, 4096);
        expect_rejected("tensor past buffer end is rejected", nullptr, { c.id }, { c }, false);
    }

    // op outside the ggml_op enum
    {
        rpc_tensor c = make_f32(3, UINT64_MAX, UINT64_MAX, 4);
        c.op = GGML_OP_COUNT + 7;
        expect_rejected("op >= GGML_OP_COUNT is rejected", nullptr, { c.id }, { c }, false);
    }

    // custom ops carry a function pointer in op_params
    {
        rpc_tensor c = make_f32(3, UINT64_MAX, UINT64_MAX, 4);
        c.op = GGML_OP_CUSTOM;
        expect_rejected("GGML_OP_CUSTOM is rejected", nullptr, { c.id }, { c }, false);
    }

    // node id that does not resolve to a tensor
    {
        session tmp;
        rpc_tensor c = make_f32(3, UINT64_MAX, UINT64_MAX, 4);
        expect_rejected("node id 0 is rejected", nullptr, { 0 }, { c }, false);
        expect_rejected("unknown node id is rejected", nullptr, { 12345 }, { c }, false);
    }

    // view whose data does not match view_src + view_offs
    {
        rpc_tensor c = make_f32(3, UINT64_MAX, UINT64_MAX, 4);
        c.op = GGML_OP_VIEW;
        c.view_src = 1;
        c.view_offs = 0;
        c.src[0] = 1;
        expect_rejected("view with inconsistent view_offs is rejected", nullptr, { c.id }, { c });
    }

    // server keeps serving valid clients after the rejections
    {
        session s;
        bool ok = s.open();
        rpc_tensor c = s.add_node(s.buf.handle, s.buf.base + 32);
        float out[4] = {};
        ok = ok && graph_compute(s.fd, { c.id }, { c, s.a, s.b }) && get_tensor(s.fd, c, out, 4) && out[3] == 44.0f;
        check(ok, "server still serves valid graphs");
    }

    printf("%s\n", n_failed == 0 ? "OK" : "FAILED");
    fflush(stdout);
    std::_Exit(n_failed == 0 ? 0 : 1);
}
