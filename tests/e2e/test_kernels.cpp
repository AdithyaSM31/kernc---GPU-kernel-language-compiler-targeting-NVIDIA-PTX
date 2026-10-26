#include <gtest/gtest.h>
#include <cuda.h>
#include "kernc/Lexer.h"
#include "kernc/Parser.h"
#include "kernc/Sema.h"
#include "kernc/IRGen.h"
#include "kernc/Passes.h"
#include "kernc/PTXEmitter.h"
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace kernc;

#define CHECK_CUDA(res) \
    ASSERT_EQ((res), CUDA_SUCCESS) << "CUDA call failed with error code: " << (res)

class CudaEnvironment : public ::testing::Environment {
public:
    static CUdevice device;
    static CUcontext context;
    static bool available;

    void SetUp() override {
        CUresult res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            std::cerr << "Warning: cuInit failed (code " << res << "). GPU tests will be skipped.\n";
            available = false;
            return;
        }

        int deviceCount = 0;
        res = cuDeviceGetCount(&deviceCount);
        if (res != CUDA_SUCCESS || deviceCount == 0) {
            std::cerr << "Warning: No CUDA devices found. GPU tests will be skipped.\n";
            available = false;
            return;
        }

        res = cuDeviceGet(&device, 0);
        if (res != CUDA_SUCCESS) {
            available = false;
            return;
        }

        char name[128] = {0};
        cuDeviceGetName(name, sizeof(name), device);
        std::cout << "[CUDA] Initialized device 0: " << name << "\n";

        res = cuCtxCreate(&context, nullptr, 0, device);
        if (res != CUDA_SUCCESS) {
            std::cerr << "Warning: cuCtxCreate failed. GPU tests will be skipped.\n";
            available = false;
            return;
        }

        available = true;
    }

    void TearDown() override {
        if (available && context) {
            cuCtxDestroy(context);
            context = nullptr;
        }
    }
};

CUdevice CudaEnvironment::device = 0;
CUcontext CudaEnvironment::context = nullptr;
bool CudaEnvironment::available = false;

static std::string compileSourceToPTX(const std::string& src, bool optimize = true,
                                      const PTXOptions& ptxOptions = {}) {
    DiagnosticsEngine diag;
    Lexer lexer(src, &diag);
    Parser parser(lexer, diag);
    auto prog = parser.parseProgram();
    EXPECT_NE(prog, nullptr);
    if (!prog) return "";

    Sema sema(diag);
    EXPECT_TRUE(sema.checkProgram(*prog));
    if (diag.hasErrors()) return "";

    IRGen irgen(diag);
    auto irMod = irgen.emitModule(*prog);
    if (optimize) {
        Optimizer::runPasses(irMod);
    }

    PTXEmitter emitter(ptxOptions);
    return emitter.emitModule(irMod);
}

class KernelE2ETest : public ::testing::Test {
public:
    void SetUp() override {
        if (!CudaEnvironment::available) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }

    CUmodule loadPTX(const std::string& ptx) {
        CUmodule mod;
        char errorLog[4096] = {0};
        char infoLog[4096] = {0};
        CUjit_option options[] = {
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        };
        void* optionValues[] = {
            (void*)infoLog,
            (void*)(uintptr_t)sizeof(infoLog),
            (void*)errorLog,
            (void*)(uintptr_t)sizeof(errorLog)
        };

        CUresult res = cuModuleLoadDataEx(&mod, ptx.c_str(), 4, options, optionValues);
        if (res != CUDA_SUCCESS) {
            std::cerr << "JIT Error Log:\n" << errorLog << "\n";
            std::cerr << "JIT Info Log:\n" << infoLog << "\n";
            std::cerr << "PTX was:\n" << ptx << "\n";
        }
        EXPECT_EQ(res, CUDA_SUCCESS);
        return mod;
    }
};

// -------------------------------------------------------------
// VecAdd Tests
// -------------------------------------------------------------
void runVecAddTest(KernelE2ETest* fixture, int N) {
    std::string src = "kernel vecadd(int* a, int* b, int* c, int n) {\n"
                      "    int i = block_idx * block_dim + thread_idx;\n"
                      "    if (i < n) {\n"
                      "        c[i] = a[i] + b[i];\n"
                      "    }\n"
                      "}\n";

    std::string ptx = compileSourceToPTX(src);
    ASSERT_FALSE(ptx.empty());

    CUmodule mod = fixture->loadPTX(ptx);
    ASSERT_NE(mod, nullptr);

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "vecadd"));

    std::vector<int> h_a(N), h_b(N), h_c(N, 0);
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 2 + 1;
        h_b[i] = (i + 5) * 3;
    }

    CUdeviceptr d_a, d_b, d_c;
    size_t sizeBytes = N * sizeof(int);
    CHECK_CUDA(cuMemAlloc(&d_a, sizeBytes));
    CHECK_CUDA(cuMemAlloc(&d_b, sizeBytes));
    CHECK_CUDA(cuMemAlloc(&d_c, sizeBytes));

    CHECK_CUDA(cuMemcpyHtoD(d_a, h_a.data(), sizeBytes));
    CHECK_CUDA(cuMemcpyHtoD(d_b, h_b.data(), sizeBytes));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    void* args[] = { &d_a, &d_b, &d_c, &N };

    CHECK_CUDA(cuLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, 0));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_c.data(), d_c, sizeBytes));

    for (int i = 0; i < N; ++i) {
        int expected = h_a[i] + h_b[i];
        ASSERT_EQ(h_c[i], expected) << "Mismatch at index " << i;
    }

    CHECK_CUDA(cuMemFree(d_a));
    CHECK_CUDA(cuMemFree(d_b));
    CHECK_CUDA(cuMemFree(d_c));
    CHECK_CUDA(cuModuleUnload(mod));
}

TEST_F(KernelE2ETest, VecAdd_ExactBlockMultiple_1024) {
    runVecAddTest(this, 1024);
}

TEST_F(KernelE2ETest, VecAdd_NonBlockMultiple_1000) {
    runVecAddTest(this, 1000);
}

TEST_F(KernelE2ETest, VecAdd_Boundary_1) {
    runVecAddTest(this, 1);
}

// -------------------------------------------------------------
// SAXPY Tests
// -------------------------------------------------------------
void runSaxpyTest(KernelE2ETest* fixture, int N, float a) {
    std::string src = "kernel saxpy(float a, float* x, float* y, int n) {\n"
                      "    int i = block_idx * block_dim + thread_idx;\n"
                      "    if (i < n) {\n"
                      "        y[i] = a * x[i] + y[i];\n"
                      "    }\n"
                      "}\n";

    std::string ptx = compileSourceToPTX(src);
    ASSERT_FALSE(ptx.empty());

    CUmodule mod = fixture->loadPTX(ptx);
    ASSERT_NE(mod, nullptr);

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "saxpy"));

    std::vector<float> h_x(N), h_y(N), h_y_orig(N);
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i) * 0.5f;
        h_y[i] = static_cast<float>(i) * 1.25f - 2.0f;
        h_y_orig[i] = h_y[i];
    }

    CUdeviceptr d_x, d_y;
    size_t sizeBytes = N * sizeof(float);
    CHECK_CUDA(cuMemAlloc(&d_x, sizeBytes));
    CHECK_CUDA(cuMemAlloc(&d_y, sizeBytes));

    CHECK_CUDA(cuMemcpyHtoD(d_x, h_x.data(), sizeBytes));
    CHECK_CUDA(cuMemcpyHtoD(d_y, h_y.data(), sizeBytes));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    void* args[] = { &a, &d_x, &d_y, &N };

    CHECK_CUDA(cuLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, 0));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_y.data(), d_y, sizeBytes));

    for (int i = 0; i < N; ++i) {
        float expected = a * h_x[i] + h_y_orig[i];
        float diff = std::fabs(h_y[i] - expected);
        float scale = std::max({1.0f, std::fabs(h_y[i]), std::fabs(expected)});
        ASSERT_LE(diff / scale, 1e-4f) << "Mismatch at index " << i
                                       << " h_y=" << h_y[i] << " expected=" << expected;
    }

    CHECK_CUDA(cuMemFree(d_x));
    CHECK_CUDA(cuMemFree(d_y));
    CHECK_CUDA(cuModuleUnload(mod));
}

TEST_F(KernelE2ETest, Saxpy_PowerOfTwo_2048) {
    runSaxpyTest(this, 2048, 2.5f);
}

TEST_F(KernelE2ETest, Saxpy_NonBlockMultiple_1000) {
    runSaxpyTest(this, 1000, 3.14159f);
}

TEST_F(KernelE2ETest, Saxpy_NonBlockMultiple_117) {
    runSaxpyTest(this, 117, -1.5f);
}

// -------------------------------------------------------------
// Clamp Kernel Tests (Branching)
// -------------------------------------------------------------
void runClampTest(KernelE2ETest* fixture, int N, float minVal, float maxVal) {
    std::string src = "kernel clamp(float* in, float* out, float min_val, float max_val, int n) {\n"
                      "    int i = block_idx * block_dim + thread_idx;\n"
                      "    if (i < n) {\n"
                      "        float val = in[i];\n"
                      "        if (val < min_val) {\n"
                      "            val = min_val;\n"
                      "        } else {\n"
                      "            if (val > max_val) {\n"
                      "                val = max_val;\n"
                      "            }\n"
                      "        }\n"
                      "        out[i] = val;\n"
                      "    }\n"
                      "}\n";

    std::string ptx = compileSourceToPTX(src);
    ASSERT_FALSE(ptx.empty());

    CUmodule mod = fixture->loadPTX(ptx);
    ASSERT_NE(mod, nullptr);

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "clamp"));

    std::vector<float> h_in(N), h_out(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        // Values spanning well below minVal and above maxVal
        h_in[i] = -10.0f + static_cast<float>(i) * (20.0f / N);
    }

    CUdeviceptr d_in, d_out;
    size_t sizeBytes = N * sizeof(float);
    CHECK_CUDA(cuMemAlloc(&d_in, sizeBytes));
    CHECK_CUDA(cuMemAlloc(&d_out, sizeBytes));

    CHECK_CUDA(cuMemcpyHtoD(d_in, h_in.data(), sizeBytes));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    void* args[] = { &d_in, &d_out, &minVal, &maxVal, &N };

    CHECK_CUDA(cuLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, 0));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, sizeBytes));

    for (int i = 0; i < N; ++i) {
        float expected = std::max(minVal, std::min(maxVal, h_in[i]));
        ASSERT_NEAR(h_out[i], expected, 1e-5f) << "Mismatch at index " << i;
    }

    CHECK_CUDA(cuMemFree(d_in));
    CHECK_CUDA(cuMemFree(d_out));
    CHECK_CUDA(cuModuleUnload(mod));
}

TEST_F(KernelE2ETest, Clamp_NonBlockMultiple_1000) {
    runClampTest(this, 1000, 0.0f, 1.0f);
}

// -------------------------------------------------------------
// ReLU Kernel Tests (Branching)
// -------------------------------------------------------------
void runReLUTest(KernelE2ETest* fixture, int N) {
    std::string src = "kernel relu(float* in, float* out, int n) {\n"
                      "    int i = block_idx * block_dim + thread_idx;\n"
                      "    if (i < n) {\n"
                      "        float val = in[i];\n"
                      "        if (val < 0.0f) {\n"
                      "            val = 0.0f;\n"
                      "        }\n"
                      "        out[i] = val;\n"
                      "    }\n"
                      "}\n";

    std::string ptx = compileSourceToPTX(src);
    ASSERT_FALSE(ptx.empty());

    CUmodule mod = fixture->loadPTX(ptx);
    ASSERT_NE(mod, nullptr);

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "relu"));

    std::vector<float> h_in(N), h_out(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        h_in[i] = -5.0f + static_cast<float>(i) * (10.0f / N);
    }

    CUdeviceptr d_in, d_out;
    size_t sizeBytes = N * sizeof(float);
    CHECK_CUDA(cuMemAlloc(&d_in, sizeBytes));
    CHECK_CUDA(cuMemAlloc(&d_out, sizeBytes));

    CHECK_CUDA(cuMemcpyHtoD(d_in, h_in.data(), sizeBytes));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    void* args[] = { &d_in, &d_out, &N };

    CHECK_CUDA(cuLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, 0));
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, sizeBytes));

    for (int i = 0; i < N; ++i) {
        float expected = (h_in[i] < 0.0f) ? 0.0f : h_in[i];
        ASSERT_NEAR(h_out[i], expected, 1e-5f) << "Mismatch at index " << i;
    }

    CHECK_CUDA(cuMemFree(d_in));
    CHECK_CUDA(cuMemFree(d_out));
    CHECK_CUDA(cuModuleUnload(mod));
}

TEST_F(KernelE2ETest, ReLU_NonBlockMultiple_1000) {
    runReLUTest(this, 1000);
}

TEST_F(KernelE2ETest, ReLU_Small_33) {
    runReLUTest(this, 33);
}

// -------------------------------------------------------------
// Early return
// -------------------------------------------------------------
// Launches more threads than elements; threads past n must return before touching memory.
TEST_F(KernelE2ETest, EarlyReturnGuardSkipsExtraThreads) {
    std::string src = "kernel early_exit(int* in, int* out, int n) {\n"
                      "    int i = block_idx * block_dim + thread_idx;\n"
                      "    if (i >= n) return;\n"
                      "    out[i] = in[i] * 2;\n"
                      "}\n";
    std::string ptx = compileSourceToPTX(src);
    ASSERT_FALSE(ptx.empty());
    CUmodule mod = loadPTX(ptx);
    ASSERT_NE(mod, nullptr);
    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "early_exit"));

    int n = 1000;
    const int blockSize = 256, gridSize = 4, total = blockSize * gridSize; // 1024 threads
    const int sentinel = -7;
    std::vector<int> h_in(total), h_out(total, sentinel);
    for (int k = 0; k < total; ++k) h_in[k] = k;

    CUdeviceptr d_in, d_out;
    CHECK_CUDA(cuMemAlloc(&d_in, total * sizeof(int)));
    CHECK_CUDA(cuMemAlloc(&d_out, total * sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(d_in, h_in.data(), total * sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(d_out, h_out.data(), total * sizeof(int)));
    void* args[] = { &d_in, &d_out, &n };
    CHECK_CUDA(cuLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, 0));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, total * sizeof(int)));

    for (int k = 0; k < total; ++k) {
        int expected = (k < n) ? 2 * k : sentinel;
        ASSERT_EQ(h_out[k], expected) << "index " << k;
    }
    CHECK_CUDA(cuMemFree(d_in));
    CHECK_CUDA(cuMemFree(d_out));
    CHECK_CUDA(cuModuleUnload(mod));
}

// A return inside a loop must leave the loop and skip the statement after it.
TEST_F(KernelE2ETest, ReturnInsideLoopStopsSearch) {
    std::string src = "kernel find_first(int* m, int* out, int rows, int cols, int target) {\n"
                      "    int r = block_idx * block_dim + thread_idx;\n"
                      "    if (r >= rows) return;\n"
                      "    for (int c = 0; c < cols; c = c + 1) {\n"
                      "        if (m[r * cols + c] == target) {\n"
                      "            out[r] = c;\n"
                      "            return;\n"
                      "        }\n"
                      "    }\n"
                      "    out[r] = -2;\n"
                      "}\n";
    std::string ptx = compileSourceToPTX(src);
    ASSERT_FALSE(ptx.empty());
    CUmodule mod = loadPTX(ptx);
    ASSERT_NE(mod, nullptr);
    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "find_first"));

    int rows = 300, cols = 50, target = 7;
    std::vector<int> h_m(rows * cols), h_out(rows, -1);
    for (int k = 0; k < rows * cols; ++k) h_m[k] = (k * 31 + k / cols) % 97;

    CUdeviceptr d_m, d_out;
    CHECK_CUDA(cuMemAlloc(&d_m, h_m.size() * sizeof(int)));
    CHECK_CUDA(cuMemAlloc(&d_out, h_out.size() * sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(d_m, h_m.data(), h_m.size() * sizeof(int)));
    void* args[] = { &d_m, &d_out, &rows, &cols, &target };
    CHECK_CUDA(cuLaunchKernel(func, (rows + 127) / 128, 1, 1, 128, 1, 1, 0, 0, args, 0));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, h_out.size() * sizeof(int)));

    int found = 0;
    for (int r = 0; r < rows; ++r) {
        int expected = -2;
        for (int c = 0; c < cols; ++c) {
            if (h_m[r * cols + c] == target) { expected = c; break; }
        }
        if (expected >= 0) ++found;
        ASSERT_EQ(h_out[r], expected) << "row " << r;
    }
    EXPECT_GT(found, 0);          // the data must exercise both paths
    EXPECT_LT(found, rows);
    CHECK_CUDA(cuMemFree(d_m));
    CHECK_CUDA(cuMemFree(d_out));
    CHECK_CUDA(cuModuleUnload(mod));
}

// -------------------------------------------------------------
// bool values
// -------------------------------------------------------------
// Each thread computes 13 bool-valued expressions and stores them as ints; the CPU reference
// uses C++ semantics, including NaN != NaN being true and cast<bool>(NaN) being true.
static const int kBoolOutputs = 13;

static std::vector<int> runBoolKernel(KernelE2ETest* fixture, bool optimize,
                                      const std::vector<int>& a, const std::vector<float>& x) {
    const char* src =
        "kernel bools(int* a, float* x, int* o, int n) {\n"
        "    int i = block_idx * block_dim + thread_idx;\n"
        "    if (i >= n) return;\n"
        "    int v = a[i];\n"
        "    float fx = x[i];\n"
        "    bool pos = v > 0;\n"
        "    bool even = v % 2 == 0;\n"
        "    int k = i * 13;\n"
        "    o[k] = cast<int>(pos);\n"
        "    o[k + 1] = cast<int>(pos && even);\n"
        "    o[k + 2] = cast<int>(pos || even);\n"
        "    o[k + 3] = cast<int>(!pos);\n"
        "    o[k + 4] = cast<int>(pos == even);\n"
        "    o[k + 5] = cast<int>(pos != even);\n"
        "    o[k + 6] = cast<int>(cast<bool>(v));\n"
        "    o[k + 7] = cast<int>(!v);\n"
        "    o[k + 8] = cast<int>(cast<bool>(fx));\n"
        "    o[k + 9] = cast<int>(fx != fx);\n"
        "    o[k + 10] = cast<int>(cast<float>(true && !false) * 3.0);\n"
        "    o[k + 11] = cast<int>(fx < 1.0 || v == 0);\n"
        "    if (v % 2) { o[k + 12] = 1; } else { o[k + 12] = 0; }\n"
        "}\n";
    int n = static_cast<int>(a.size());
    std::vector<int> out;
    std::string ptx = compileSourceToPTX(src, optimize);
    EXPECT_FALSE(ptx.empty());
    if (ptx.empty()) return out;
    CUmodule mod = fixture->loadPTX(ptx);
    if (!mod) return out;
    CUfunction func;
    EXPECT_EQ(cuModuleGetFunction(&func, mod, "bools"), CUDA_SUCCESS);

    out.assign(n * kBoolOutputs, -1);
    CUdeviceptr d_a, d_x, d_o;
    EXPECT_EQ(cuMemAlloc(&d_a, n * sizeof(int)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemAlloc(&d_x, n * sizeof(float)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemAlloc(&d_o, out.size() * sizeof(int)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemcpyHtoD(d_a, a.data(), n * sizeof(int)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemcpyHtoD(d_x, x.data(), n * sizeof(float)), CUDA_SUCCESS);
    void* args[] = { &d_a, &d_x, &d_o, &n };
    EXPECT_EQ(cuLaunchKernel(func, (n + 127) / 128, 1, 1, 128, 1, 1, 0, 0, args, 0), CUDA_SUCCESS);
    EXPECT_EQ(cuCtxSynchronize(), CUDA_SUCCESS);
    EXPECT_EQ(cuMemcpyDtoH(out.data(), d_o, out.size() * sizeof(int)), CUDA_SUCCESS);
    cuMemFree(d_a);
    cuMemFree(d_x);
    cuMemFree(d_o);
    cuModuleUnload(mod);
    return out;
}

TEST_F(KernelE2ETest, BoolValuesMatchCpu) {
    const int n = 333;
    std::vector<int> a(n);
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (i % 7) - 3; // negatives, zero, odd and even
        switch (i % 5) {
            case 0: x[i] = std::numeric_limits<float>::quiet_NaN(); break;
            case 1: x[i] = 0.0f; break;
            case 2: x[i] = -0.0f; break;
            default: x[i] = static_cast<float>(i) * 0.5f - 40.0f; break;
        }
    }
    std::vector<int> expected(n * kBoolOutputs);
    for (int i = 0; i < n; ++i) {
        int v = a[i];
        float fx = x[i];
        bool pos = v > 0, even = v % 2 == 0;
        int* e = &expected[i * kBoolOutputs];
        e[0] = pos;
        e[1] = pos && even;
        e[2] = pos || even;
        e[3] = !pos;
        e[4] = pos == even;
        e[5] = pos != even;
        e[6] = static_cast<bool>(v);
        e[7] = !v;
        e[8] = static_cast<bool>(fx);
        e[9] = fx != fx;
        e[10] = 3;
        e[11] = fx < 1.0f || v == 0;
        e[12] = (v % 2) != 0; // arithmetic used directly as a condition
    }
    const char* names[kBoolOutputs] = {"pos", "&&", "||", "!pos", "==", "!=", "cast<bool>(int)",
                                       "!int", "cast<bool>(float)", "NaN !=", "cast<float>", "|| mixed",
                                       "if (v % 2)"};
    for (bool optimize : {false, true}) {
        std::vector<int> got = runBoolKernel(this, optimize, a, x);
        ASSERT_EQ(got.size(), expected.size());
        for (size_t k = 0; k < got.size(); ++k) {
            ASSERT_EQ(got[k], expected[k]) << (optimize ? "-O1 " : "-O0 ") << names[k % kBoolOutputs]
                                           << " for element " << k / kBoolOutputs
                                           << " (a=" << a[k / kBoolOutputs]
                                           << ", x=" << x[k / kBoolOutputs] << ")";
        }
    }
}

// -------------------------------------------------------------
// Shared memory
// -------------------------------------------------------------
// Each thread writes tile[t] and reads tile[block_dim - 1 - t], which another warp wrote, so
// the result is only correct if the barrier orders the writes before the reads.
TEST_F(KernelE2ETest, BlockReverseThroughSharedMemory) {
    std::string src = "kernel reverse_block(int* in, int* out) {\n"
                      "    shared int tile[256];\n"
                      "    int t = thread_idx;\n"
                      "    int base = block_idx * block_dim;\n"
                      "    tile[t] = in[base + t];\n"
                      "    barrier();\n"
                      "    out[base + t] = tile[block_dim - 1 - t];\n"
                      "}\n";
    std::string ptx = compileSourceToPTX(src);
    ASSERT_FALSE(ptx.empty());
    CUmodule mod = loadPTX(ptx);
    ASSERT_NE(mod, nullptr);
    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "reverse_block"));

    const int blockSize = 256, gridSize = 8, total = blockSize * gridSize;
    std::vector<int> h_in(total), h_out(total, 0);
    for (int k = 0; k < total; ++k) h_in[k] = k * 3 + 1;
    CUdeviceptr d_in, d_out;
    CHECK_CUDA(cuMemAlloc(&d_in, total * sizeof(int)));
    CHECK_CUDA(cuMemAlloc(&d_out, total * sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(d_in, h_in.data(), total * sizeof(int)));
    void* args[] = { &d_in, &d_out };
    CHECK_CUDA(cuLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, 0));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, total * sizeof(int)));

    for (int b = 0; b < gridSize; ++b) {
        for (int t = 0; t < blockSize; ++t) {
            ASSERT_EQ(h_out[b * blockSize + t], h_in[b * blockSize + blockSize - 1 - t])
                << "block " << b << " thread " << t;
        }
    }
    CHECK_CUDA(cuMemFree(d_in));
    CHECK_CUDA(cuMemFree(d_out));
    CHECK_CUDA(cuModuleUnload(mod));
}

static std::string readFile(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Reduces `data` to one value by relaunching the kernel on the partial sums until one remains.
template <typename T>
static T reduceOnGpu(CUfunction func, const std::vector<T>& data) {
    const int blockSize = 256;
    int n = static_cast<int>(data.size());
    int maxBlocks = (n + blockSize - 1) / blockSize;
    CUdeviceptr d_a, d_b;
    EXPECT_EQ(cuMemAlloc(&d_a, n * sizeof(T)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemAlloc(&d_b, maxBlocks * sizeof(T)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemcpyHtoD(d_a, data.data(), n * sizeof(T)), CUDA_SUCCESS);
    CUdeviceptr in = d_a, out = d_b;
    do {
        int grid = (n + blockSize - 1) / blockSize;
        void* args[] = { &in, &out, &n };
        EXPECT_EQ(cuLaunchKernel(func, grid, 1, 1, blockSize, 1, 1, 0, 0, args, 0), CUDA_SUCCESS);
        std::swap(in, out);
        n = grid;
    } while (n > 1);
    EXPECT_EQ(cuCtxSynchronize(), CUDA_SUCCESS);
    T result{};
    EXPECT_EQ(cuMemcpyDtoH(&result, in, sizeof(T)), CUDA_SUCCESS);
    cuMemFree(d_a);
    cuMemFree(d_b);
    return result;
}

// Runs examples/reduce.kern itself, so the shipped example is what gets tested.
TEST_F(KernelE2ETest, ReductionExampleMatchesCpu) {
    std::string src = readFile(std::string(KERNC_EXAMPLES_DIR) + "/reduce.kern");
    ASSERT_FALSE(src.empty()) << "could not read examples/reduce.kern";
    for (bool optimize : {false, true}) {
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty());
        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr);
        CUfunction fsum, isum;
        CHECK_CUDA(cuModuleGetFunction(&fsum, mod, "reduce_sum"));
        CHECK_CUDA(cuModuleGetFunction(&isum, mod, "reduce_sum_int"));

        for (int n : {1, 255, 256, 257, 1000, (1 << 20) + 3}) {
            std::vector<int> ints(n);
            std::vector<float> floats(n);
            long long exact = 0;
            double ref = 0.0;
            for (int k = 0; k < n; ++k) {
                ints[k] = static_cast<int>((static_cast<unsigned>(k) * 7919u) % 10u) - 3;
                floats[k] = static_cast<float>((k * 2654435761u) % 1000) / 1000.0f;
                exact += ints[k];
                ref += floats[k];
            }
            const char* opt = optimize ? "-O1" : "-O0";
            // Integer addition is associative, so the tree sum is exact.
            EXPECT_EQ(reduceOnGpu(isum, ints), exact) << opt << " int n=" << n;
            // A tree sum adds floats in a different order than a sequential loop. Its error
            // grows with log2(n), far below this bound for n up to 2^20 and inputs in [0, 1).
            float got = reduceOnGpu(fsum, floats);
            EXPECT_LE(std::fabs(got - ref), 1e-5 * std::max(1.0, ref))
                << opt << " float n=" << n << " got " << got << " expected " << ref;
        }
        CHECK_CUDA(cuModuleUnload(mod));
    }
}

// -------------------------------------------------------------
// Device functions
// -------------------------------------------------------------
static int fibRef(int k) { return k < 2 ? k : fibRef(k - 1) + fibRef(k - 2); }
static int collatzRef(int v) {
    int s = 0;
    while (v != 1) { v = (v % 2 == 0) ? v / 2 : 3 * v + 1; ++s; }
    return s;
}

// Runs examples/functions.kern: float, bool, pointer and void functions, recursion and mutual
// recursion, checked against a CPU reference at -O0 and -O1.
TEST_F(KernelE2ETest, DeviceFunctionsExampleMatchesCpu) {
    std::string src = readFile(std::string(KERNC_EXAMPLES_DIR) + "/functions.kern");
    ASSERT_FALSE(src.empty()) << "could not read examples/functions.kern";
    const int n = 300, blockSize = 128, grid = 4; // 512 threads for 300 elements
    std::vector<float> h_x(n);
    for (int k = 0; k < n; ++k) h_x[k] = 0.1f * k - 7.0f;

    for (bool optimize : {false, true}) {
        const char* opt = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty());
        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr);
        CUfunction func;
        CHECK_CUDA(cuModuleGetFunction(&func, mod, "use_functions"));

        std::vector<float> h_y(n + 64, -1.0f);
        std::vector<int> h_z((n + 64) * 4, -1);
        CUdeviceptr d_x, d_y, d_z;
        CHECK_CUDA(cuMemAlloc(&d_x, n * sizeof(float)));
        CHECK_CUDA(cuMemAlloc(&d_y, h_y.size() * sizeof(float)));
        CHECK_CUDA(cuMemAlloc(&d_z, h_z.size() * sizeof(int)));
        CHECK_CUDA(cuMemcpyHtoD(d_x, h_x.data(), n * sizeof(float)));
        CHECK_CUDA(cuMemcpyHtoD(d_y, h_y.data(), h_y.size() * sizeof(float)));
        CHECK_CUDA(cuMemcpyHtoD(d_z, h_z.data(), h_z.size() * sizeof(int)));
        int nArg = n;
        void* args[] = { &d_x, &d_y, &d_z, &nArg };
        CHECK_CUDA(cuLaunchKernel(func, grid, 1, 1, blockSize, 1, 1, 0, 0, args, 0));
        CHECK_CUDA(cuCtxSynchronize());
        CHECK_CUDA(cuMemcpyDtoH(h_y.data(), d_y, h_y.size() * sizeof(float)));
        CHECK_CUDA(cuMemcpyDtoH(h_z.data(), d_z, h_z.size() * sizeof(int)));

        for (int i = 0; i < n; ++i) {
            float want = 2.0f * h_x[i] * h_x[i];
            ASSERT_NEAR(h_y[i], want, 1e-5f * std::max(1.0f, want)) << opt << " y[" << i << "]";
            ASSERT_EQ(h_z[i * 4], fibRef(i % 15)) << opt << " fib, i=" << i;
            ASSERT_EQ(h_z[i * 4 + 1], collatzRef(i + 1)) << opt << " collatz, i=" << i;
            ASSERT_EQ(h_z[i * 4 + 2], (i % 20) % 2 == 0 ? 1 : 0) << opt << " is_even, i=" << i;
            ASSERT_EQ(h_z[i * 4 + 3], (i - 5 >= 0 && i - 5 < 10) ? 1 : 0) << opt << " in_range, i=" << i;
        }
        // Threads past n returned early through in_range(), so nothing past n was written.
        for (int i = n; i < n + 64; ++i) {
            ASSERT_EQ(h_y[i], -1.0f) << opt << " y[" << i << "] written by an out-of-range thread";
        }
        CHECK_CUDA(cuMemFree(d_x));
        CHECK_CUDA(cuMemFree(d_y));
        CHECK_CUDA(cuMemFree(d_z));
        CHECK_CUDA(cuModuleUnload(mod));
    }
}

// User names that match kernc's old generated PTX names (labels, .param names, call-site
// temporaries, shared-array symbols) must still compile, and each call must reach the
// intended function. Values are listed in tests/regress/name_collisions.kern.
TEST_F(KernelE2ETest, UserNamesCannotCollideWithGeneratedNames) {
    std::string src = readFile(std::string(KERNC_REGRESS_DIR) + "/name_collisions.kern");
    ASSERT_FALSE(src.empty()) << "could not read tests/regress/name_collisions.kern";
    for (bool optimize : {false, true}) {
        const char* opt = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty());
        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr) << opt;
        CUfunction k, k_b;
        CHECK_CUDA(cuModuleGetFunction(&k, mod, "k"));
        CHECK_CUDA(cuModuleGetFunction(&k_b, mod, "k_b"));

        std::vector<int> h_o(8, -1);
        CUdeviceptr d_o;
        CHECK_CUDA(cuMemAlloc(&d_o, h_o.size() * sizeof(int)));
        CHECK_CUDA(cuMemcpyHtoD(d_o, h_o.data(), h_o.size() * sizeof(int)));
        void* args[] = { &d_o };
        CHECK_CUDA(cuLaunchKernel(k, 1, 1, 1, 32, 1, 1, 0, 0, args, 0));
        CHECK_CUDA(cuLaunchKernel(k_b, 1, 1, 1, 32, 1, 1, 0, 0, args, 0));
        CHECK_CUDA(cuCtxSynchronize());
        CHECK_CUDA(cuMemcpyDtoH(h_o.data(), d_o, h_o.size() * sizeof(int)));

        const int expected[8] = {2, 6, 11, 1101, 6, 0, 2, 7};
        for (int j = 0; j < 8; ++j) {
            EXPECT_EQ(h_o[j], expected[j]) << opt << " o[" << j << "]";
        }
        CHECK_CUDA(cuMemFree(d_o));
        CHECK_CUDA(cuModuleUnload(mod));
    }
}

// -------------------------------------------------------------
// Line info
// -------------------------------------------------------------
// PTX with .file/.loc must JIT with CU_JIT_GENERATE_LINE_INFO and still compute correctly.
TEST_F(KernelE2ETest, LineInfoPTXJitsAndRuns) {
    std::string src = readFile(std::string(KERNC_EXAMPLES_DIR) + "/saxpy.kern");
    ASSERT_FALSE(src.empty());
    PTXOptions opts;
    opts.lineInfo = true;
    opts.sourceFile = "saxpy.kern";
    std::string ptx = compileSourceToPTX(src, true, opts);
    ASSERT_NE(ptx.find(".loc 1 "), std::string::npos) << ptx;

    CUmodule mod;
    char errorLog[4096] = {0};
    CUjit_option options[] = { CU_JIT_GENERATE_LINE_INFO, CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
    void* values[] = { (void*)(uintptr_t)1, (void*)errorLog, (void*)(uintptr_t)sizeof(errorLog) };
    ASSERT_EQ(cuModuleLoadDataEx(&mod, ptx.c_str(), 3, options, values), CUDA_SUCCESS) << errorLog;
    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, "saxpy"));

    int n = 1000;
    float a = 1.5f;
    std::vector<float> h_x(n), h_y(n), h_y0(n);
    for (int k = 0; k < n; ++k) {
        h_x[k] = 0.25f * k;
        h_y[k] = h_y0[k] = 3.0f - k;
    }
    CUdeviceptr d_x, d_y;
    CHECK_CUDA(cuMemAlloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_y, n * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_x, h_x.data(), n * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_y, h_y.data(), n * sizeof(float)));
    void* args[] = { &a, &d_x, &d_y, &n };
    CHECK_CUDA(cuLaunchKernel(func, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, 0));
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_y.data(), d_y, n * sizeof(float)));
    for (int k = 0; k < n; ++k) {
        ASSERT_NEAR(h_y[k], a * h_x[k] + h_y0[k], 1e-4f * std::max(1.0f, std::fabs(h_y[k])))
            << "index " << k;
    }
    CHECK_CUDA(cuMemFree(d_x));
    CHECK_CUDA(cuMemFree(d_y));
    CHECK_CUDA(cuModuleUnload(mod));
}

// -------------------------------------------------------------
// Differential test: the optimizer must not change observable results
// -------------------------------------------------------------
// Each output is a constant expression, so -O1 folds it inside kernc while -O0 leaves it to
// the GPU. The two runs must agree bit-for-bit.
static std::vector<int32_t> runFoldEdges(KernelE2ETest* fixture, bool optimize) {
    const char* src =
        "kernel fold_edges(int* o, float* f) {\n"
        "    o[0] = 2147483647 + 1;\n"
        "    o[1] = (-2147483647 - 1) - 1;\n"
        "    o[2] = 65536 * 65536;\n"
        "    o[3] = -(-2147483647 - 1);\n"
        "    o[4] = (-2147483647 - 1) / -1;\n"
        "    o[5] = (-2147483647 - 1) % -1;\n"
        "    o[6] = cast<int>(1e20);\n"
        "    o[7] = cast<int>(-1e20);\n"
        "    o[8] = cast<int>(2.9);\n"
        "    o[9] = cast<int>(-2.9);\n"
        "    o[10] = cast<int>(2147483520.0);\n"
        "    o[11] = 7 / -2;\n"
        "    o[12] = -7 % 3;\n"
        "    f[0] = 0.1 + 0.2;\n"
        "    f[1] = 1.0 / 3.0;\n"
        "    f[2] = cast<float>(16777217);\n"
        "    f[3] = -0.0 - 0.0;\n"
        "}\n";
    const int numInts = 13, numFloats = 4;

    std::vector<int32_t> result;
    std::string ptx = compileSourceToPTX(src, optimize);
    EXPECT_FALSE(ptx.empty());
    if (ptx.empty()) return result;

    CUmodule mod = fixture->loadPTX(ptx);
    if (!mod) return result;
    CUfunction func;
    EXPECT_EQ(cuModuleGetFunction(&func, mod, "fold_edges"), CUDA_SUCCESS);

    CUdeviceptr d_o, d_f;
    EXPECT_EQ(cuMemAlloc(&d_o, numInts * sizeof(int32_t)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemAlloc(&d_f, numFloats * sizeof(float)), CUDA_SUCCESS);
    void* args[] = { &d_o, &d_f };
    EXPECT_EQ(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, 0, args, 0), CUDA_SUCCESS);
    EXPECT_EQ(cuCtxSynchronize(), CUDA_SUCCESS);

    // Floats are compared as raw bits so -0.0 vs +0.0 and rounding differences are caught.
    result.resize(numInts + numFloats);
    EXPECT_EQ(cuMemcpyDtoH(result.data(), d_o, numInts * sizeof(int32_t)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemcpyDtoH(result.data() + numInts, d_f, numFloats * sizeof(float)), CUDA_SUCCESS);

    cuMemFree(d_o);
    cuMemFree(d_f);
    cuModuleUnload(mod);
    return result;
}

TEST_F(KernelE2ETest, ConstantFoldingMatchesUnoptimizedGPU) {
    std::vector<int32_t> unopt = runFoldEdges(this, false);
    std::vector<int32_t> opt = runFoldEdges(this, true);
    ASSERT_EQ(unopt.size(), opt.size());
    ASSERT_FALSE(unopt.empty());
    for (size_t k = 0; k < unopt.size(); ++k) {
        EXPECT_EQ(opt[k], unopt[k]) << "output " << k << " differs between -O1 and -O0";
    }
}

static std::vector<int32_t> runStrengthReduceEdges(KernelE2ETest* fixture, bool optimize) {
    const char* src =
        "kernel test_div_mod_strength(int* out, int* in, int n) {\n"
        "    int i = block_idx * block_dim + thread_idx;\n"
        "    if (i < n) {\n"
        "        int v = in[i];\n"
        "        int base = i * 12;\n"
        "        out[base + 0] = v / 2;\n"
        "        out[base + 1] = v / 4;\n"
        "        out[base + 2] = v / 16;\n"
        "        out[base + 3] = v / 128;\n"
        "        out[base + 4] = v / 1024;\n"
        "        out[base + 5] = v / 1;\n"
        "        out[base + 6] = v % 2;\n"
        "        out[base + 7] = v % 4;\n"
        "        out[base + 8] = v % 16;\n"
        "        out[base + 9] = v % 128;\n"
        "        out[base + 10] = v % 1024;\n"
        "        out[base + 11] = v % 1;\n"
        "    }\n"
        "}\n";

    std::vector<int32_t> h_in = {
        0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 7, -7, 15, -15, 16, -16, 17, -17,
        127, -127, 128, -128, 129, -129, 1023, -1023, 1024, -1024,
        2147483647, -2147483647 - 1, -2147483647
    };
    int n = static_cast<int>(h_in.size());
    const int opsPerItem = 12;
    int totalOut = n * opsPerItem;

    std::vector<int32_t> result(totalOut, 0);
    std::string ptx = compileSourceToPTX(src, optimize);
    EXPECT_FALSE(ptx.empty());
    if (ptx.empty()) return result;

    CUmodule mod = fixture->loadPTX(ptx);
    if (!mod) return result;
    CUfunction func;
    EXPECT_EQ(cuModuleGetFunction(&func, mod, "test_div_mod_strength"), CUDA_SUCCESS);

    CUdeviceptr d_in, d_out;
    EXPECT_EQ(cuMemAlloc(&d_in, n * sizeof(int32_t)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemAlloc(&d_out, totalOut * sizeof(int32_t)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemcpyHtoD(d_in, h_in.data(), n * sizeof(int32_t)), CUDA_SUCCESS);

    void* args[] = { &d_out, &d_in, &n };
    EXPECT_EQ(cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, 0, args, 0), CUDA_SUCCESS);
    EXPECT_EQ(cuCtxSynchronize(), CUDA_SUCCESS);

    EXPECT_EQ(cuMemcpyDtoH(result.data(), d_out, totalOut * sizeof(int32_t)), CUDA_SUCCESS);

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    return result;
}

TEST_F(KernelE2ETest, StrengthReducedDivModMatchesUnoptimizedGPU) {
    std::vector<int32_t> unopt = runStrengthReduceEdges(this, false);
    std::vector<int32_t> opt = runStrengthReduceEdges(this, true);
    ASSERT_EQ(unopt.size(), opt.size());
    ASSERT_FALSE(unopt.empty());
    for (size_t k = 0; k < unopt.size(); ++k) {
        EXPECT_EQ(opt[k], unopt[k]) << "output " << k << " differs between -O1 and -O0";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CudaEnvironment);
    return RUN_ALL_TESTS();
}
