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
        Optimizer::runInlining(irMod);
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

TEST_F(KernelE2ETest, NestedInliningAndEarlyReturnsMatchCpu) {
    const char* src =
        "func int min_val(int a, int b) {\n"
        "    if (a < b) return a;\n"
        "    return b;\n"
        "}\n"
        "func int max_val(int a, int b) {\n"
        "    if (a > b) return a;\n"
        "    return b;\n"
        "}\n"
        "func int clamp_val(int v, int lo, int hi) {\n"
        "    return min_val(max_val(v, lo), hi);\n"
        "}\n"
        "func int classify(int x, int y) {\n"
        "    if (x > 0 && y > 0) return 1;\n"
        "    if (x < 0 && y > 0) return 2;\n"
        "    if (x < 0 && y < 0) return 3;\n"
        "    if (x > 0 && y < 0) return 4;\n"
        "    return 0;\n"
        "}\n"
        "func void store_clamped_quadrant(int* out, int i, int x, int y) {\n"
        "    int c_x = clamp_val(x, -50, 50);\n"
        "    int c_y = clamp_val(y, -50, 50);\n"
        "    out[i * 2] = c_x + c_y;\n"
        "    out[i * 2 + 1] = classify(c_x, c_y);\n"
        "}\n"
        "kernel test_nested(int* out, int n) {\n"
        "    int i = block_idx * block_dim + thread_idx;\n"
        "    if (i >= n) return;\n"
        "    int x = (i - 100) * 2;\n"
        "    int y = (100 - i) * 3;\n"
        "    store_clamped_quadrant(out, i, x, y);\n"
        "}\n";

    const int n = 200, blockSize = 64, grid = 4;
    for (bool optimize : {false, true}) {
        const char* opt = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty());
        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr);
        CUfunction func;
        CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_nested"));

        std::vector<int> h_out(n * 2, -999);
        CUdeviceptr d_out;
        CHECK_CUDA(cuMemAlloc(&d_out, h_out.size() * sizeof(int)));
        CHECK_CUDA(cuMemcpyHtoD(d_out, h_out.data(), h_out.size() * sizeof(int)));

        int nArg = n;
        void* args[] = { &d_out, &nArg };
        CHECK_CUDA(cuLaunchKernel(func, grid, 1, 1, blockSize, 1, 1, 0, 0, args, 0));
        CHECK_CUDA(cuCtxSynchronize());
        CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, h_out.size() * sizeof(int)));

        for (int i = 0; i < n; ++i) {
            int x = (i - 100) * 2;
            int y = (100 - i) * 3;
            int cx = std::max(-50, std::min(50, x));
            int cy = std::max(-50, std::min(50, y));
            int quad = 0;
            if (cx > 0 && cy > 0) quad = 1;
            else if (cx < 0 && cy > 0) quad = 2;
            else if (cx < 0 && cy < 0) quad = 3;
            else if (cx > 0 && cy < 0) quad = 4;

            ASSERT_EQ(h_out[i * 2], cx + cy) << opt << " sum at i=" << i;
            ASSERT_EQ(h_out[i * 2 + 1], quad) << opt << " quad at i=" << i;
        }

        CHECK_CUDA(cuMemFree(d_out));
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
// Memory ordering under the -O1 instruction scheduler
// -------------------------------------------------------------
// The scheduler may only reorder a load and a store when they provably touch different
// memory. Two pointer parameters can point to the same buffer (as in saxpy(a, y, y, n)), and
// two accesses to one shared array are the same object, so each store below must stay before
// the load that follows it.
TEST_F(KernelE2ETest, SchedulerKeepsStoreBeforeLoadWhenMemoryMayAlias) {
    const char* src =
        "func void store_then_read(int* dst, int* src, int* out, int i) {\n"
        "    dst[i] = 9;\n"
        "    out[i] = src[i];\n"
        "}\n"
        "kernel alias_params(int* a, int* b, int* out) {\n"
        "    int i = thread_idx;\n"
        "    a[i] = 42;\n"
        "    out[i] = b[i];\n"
        "}\n"
        "kernel alias_shared(int* out) {\n"
        "    shared int s[64];\n"
        "    int t = thread_idx;\n"
        "    s[t] = t * 3;\n"
        "    out[t] = s[t];\n"
        "}\n"
        "kernel alias_func(int* a, int* out) {\n"
        "    store_then_read(a, a, out, thread_idx);\n"
        "}\n";
    const int n = 32;
    for (bool optimize : {false, true}) {
        const char* opt = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty());
        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr) << opt;
        CUfunction fParams, fShared, fFunc;
        CHECK_CUDA(cuModuleGetFunction(&fParams, mod, "alias_params"));
        CHECK_CUDA(cuModuleGetFunction(&fShared, mod, "alias_shared"));
        CHECK_CUDA(cuModuleGetFunction(&fFunc, mod, "alias_func"));

        std::vector<int> seven(n, 7), out(n, -1);
        CUdeviceptr d_buf, d_out;
        CHECK_CUDA(cuMemAlloc(&d_buf, n * sizeof(int)));
        CHECK_CUDA(cuMemAlloc(&d_out, n * sizeof(int)));

        // Kernel parameters a and b are the same buffer.
        CHECK_CUDA(cuMemcpyHtoD(d_buf, seven.data(), n * sizeof(int)));
        void* argsParams[] = { &d_buf, &d_buf, &d_out };
        CHECK_CUDA(cuLaunchKernel(fParams, 1, 1, 1, n, 1, 1, 0, 0, argsParams, 0));
        CHECK_CUDA(cuCtxSynchronize());
        CHECK_CUDA(cuMemcpyDtoH(out.data(), d_out, n * sizeof(int)));
        for (int i = 0; i < n; ++i) ASSERT_EQ(out[i], 42) << opt << " alias_params lane " << i;

        // One thread writes and then reads the same shared element.
        void* argsShared[] = { &d_out };
        CHECK_CUDA(cuLaunchKernel(fShared, 1, 1, 1, n, 1, 1, 0, 0, argsShared, 0));
        CHECK_CUDA(cuCtxSynchronize());
        CHECK_CUDA(cuMemcpyDtoH(out.data(), d_out, n * sizeof(int)));
        for (int i = 0; i < n; ++i) ASSERT_EQ(out[i], 3 * i) << opt << " alias_shared lane " << i;

        // Device-function parameters dst and src are the same buffer.
        CHECK_CUDA(cuMemcpyHtoD(d_buf, seven.data(), n * sizeof(int)));
        void* argsFunc[] = { &d_buf, &d_out };
        CHECK_CUDA(cuLaunchKernel(fFunc, 1, 1, 1, n, 1, 1, 0, 0, argsFunc, 0));
        CHECK_CUDA(cuCtxSynchronize());
        CHECK_CUDA(cuMemcpyDtoH(out.data(), d_out, n * sizeof(int)));
        for (int i = 0; i < n; ++i) ASSERT_EQ(out[i], 9) << opt << " alias_func lane " << i;

        CHECK_CUDA(cuMemFree(d_buf));
        CHECK_CUDA(cuMemFree(d_out));
        CHECK_CUDA(cuModuleUnload(mod));
    }
}

// -------------------------------------------------------------
// Vector intrinsic offsets count vectors (so accesses are naturally aligned)
// -------------------------------------------------------------
// load_int4(p, i) reads p[4i .. 4i+3], like reinterpret_cast<int4*>(p)[i] in CUDA. An offset
// counted in scalars made load_int4(p, 1) a 16-byte access at a 4-byte boundary, which faults
// with CUDA_ERROR_MISALIGNED_ADDRESS. Shared scalar arrays must be 16-byte aligned too.
TEST_F(KernelE2ETest, VectorIntrinsicOffsetsCountVectors) {
    const char* src =
        "kernel vec_offsets(int* p, float* f, int* o) {\n"
        "    int4 q = load_int4(p, 1);\n"
        "    int2 r = load_int2(p, 3);\n"
        "    o[0] = q.x; o[1] = q.y; o[2] = q.z; o[3] = q.w;\n"
        "    o[4] = r.x; o[5] = r.y;\n"
        "    store_float4(f, 1, float4(1.0, 2.0, 3.0, 4.0));\n"
        "}\n"
        "kernel vec_shared(int* o) {\n"
        "    shared int pad[1];\n"
        "    shared int s[16];\n"
        "    pad[0] = 0;\n"
        "    s[thread_idx] = thread_idx * 10;\n"
        "    barrier();\n"
        "    if (thread_idx == 0) {\n"
        "        int4 q = load_int4(s, 1);\n"
        "        o[6] = q.x; o[7] = q.y; o[8] = q.z; o[9] = q.w;\n"
        "    }\n"
        "}\n";
    for (bool optimize : {false, true}) {
        const char* opt = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty());
        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr) << opt;
        CUfunction fGlobal, fShared;
        CHECK_CUDA(cuModuleGetFunction(&fGlobal, mod, "vec_offsets"));
        CHECK_CUDA(cuModuleGetFunction(&fShared, mod, "vec_shared"));

        std::vector<int> h_p(16), h_o(10, -1);
        for (int k = 0; k < 16; ++k) h_p[k] = k;
        std::vector<float> h_f(12, -1.0f);
        CUdeviceptr d_p, d_f, d_o;
        CHECK_CUDA(cuMemAlloc(&d_p, h_p.size() * sizeof(int)));
        CHECK_CUDA(cuMemAlloc(&d_f, h_f.size() * sizeof(float)));
        CHECK_CUDA(cuMemAlloc(&d_o, h_o.size() * sizeof(int)));
        CHECK_CUDA(cuMemcpyHtoD(d_p, h_p.data(), h_p.size() * sizeof(int)));
        CHECK_CUDA(cuMemcpyHtoD(d_f, h_f.data(), h_f.size() * sizeof(float)));
        CHECK_CUDA(cuMemcpyHtoD(d_o, h_o.data(), h_o.size() * sizeof(int)));

        void* argsGlobal[] = { &d_p, &d_f, &d_o };
        CHECK_CUDA(cuLaunchKernel(fGlobal, 1, 1, 1, 1, 1, 1, 0, 0, argsGlobal, 0));
        CHECK_CUDA(cuCtxSynchronize());
        void* argsShared[] = { &d_o };
        CHECK_CUDA(cuLaunchKernel(fShared, 1, 1, 1, 16, 1, 1, 0, 0, argsShared, 0));
        CHECK_CUDA(cuCtxSynchronize());
        CHECK_CUDA(cuMemcpyDtoH(h_o.data(), d_o, h_o.size() * sizeof(int)));
        CHECK_CUDA(cuMemcpyDtoH(h_f.data(), d_f, h_f.size() * sizeof(float)));

        const int wantO[10] = {4, 5, 6, 7, 6, 7, 40, 50, 60, 70};
        for (int k = 0; k < 10; ++k) EXPECT_EQ(h_o[k], wantO[k]) << opt << " o[" << k << "]";
        const float wantF[12] = {-1, -1, -1, -1, 1, 2, 3, 4, -1, -1, -1, -1};
        for (int k = 0; k < 12; ++k) EXPECT_EQ(h_f[k], wantF[k]) << opt << " f[" << k << "]";

        CHECK_CUDA(cuMemFree(d_p));
        CHECK_CUDA(cuMemFree(d_f));
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
        // PTX clamps shift counts >= 32 (unsigned, so negative counts too) to 32.
        "    o[13] = 1 << 32;\n"
        "    o[14] = 1000 >> 40;\n"
        "    o[15] = 1 << -1;\n"
        "    o[16] = -8 >> 40;\n"
        "    o[17] = 1 << 31;\n"
        "    o[18] = -8 >> 1;\n"
        "    o[19] = 7 << 33;\n"
        "    f[0] = 0.1 + 0.2;\n"
        "    f[1] = 1.0 / 3.0;\n"
        "    f[2] = cast<float>(16777217);\n"
        "    f[3] = -0.0 - 0.0;\n"
        "}\n";
    const int numInts = 20, numFloats = 4;

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

static std::vector<int32_t> runNonPowerOfTwoEdges(KernelE2ETest* fixture, bool optimize) {
    const char* src =
        "kernel test_non_power_of_two_div_mod(int* out, int* in, int n) {\n"
        "    int i = block_idx * block_dim + thread_idx;\n"
        "    if (i < n) {\n"
        "        int v = in[i];\n"
        "        int base = i * 24;\n"
        "        out[base + 0] = v / 3;\n"
        "        out[base + 1] = v / 5;\n"
        "        out[base + 2] = v / 7;\n"
        "        out[base + 3] = v / 10;\n"
        "        out[base + 4] = v / 25;\n"
        "        out[base + 5] = v / 1000;\n"
        "        out[base + 6] = v / -3;\n"
        "        out[base + 7] = v / -5;\n"
        "        out[base + 8] = v / -7;\n"
        "        out[base + 9] = v / -10;\n"
        "        out[base + 10] = v / -25;\n"
        "        out[base + 11] = v / -1000;\n"
        "        out[base + 12] = v % 3;\n"
        "        out[base + 13] = v % 5;\n"
        "        out[base + 14] = v % 7;\n"
        "        out[base + 15] = v % 10;\n"
        "        out[base + 16] = v % 25;\n"
        "        out[base + 17] = v % 1000;\n"
        "        out[base + 18] = v % -3;\n"
        "        out[base + 19] = v % -5;\n"
        "        out[base + 20] = v % -7;\n"
        "        out[base + 21] = v % -10;\n"
        "        out[base + 22] = v % -25;\n"
        "        out[base + 23] = v % -1000;\n"
        "    }\n"
        "}\n";

    std::vector<int32_t> h_in = {
        0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 7, -7, 10, -10, 15, -15, 24, -24, 25, -25,
        26, -26, 99, -99, 100, -100, 101, -101, 999, -999, 1000, -1000, 1001, -1001,
        1234567, -1234567, 2147483647, -2147483647 - 1, -2147483647, 1073741823, -1073741824
    };
    int n = static_cast<int>(h_in.size());
    const int opsPerItem = 24;
    int totalOut = n * opsPerItem;

    std::vector<int32_t> result(totalOut, 0);
    std::string ptx = compileSourceToPTX(src, optimize);
    EXPECT_FALSE(ptx.empty());
    if (ptx.empty()) return result;

    if (optimize) {
        EXPECT_EQ(ptx.find("div.s32"), std::string::npos) << "Expected div.s32 to be eliminated at -O1:\n" << ptx;
        EXPECT_EQ(ptx.find("rem.s32"), std::string::npos) << "Expected rem.s32 to be eliminated at -O1:\n" << ptx;
        EXPECT_NE(ptx.find("mul.hi.s32"), std::string::npos) << "Expected mul.hi.s32 to be emitted at -O1:\n" << ptx;
    }

    CUmodule mod = fixture->loadPTX(ptx);
    if (!mod) return result;
    CUfunction func;
    EXPECT_EQ(cuModuleGetFunction(&func, mod, "test_non_power_of_two_div_mod"), CUDA_SUCCESS);

    CUdeviceptr d_in, d_out;
    EXPECT_EQ(cuMemAlloc(&d_in, n * sizeof(int32_t)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemAlloc(&d_out, totalOut * sizeof(int32_t)), CUDA_SUCCESS);
    EXPECT_EQ(cuMemcpyHtoD(d_in, h_in.data(), n * sizeof(int32_t)), CUDA_SUCCESS);

    void* args[] = { &d_out, &d_in, &n };
    EXPECT_EQ(cuLaunchKernel(func, 1, 1, 1, 64, 1, 1, 0, 0, args, 0), CUDA_SUCCESS);
    EXPECT_EQ(cuCtxSynchronize(), CUDA_SUCCESS);

    EXPECT_EQ(cuMemcpyDtoH(result.data(), d_out, totalOut * sizeof(int32_t)), CUDA_SUCCESS);

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    return result;
}

TEST_F(KernelE2ETest, NonPowerOfTwoDivModMatchesUnoptimizedGPUAndCpu) {
    std::vector<int32_t> h_in = {
        0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 7, -7, 10, -10, 15, -15, 24, -24, 25, -25,
        26, -26, 99, -99, 100, -100, 101, -101, 999, -999, 1000, -1000, 1001, -1001,
        1234567, -1234567, 2147483647, -2147483647 - 1, -2147483647, 1073741823, -1073741824
    };
    int const divisors[12] = { 3, 5, 7, 10, 25, 1000, -3, -5, -7, -10, -25, -1000 };

    std::vector<int32_t> unopt = runNonPowerOfTwoEdges(this, false);
    std::vector<int32_t> opt = runNonPowerOfTwoEdges(this, true);
    ASSERT_EQ(unopt.size(), opt.size());
    ASSERT_FALSE(unopt.empty());

    for (size_t i = 0; i < h_in.size(); ++i) {
        int32_t v = h_in[i];
        size_t base = i * 24;
        for (int dIdx = 0; dIdx < 12; ++dIdx) {
            int32_t d = divisors[dIdx];
            int32_t expected_div = v / d;
            int32_t expected_mod = v % d;

            int32_t actual_unopt_div = unopt[base + dIdx];
            int32_t actual_opt_div = opt[base + dIdx];
            EXPECT_EQ(actual_unopt_div, expected_div) << "Unopt div mismatch at v=" << v << ", d=" << d;
            EXPECT_EQ(actual_opt_div, expected_div) << "Opt div mismatch at v=" << v << ", d=" << d;
            EXPECT_EQ(actual_opt_div, actual_unopt_div) << "Div diff between opt and unopt at v=" << v << ", d=" << d;

            int32_t actual_unopt_mod = unopt[base + 12 + dIdx];
            int32_t actual_opt_mod = opt[base + 12 + dIdx];
            EXPECT_EQ(actual_unopt_mod, expected_mod) << "Unopt mod mismatch at v=" << v << ", d=" << d;
            EXPECT_EQ(actual_opt_mod, expected_mod) << "Opt mod mismatch at v=" << v << ", d=" << d;
            EXPECT_EQ(actual_opt_mod, actual_unopt_mod) << "Mod diff between opt and unopt at v=" << v << ", d=" << d;
        }
    }
}

TEST_F(KernelE2ETest, BitwiseOperatorsMatchCpu) {
    const char* src =
        "kernel test_bitwise(int* out, int* a, int* b, int n) {\n"
        "    int i = block_idx * block_dim + thread_idx;\n"
        "    if (i < n) {\n"
        "        int x = a[i];\n"
        "        int y = b[i];\n"
        "        int base = i * 8;\n"
        "        out[base + 0] = x & y;\n"
        "        out[base + 1] = x | y;\n"
        "        out[base + 2] = x ^ y;\n"
        "        out[base + 3] = ~x;\n"
        "        out[base + 4] = x << (y % 31);\n"
        "        out[base + 5] = x >> (y % 31);\n"
        "        out[base + 6] = (x & y) ^ (x | y);\n"
        "        out[base + 7] = ~(x ^ y);\n"
        "    }\n"
        "}\n";

    std::vector<int32_t> h_a = {
        0, 1, -1, 2, -2, 7, -7, 15, -15, 127, -127, 255, -255, 1024, -1024,
        2147483647, -2147483647 - 1, 0x12345678, -0x12345678
    };
    std::vector<int32_t> h_b = {
        0, 1, 2, 3, 5, 7, 11, 1, 2, 3, 4, 7, 8, 12, 16,
        24, 30, 13, 27
    };
    ASSERT_EQ(h_a.size(), h_b.size());
    int n = static_cast<int>(h_a.size());
    const int opsPerItem = 8;
    int totalOut = n * opsPerItem;

    std::vector<int32_t> expected(totalOut, 0);
    for (int i = 0; i < n; ++i) {
        int32_t x = h_a[i];
        int32_t y = h_b[i];
        int shift = y % 31;
        int base = i * opsPerItem;
        expected[base + 0] = x & y;
        expected[base + 1] = x | y;
        expected[base + 2] = x ^ y;
        expected[base + 3] = ~x;
        expected[base + 4] = static_cast<int32_t>(static_cast<uint32_t>(x) << (shift & 31));
        expected[base + 5] = x >> (shift & 31);
        expected[base + 6] = (x & y) ^ (x | y);
        expected[base + 7] = ~(x ^ y);
    }

    for (bool optimize : {false, true}) {
        const char* opt = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty()) << opt;
        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr) << opt;
        CUfunction func;
        CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_bitwise"));

        std::vector<int32_t> h_out(totalOut, 0);
        CUdeviceptr d_out, d_a, d_b;
        CHECK_CUDA(cuMemAlloc(&d_out, totalOut * sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_a, n * sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_b, n * sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_a, h_a.data(), n * sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_b, h_b.data(), n * sizeof(int32_t)));

        void* args[] = { &d_out, &d_a, &d_b, &n };
        CHECK_CUDA(cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, 0, args, 0));
        CHECK_CUDA(cuCtxSynchronize());
        CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, totalOut * sizeof(int32_t)));

        for (int k = 0; k < totalOut; ++k) {
            EXPECT_EQ(h_out[k], expected[k]) << opt << " item " << k;
        }

        cuMemFree(d_out);
        cuMemFree(d_a);
        cuMemFree(d_b);
        cuModuleUnload(mod);
    }
}

TEST_F(KernelE2ETest, MultiDimensionalExecutionMatchesCpu) {
    // 1. 2D Coordinate & Dimension Test
    std::string src2D =
        "kernel test_dims_2d(int* out, int width, int height) {\n"
        "    int x = block_idx_x * block_dim_x + thread_idx_x;\n"
        "    int y = block_idx_y * block_dim_y + thread_idx_y;\n"
        "    if (x < width && y < height) {\n"
        "        out[y * width + x] = (x * 1000) + y + (grid_dim_x * 100) + grid_dim_y;\n"
        "    }\n"
        "}\n";

    // 2. 3D Coordinate & Dimension Test
    std::string src3D =
        "kernel test_dims_3d(int* out, int dim_x, int dim_y, int dim_z) {\n"
        "    int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    int y = block_idx.y * block_dim.y + thread_idx.y;\n"
        "    int z = block_idx_z * block_dim_z + thread_idx_z;\n"
        "    if (x < dim_x && y < dim_y && z < dim_z) {\n"
        "        int idx = (z * dim_y + y) * dim_x + x;\n"
        "        out[idx] = (z * 10000) + (y * 100) + x + (grid_dim_z * 100000);\n"
        "    }\n"
        "}\n";

    for (bool optimize : {false, true}) {
        const char* opt = optimize ? "-O1" : "-O0";

        // Test 2D Kernel
        {
            int width = 30;  // 4 blocks of 8 = 32 (with boundary clipping)
            int height = 45; // 3 blocks of 16 = 48 (with boundary clipping)
            int total = width * height;
            unsigned int gx = 4, gy = 3, gz = 1;
            unsigned int bx = 8, by = 16, bz = 1;

            std::vector<int32_t> expected(total);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    expected[y * width + x] = (x * 1000) + y + (static_cast<int>(gx) * 100) + static_cast<int>(gy);
                }
            }

            std::string ptx = compileSourceToPTX(src2D, optimize);
            ASSERT_FALSE(ptx.empty()) << opt;
            CUmodule mod = loadPTX(ptx);
            ASSERT_NE(mod, nullptr) << opt;
            CUfunction func;
            CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_dims_2d"));

            std::vector<int32_t> h_out(total, 0);
            CUdeviceptr d_out;
            CHECK_CUDA(cuMemAlloc(&d_out, total * sizeof(int32_t)));

            void* args[] = { &d_out, &width, &height };
            CHECK_CUDA(cuLaunchKernel(func, gx, gy, gz, bx, by, bz, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());
            CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, total * sizeof(int32_t)));

            for (int i = 0; i < total; ++i) {
                EXPECT_EQ(h_out[i], expected[i]) << opt << " 2D index " << i;
            }

            cuMemFree(d_out);
            cuModuleUnload(mod);
        }

        // Test 3D Kernel
        {
            int dx = 7; // 2 blocks of 4 = 8
            int dy = 5; // 3 blocks of 2 = 6
            int dz = 7; // 4 blocks of 2 = 8
            int total = dx * dy * dz;
            unsigned int gx = 2, gy = 3, gz = 4;
            unsigned int bx = 4, by = 2, bz = 2;

            std::vector<int32_t> expected(total);
            for (int z = 0; z < dz; ++z) {
                for (int y = 0; y < dy; ++y) {
                    for (int x = 0; x < dx; ++x) {
                        int idx = (z * dy + y) * dx + x;
                        expected[idx] = (z * 10000) + (y * 100) + x + (static_cast<int>(gz) * 100000);
                    }
                }
            }

            std::string ptx = compileSourceToPTX(src3D, optimize);
            ASSERT_FALSE(ptx.empty()) << opt;
            CUmodule mod = loadPTX(ptx);
            ASSERT_NE(mod, nullptr) << opt;
            CUfunction func;
            CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_dims_3d"));

            std::vector<int32_t> h_out(total, 0);
            CUdeviceptr d_out;
            CHECK_CUDA(cuMemAlloc(&d_out, total * sizeof(int32_t)));

            void* args[] = { &d_out, &dx, &dy, &dz };
            CHECK_CUDA(cuLaunchKernel(func, gx, gy, gz, bx, by, bz, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());
            CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, total * sizeof(int32_t)));

            for (int i = 0; i < total; ++i) {
                EXPECT_EQ(h_out[i], expected[i]) << opt << " 3D index " << i;
            }

            cuMemFree(d_out);
            cuModuleUnload(mod);
        }
    }
}

TEST_F(KernelE2ETest, WarpIntrinsicsMatchCpu) {
    std::string src = 
        "kernel test_warp_intrinsics(\n"
        "    int* in_i, float* in_f,\n"
        "    int* out_reduce_i, float* out_reduce_f,\n"
        "    int* out_xor_i, float* out_xor_f,\n"
        "    int* out_bcast_i, float* out_bcast_f,\n"
        "    int* out_ballot_i, int* out_ballot_int_i,\n"
        "    int* out_activemask_i, int* out_activemask2_i,\n"
        "    int* out_up_i, float* out_up_f\n"
        ") {\n"
        "    int tid = thread_idx;\n"
        "    int mask = __activemask();\n"
        "    int mask2 = activemask();\n"
        "    out_activemask_i[tid] = mask;\n"
        "    out_activemask2_i[tid] = mask2;\n"
        "\n"
        "    // Int tree reduction (__shfl_down_sync)\n"
        "    int vali = in_i[tid];\n"
        "    vali = vali + __shfl_down_sync(mask, vali, 16);\n"
        "    vali = vali + __shfl_down_sync(mask, vali, 8);\n"
        "    vali = vali + __shfl_down_sync(mask, vali, 4);\n"
        "    vali = vali + __shfl_down_sync(mask, vali, 2);\n"
        "    vali = vali + __shfl_down_sync(mask, vali, 1);\n"
        "    out_reduce_i[tid] = vali;\n"
        "\n"
        "    // Float tree reduction (shfl_down_sync, unprefixed)\n"
        "    float valf = in_f[tid];\n"
        "    valf = valf + shfl_down_sync(mask, valf, 16);\n"
        "    valf = valf + shfl_down_sync(mask, valf, 8);\n"
        "    valf = valf + shfl_down_sync(mask, valf, 4);\n"
        "    valf = valf + shfl_down_sync(mask, valf, 2);\n"
        "    valf = valf + shfl_down_sync(mask, valf, 1);\n"
        "    out_reduce_f[tid] = valf;\n"
        "\n"
        "    // Butterfly exchange (XOR) with delta 16 (int and float)\n"
        "    out_xor_i[tid] = __shfl_xor_sync(mask, in_i[tid], 16);\n"
        "    out_xor_f[tid] = shfl_xor_sync(mask2, in_f[tid], 16);\n"
        "\n"
        "    // Broadcast from lane 7 (int and float)\n"
        "    out_bcast_i[tid] = __shfl_sync(mask, in_i[tid], 7);\n"
        "    out_bcast_f[tid] = shfl_sync(mask2, in_f[tid], 7);\n"
        "\n"
        "    // Ballot: bool predicate (tid >= 16) and int predicate (in_i[tid] % 2)\n"
        "    out_ballot_i[tid] = __ballot_sync(mask, tid >= 16);\n"
        "    out_ballot_int_i[tid] = ballot_sync(mask2, in_i[tid] % 2);\n"
        "\n"
        "    // Shift up with delta 3 (int and float)\n"
        "    out_up_i[tid] = __shfl_up_sync(mask, in_i[tid], 3);\n"
        "    out_up_f[tid] = shfl_up_sync(mask2, in_f[tid], 3);\n"
        "}\n";

    constexpr int kWarpSize = 32;
    std::vector<int32_t> h_in_i(kWarpSize);
    std::vector<float> h_in_f(kWarpSize);
    int32_t expected_sum_i = 0;
    float expected_sum_f = 0.0f;

    for (int i = 0; i < kWarpSize; ++i) {
        h_in_i[i] = i + 1;
        h_in_f[i] = (i + 1) * 0.5f;
        expected_sum_i += h_in_i[i];
        expected_sum_f += h_in_f[i];
    }

    for (bool optimize : {false, true}) {
        std::string optName = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty()) << optName;

        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr) << optName;

        CUfunction func;
        CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_warp_intrinsics"));

        CUdeviceptr d_in_i, d_in_f;
        CUdeviceptr d_out_reduce_i, d_out_reduce_f;
        CUdeviceptr d_out_xor_i, d_out_xor_f;
        CUdeviceptr d_out_bcast_i, d_out_bcast_f;
        CUdeviceptr d_out_ballot_i, d_out_ballot_int_i;
        CUdeviceptr d_out_activemask_i, d_out_activemask2_i;
        CUdeviceptr d_out_up_i, d_out_up_f;

        size_t sizeBytesI = kWarpSize * sizeof(int32_t);
        size_t sizeBytesF = kWarpSize * sizeof(float);

        CHECK_CUDA(cuMemAlloc(&d_in_i, sizeBytesI));
        CHECK_CUDA(cuMemAlloc(&d_in_f, sizeBytesF));
        CHECK_CUDA(cuMemAlloc(&d_out_reduce_i, sizeBytesI));
        CHECK_CUDA(cuMemAlloc(&d_out_reduce_f, sizeBytesF));
        CHECK_CUDA(cuMemAlloc(&d_out_xor_i, sizeBytesI));
        CHECK_CUDA(cuMemAlloc(&d_out_xor_f, sizeBytesF));
        CHECK_CUDA(cuMemAlloc(&d_out_bcast_i, sizeBytesI));
        CHECK_CUDA(cuMemAlloc(&d_out_bcast_f, sizeBytesF));
        CHECK_CUDA(cuMemAlloc(&d_out_ballot_i, sizeBytesI));
        CHECK_CUDA(cuMemAlloc(&d_out_ballot_int_i, sizeBytesI));
        CHECK_CUDA(cuMemAlloc(&d_out_activemask_i, sizeBytesI));
        CHECK_CUDA(cuMemAlloc(&d_out_activemask2_i, sizeBytesI));
        CHECK_CUDA(cuMemAlloc(&d_out_up_i, sizeBytesI));
        CHECK_CUDA(cuMemAlloc(&d_out_up_f, sizeBytesF));

        CHECK_CUDA(cuMemcpyHtoD(d_in_i, h_in_i.data(), sizeBytesI));
        CHECK_CUDA(cuMemcpyHtoD(d_in_f, h_in_f.data(), sizeBytesF));

        void* args[] = {
            &d_in_i, &d_in_f,
            &d_out_reduce_i, &d_out_reduce_f,
            &d_out_xor_i, &d_out_xor_f,
            &d_out_bcast_i, &d_out_bcast_f,
            &d_out_ballot_i, &d_out_ballot_int_i,
            &d_out_activemask_i, &d_out_activemask2_i,
            &d_out_up_i, &d_out_up_f
        };

        CHECK_CUDA(cuLaunchKernel(func, 1, 1, 1, kWarpSize, 1, 1, 0, 0, args, 0));
        CHECK_CUDA(cuCtxSynchronize());

        std::vector<int32_t> h_out_reduce_i(kWarpSize, 0);
        std::vector<float> h_out_reduce_f(kWarpSize, 0.0f);
        std::vector<int32_t> h_out_xor_i(kWarpSize, 0);
        std::vector<float> h_out_xor_f(kWarpSize, 0.0f);
        std::vector<int32_t> h_out_bcast_i(kWarpSize, 0);
        std::vector<float> h_out_bcast_f(kWarpSize, 0.0f);
        std::vector<int32_t> h_out_ballot_i(kWarpSize, 0);
        std::vector<int32_t> h_out_ballot_int_i(kWarpSize, 0);
        std::vector<int32_t> h_out_activemask_i(kWarpSize, 0);
        std::vector<int32_t> h_out_activemask2_i(kWarpSize, 0);
        std::vector<int32_t> h_out_up_i(kWarpSize, 0);
        std::vector<float> h_out_up_f(kWarpSize, 0.0f);

        CHECK_CUDA(cuMemcpyDtoH(h_out_reduce_i.data(), d_out_reduce_i, sizeBytesI));
        CHECK_CUDA(cuMemcpyDtoH(h_out_reduce_f.data(), d_out_reduce_f, sizeBytesF));
        CHECK_CUDA(cuMemcpyDtoH(h_out_xor_i.data(), d_out_xor_i, sizeBytesI));
        CHECK_CUDA(cuMemcpyDtoH(h_out_xor_f.data(), d_out_xor_f, sizeBytesF));
        CHECK_CUDA(cuMemcpyDtoH(h_out_bcast_i.data(), d_out_bcast_i, sizeBytesI));
        CHECK_CUDA(cuMemcpyDtoH(h_out_bcast_f.data(), d_out_bcast_f, sizeBytesF));
        CHECK_CUDA(cuMemcpyDtoH(h_out_ballot_i.data(), d_out_ballot_i, sizeBytesI));
        CHECK_CUDA(cuMemcpyDtoH(h_out_ballot_int_i.data(), d_out_ballot_int_i, sizeBytesI));
        CHECK_CUDA(cuMemcpyDtoH(h_out_activemask_i.data(), d_out_activemask_i, sizeBytesI));
        CHECK_CUDA(cuMemcpyDtoH(h_out_activemask2_i.data(), d_out_activemask2_i, sizeBytesI));
        CHECK_CUDA(cuMemcpyDtoH(h_out_up_i.data(), d_out_up_i, sizeBytesI));
        CHECK_CUDA(cuMemcpyDtoH(h_out_up_f.data(), d_out_up_f, sizeBytesF));

        // 1. activemask (__ and unprefixed): all 32 lanes active in warp -> 0xFFFFFFFF (-1)
        for (int i = 0; i < kWarpSize; ++i) {
            EXPECT_EQ(h_out_activemask_i[i], -1) << optName << " activemask lane " << i;
            EXPECT_EQ(h_out_activemask2_i[i], -1) << optName << " activemask2 lane " << i;
        }

        // 2. Tree reduction via shfl_down_sync: lane 0 must contain full warp sum
        EXPECT_EQ(h_out_reduce_i[0], expected_sum_i) << optName << " int reduction lane 0";
        EXPECT_FLOAT_EQ(h_out_reduce_f[0], expected_sum_f) << optName << " float reduction lane 0";

        // 3. Butterfly exchange via shfl_xor_sync: lane i gets lane (i ^ 16)
        for (int i = 0; i < kWarpSize; ++i) {
            EXPECT_EQ(h_out_xor_i[i], h_in_i[i ^ 16]) << optName << " xor int lane " << i;
            EXPECT_FLOAT_EQ(h_out_xor_f[i], h_in_f[i ^ 16]) << optName << " xor float lane " << i;
        }

        // 4. Broadcast via shfl_sync: all lanes get lane 7
        for (int i = 0; i < kWarpSize; ++i) {
            EXPECT_EQ(h_out_bcast_i[i], h_in_i[7]) << optName << " broadcast int lane " << i;
            EXPECT_FLOAT_EQ(h_out_bcast_f[i], h_in_f[7]) << optName << " broadcast float lane " << i;
        }

        // 5. Ballot via ballot_sync:
        //    - Bool predicate (tid >= 16): bits 16..31 are 1 -> 0xFFFF0000
        //    - Int predicate (in_i[tid] % 2): alternating bits 0, 2, 4, ... are 1 -> 0x55555555
        for (int i = 0; i < kWarpSize; ++i) {
            EXPECT_EQ(h_out_ballot_i[i], static_cast<int32_t>(0xFFFF0000)) << optName << " ballot bool lane " << i;
            EXPECT_EQ(h_out_ballot_int_i[i], static_cast<int32_t>(0x55555555)) << optName << " ballot int lane " << i;
        }

        // 6. Shift up via shfl_up_sync: lane i gets lane i - 3 if i >= 3, else lane i
        for (int i = 0; i < kWarpSize; ++i) {
            int32_t expected_up_i = (i >= 3) ? h_in_i[i - 3] : h_in_i[i];
            float expected_up_f = (i >= 3) ? h_in_f[i - 3] : h_in_f[i];
            EXPECT_EQ(h_out_up_i[i], expected_up_i) << optName << " up int lane " << i;
            EXPECT_FLOAT_EQ(h_out_up_f[i], expected_up_f) << optName << " up float lane " << i;
        }

        CHECK_CUDA(cuMemFree(d_in_i));
        CHECK_CUDA(cuMemFree(d_in_f));
        CHECK_CUDA(cuMemFree(d_out_reduce_i));
        CHECK_CUDA(cuMemFree(d_out_reduce_f));
        CHECK_CUDA(cuMemFree(d_out_xor_i));
        CHECK_CUDA(cuMemFree(d_out_xor_f));
        CHECK_CUDA(cuMemFree(d_out_bcast_i));
        CHECK_CUDA(cuMemFree(d_out_bcast_f));
        CHECK_CUDA(cuMemFree(d_out_ballot_i));
        CHECK_CUDA(cuMemFree(d_out_ballot_int_i));
        CHECK_CUDA(cuMemFree(d_out_activemask_i));
        CHECK_CUDA(cuMemFree(d_out_activemask2_i));
        CHECK_CUDA(cuMemFree(d_out_up_i));
        CHECK_CUDA(cuMemFree(d_out_up_f));

        CHECK_CUDA(cuModuleUnload(mod));
    }
}

TEST_F(KernelE2ETest, AtomicOperationsMatchCpu) {
    std::string src =
        "func int dev_atomic_add(int* p, int val) {\n"
        "    return atomic_add(p, val);\n"
        "}\n"
        "\n"
        "kernel test_atomics_e2e(\n"
        "    int* d_counter, float* d_counter_f, int* d_hist,\n"
        "    int* d_tickets, int* d_sub, int* d_min, int* d_max,\n"
        "    int* d_and, int* d_or, int* d_xor,\n"
        "    int* d_exch_i, float* d_exch_f,\n"
        "    int* out_old_exch_i, float* out_old_exch_f,\n"
        "    int* d_cas_winner, int* d_cas_success,\n"
        "    int* d_dev_counter,\n"
        "    int* out_shared_sum, float* out_shared_sum_f,\n"
        "    int* out_shared_cas_winners, int* out_shared_min, int* out_shared_max,\n"
        "    int* out_shared_sub, int* out_shared_bitwise, int* out_shared_exch,\n"
        "    int n\n"
        ") {\n"
        "    shared int s_counter[16];\n"
        "    shared float s_counter_f[16];\n"
        "    shared int s_min[1];\n"
        "    shared int s_max[1];\n"
        "    shared int s_cas_winner[1];\n"
        "    shared int s_sub[1];\n"
        "    shared int s_bitwise[3];\n"
        "    shared int s_exch[1];\n"
        "    shared float s_exch_f[1];\n"
        "    shared int s_old_exch[1];\n"
        "\n"
        "    int tid = thread_idx;\n"
        "    int gid = block_idx * block_dim + thread_idx;\n"
        "\n"
        "    if (tid < 16) {\n"
        "        s_counter[tid] = 0;\n"
        "        s_counter_f[tid] = 0.0;\n"
        "    }\n"
        "    if (tid == 0) {\n"
        "        s_min[0] = 999999;\n"
        "        s_max[0] = -999999;\n"
        "        s_cas_winner[0] = 0;\n"
        "        s_sub[0] = 256;\n"
        "        s_bitwise[0] = 0;\n"
        "        s_bitwise[1] = 0xFFFFFFFF;\n"
        "        s_bitwise[2] = 0xAAAAAAAA;\n"
        "        s_exch[0] = 777;\n"
        "        s_exch_f[0] = 33.5;\n"
        "        s_old_exch[0] = 0;\n"
        "    }\n"
        "    barrier();\n"
        "\n"
        "    if (gid < n) {\n"
        "        // 1. Global atomic add (int and float) with ticket collection verifying old value\n"
        "        int ticket = atomic_add(d_counter, 1);\n"
        "        d_tickets[ticket] = 1;\n"
        "        atomic_add(d_counter_f, 1.0);\n"
        "        atomic_add(d_hist, gid % 16, 1);\n"
        "\n"
        "        // 2. Global atomic sub\n"
        "        atomic_sub(d_sub, 2);\n"
        "\n"
        "        // 3. Global atomic min / max\n"
        "        atomic_min(d_min, gid * 7 + 13);\n"
        "        atomic_max(d_max, gid * 7 + 13);\n"
        "\n"
        "        // 4. Global bitwise atomics (xor flips all 32 bits from 0xAAAAAAAA to 0x55555555)\n"
        "        if (gid < 32) {\n"
        "            atomic_or(d_or, 1 << gid);\n"
        "            atomic_xor(d_xor, 1 << gid);\n"
        "        }\n"
        "        atomic_and(d_and, ~(1 << (gid % 32)));\n"
        "\n"
        "        // 5. Global atomic exch returning prior values\n"
        "        if (gid == 0) {\n"
        "            out_old_exch_i[0] = atomic_exch(d_exch_i, 12345);\n"
        "            out_old_exch_f[0] = atomic_exch(d_exch_f, 98.75);\n"
        "        }\n"
        "\n"
        "        // 6. Global atomic CAS\n"
        "        int old_cas = atomic_cas(d_cas_winner, 0, gid + 1);\n"
        "        if (old_cas == 0) {\n"
        "            atomic_add(d_cas_success, 1);\n"
        "        }\n"
        "\n"
        "        // 7. Device function calling atomic\n"
        "        dev_atomic_add(d_dev_counter, 1);\n"
        "\n"
        "        // 8. Shared memory atomics: add, sub, min, max, or, and, xor, exch, cas\n"
        "        atomic_add(s_counter, tid % 16, 1);\n"
        "        atomic_add(s_counter_f, tid % 16, 1.0);\n"
        "        atomic_min(s_min, tid + 1);\n"
        "        atomic_max(s_max, tid + 1);\n"
        "        atomic_sub(s_sub, 1);\n"
        "\n"
        "        if (tid < 32) {\n"
        "            atomic_or(s_bitwise, 0, 1 << tid);\n"
        "            atomic_and(s_bitwise, 1, ~(1 << tid));\n"
        "            atomic_xor(s_bitwise, 2, 1 << tid);\n"
        "        }\n"
        "\n"
        "        if (tid == 0) {\n"
        "            s_old_exch[0] = atomic_exch(s_exch, 888);\n"
        "            atomic_exch(s_exch_f, 55.5);\n"
        "        }\n"
        "\n"
        "        int s_old = atomic_cas(s_cas_winner, 0, tid + 1);\n"
        "        if (s_old == 0) {\n"
        "            atomic_add(out_shared_cas_winners, 1);\n"
        "        }\n"
        "    }\n"
        "\n"
        "    barrier();\n"
        "\n"
        "    // Accumulate shared results to global\n"
        "    if (gid < n) {\n"
        "        if (tid < 16) {\n"
        "            atomic_add(out_shared_sum, tid, s_counter[tid]);\n"
        "            atomic_add(out_shared_sum_f, tid, s_counter_f[tid]);\n"
        "        }\n"
        "        if (tid == 0) {\n"
        "            atomic_min(out_shared_min, s_min[0]);\n"
        "            atomic_max(out_shared_max, s_max[0]);\n"
        "            atomic_add(out_shared_sub, s_sub[0]);\n"
        "            atomic_add(out_shared_bitwise, 0, s_bitwise[0]);\n"
        "            atomic_add(out_shared_bitwise, 1, s_bitwise[1]);\n"
        "            atomic_add(out_shared_bitwise, 2, s_bitwise[2]);\n"
        "            atomic_add(out_shared_exch, s_exch[0] + s_old_exch[0]);\n"
        "        }\n"
        "    }\n"
        "}\n";

    constexpr int kNumBlocks = 16;
    constexpr int kBlockSize = 256;
    constexpr int N = kNumBlocks * kBlockSize; // 4096 threads in parallel

    for (bool optimize : {false, true}) {
        std::string optName = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty()) << optName;

        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr) << optName;

        CUfunction func;
        CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_atomics_e2e"));

        // Host buffers and initialization
        int32_t h_counter = 0;
        float h_counter_f = 0.0f;
        std::vector<int32_t> h_hist(16, 0);
        std::vector<int32_t> h_tickets(N, 0);
        int32_t h_sub = 10000;
        int32_t h_min = 999999;
        int32_t h_max = -999999;
        int32_t h_and = static_cast<int32_t>(0xFFFFFFFF);
        int32_t h_or = 0;
        int32_t h_xor = static_cast<int32_t>(0xAAAAAAAA);
        int32_t h_exch_i = -42;
        float h_exch_f = 3.14159f;
        int32_t h_out_old_exch_i = 0;
        float h_out_old_exch_f = 0.0f;
        int32_t h_cas_winner = 0;
        int32_t h_cas_success = 0;
        int32_t h_dev_counter = 0;
        std::vector<int32_t> h_out_shared_sum(16, 0);
        std::vector<float> h_out_shared_sum_f(16, 0.0f);
        int32_t h_out_shared_cas_winners = 0;
        int32_t h_out_shared_min = 999999;
        int32_t h_out_shared_max = -999999;
        int32_t h_out_shared_sub = 0;
        std::vector<int32_t> h_out_shared_bitwise(3, 0);
        int32_t h_out_shared_exch = 0;

        // Device pointers
        CUdeviceptr d_counter, d_counter_f, d_hist, d_tickets;
        CUdeviceptr d_sub, d_min, d_max;
        CUdeviceptr d_and, d_or, d_xor;
        CUdeviceptr d_exch_i, d_exch_f;
        CUdeviceptr d_out_old_exch_i, d_out_old_exch_f;
        CUdeviceptr d_cas_winner, d_cas_success;
        CUdeviceptr d_dev_counter;
        CUdeviceptr d_out_shared_sum, d_out_shared_sum_f;
        CUdeviceptr d_out_shared_cas_winners, d_out_shared_min, d_out_shared_max;
        CUdeviceptr d_out_shared_sub, d_out_shared_bitwise, d_out_shared_exch;

        CHECK_CUDA(cuMemAlloc(&d_counter, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_counter_f, sizeof(float)));
        CHECK_CUDA(cuMemAlloc(&d_hist, 16 * sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_tickets, N * sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_sub, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_min, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_max, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_and, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_or, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_xor, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_exch_i, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_exch_f, sizeof(float)));
        CHECK_CUDA(cuMemAlloc(&d_out_old_exch_i, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_out_old_exch_f, sizeof(float)));
        CHECK_CUDA(cuMemAlloc(&d_cas_winner, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_cas_success, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_dev_counter, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_out_shared_sum, 16 * sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_out_shared_sum_f, 16 * sizeof(float)));
        CHECK_CUDA(cuMemAlloc(&d_out_shared_cas_winners, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_out_shared_min, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_out_shared_max, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_out_shared_sub, sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_out_shared_bitwise, 3 * sizeof(int32_t)));
        CHECK_CUDA(cuMemAlloc(&d_out_shared_exch, sizeof(int32_t)));

        CHECK_CUDA(cuMemcpyHtoD(d_counter, &h_counter, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_counter_f, &h_counter_f, sizeof(float)));
        CHECK_CUDA(cuMemcpyHtoD(d_hist, h_hist.data(), 16 * sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_tickets, h_tickets.data(), N * sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_sub, &h_sub, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_min, &h_min, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_max, &h_max, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_and, &h_and, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_or, &h_or, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_xor, &h_xor, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_exch_i, &h_exch_i, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_exch_f, &h_exch_f, sizeof(float)));
        CHECK_CUDA(cuMemcpyHtoD(d_out_old_exch_i, &h_out_old_exch_i, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_out_old_exch_f, &h_out_old_exch_f, sizeof(float)));
        CHECK_CUDA(cuMemcpyHtoD(d_cas_winner, &h_cas_winner, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_cas_success, &h_cas_success, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_dev_counter, &h_dev_counter, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_out_shared_sum, h_out_shared_sum.data(), 16 * sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_out_shared_sum_f, h_out_shared_sum_f.data(), 16 * sizeof(float)));
        CHECK_CUDA(cuMemcpyHtoD(d_out_shared_cas_winners, &h_out_shared_cas_winners, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_out_shared_min, &h_out_shared_min, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_out_shared_max, &h_out_shared_max, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_out_shared_sub, &h_out_shared_sub, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_out_shared_bitwise, h_out_shared_bitwise.data(), 3 * sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyHtoD(d_out_shared_exch, &h_out_shared_exch, sizeof(int32_t)));

        int n_param = N;
        void* args[] = {
            &d_counter, &d_counter_f, &d_hist,
            &d_tickets, &d_sub, &d_min, &d_max,
            &d_and, &d_or, &d_xor,
            &d_exch_i, &d_exch_f,
            &d_out_old_exch_i, &d_out_old_exch_f,
            &d_cas_winner, &d_cas_success,
            &d_dev_counter,
            &d_out_shared_sum, &d_out_shared_sum_f,
            &d_out_shared_cas_winners, &d_out_shared_min, &d_out_shared_max,
            &d_out_shared_sub, &d_out_shared_bitwise, &d_out_shared_exch,
            &n_param
        };

        CHECK_CUDA(cuLaunchKernel(func, kNumBlocks, 1, 1, kBlockSize, 1, 1, 0, 0, args, 0));
        CHECK_CUDA(cuCtxSynchronize());

        CHECK_CUDA(cuMemcpyDtoH(&h_counter, d_counter, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_counter_f, d_counter_f, sizeof(float)));
        CHECK_CUDA(cuMemcpyDtoH(h_hist.data(), d_hist, 16 * sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(h_tickets.data(), d_tickets, N * sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_sub, d_sub, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_min, d_min, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_max, d_max, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_and, d_and, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_or, d_or, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_xor, d_xor, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_exch_i, d_exch_i, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_exch_f, d_exch_f, sizeof(float)));
        CHECK_CUDA(cuMemcpyDtoH(&h_out_old_exch_i, d_out_old_exch_i, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_out_old_exch_f, d_out_old_exch_f, sizeof(float)));
        CHECK_CUDA(cuMemcpyDtoH(&h_cas_winner, d_cas_winner, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_cas_success, d_cas_success, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_dev_counter, d_dev_counter, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(h_out_shared_sum.data(), d_out_shared_sum, 16 * sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(h_out_shared_sum_f.data(), d_out_shared_sum_f, 16 * sizeof(float)));
        CHECK_CUDA(cuMemcpyDtoH(&h_out_shared_cas_winners, d_out_shared_cas_winners, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_out_shared_min, d_out_shared_min, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_out_shared_max, d_out_shared_max, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_out_shared_sub, d_out_shared_sub, sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(h_out_shared_bitwise.data(), d_out_shared_bitwise, 3 * sizeof(int32_t)));
        CHECK_CUDA(cuMemcpyDtoH(&h_out_shared_exch, d_out_shared_exch, sizeof(int32_t)));

        // 1. Contention verification on atomic_add & ticket distribution (validating return value)
        EXPECT_EQ(h_counter, N) << optName << " global int atomic_add sum";
        EXPECT_FLOAT_EQ(h_counter_f, static_cast<float>(N)) << optName << " global float atomic_add sum";
        for (int b = 0; b < 16; ++b) {
            EXPECT_EQ(h_hist[b], N / 16) << optName << " global histogram bin " << b;
        }
        for (int i = 0; i < N; ++i) {
            ASSERT_EQ(h_tickets[i], 1) << optName << " ticket " << i << " was not claimed exactly once";
        }

        // 2. Atomic sub verification
        EXPECT_EQ(h_sub, 10000 - N * 2) << optName << " global atomic_sub";

        // 3. Atomic min and max verification
        EXPECT_EQ(h_min, 13) << optName << " global atomic_min";
        EXPECT_EQ(h_max, (N - 1) * 7 + 13) << optName << " global atomic_max";

        // 4. Bitwise atomics verification
        EXPECT_EQ(h_or, static_cast<int32_t>(0xFFFFFFFF)) << optName << " global atomic_or";
        EXPECT_EQ(h_and, 0) << optName << " global atomic_and";
        EXPECT_EQ(h_xor, static_cast<int32_t>(0x55555555)) << optName << " global atomic_xor non-trivial flip";

        // 5. Atomic exch verification (value and return value)
        EXPECT_EQ(h_exch_i, 12345) << optName << " global atomic_exch int new value";
        EXPECT_FLOAT_EQ(h_exch_f, 98.75f) << optName << " global atomic_exch float new value";
        EXPECT_EQ(h_out_old_exch_i, -42) << optName << " global atomic_exch int return value (old value)";
        EXPECT_FLOAT_EQ(h_out_old_exch_f, 3.14159f) << optName << " global atomic_exch float return value (old value)";

        // 6. Atomic CAS single-winner verification
        EXPECT_GT(h_cas_winner, 0) << optName << " CAS winner id";
        EXPECT_LE(h_cas_winner, N) << optName << " CAS winner id upper bound";
        EXPECT_EQ(h_cas_success, 1) << optName << " CAS success count (exactly 1 winner)";

        // 7. Device function atomic verification
        EXPECT_EQ(h_dev_counter, N) << optName << " device function atomic_add sum";

        // 8. Shared memory atomics verification
        for (int b = 0; b < 16; ++b) {
            EXPECT_EQ(h_out_shared_sum[b], N / 16) << optName << " shared atomic_add bin " << b;
            EXPECT_FLOAT_EQ(h_out_shared_sum_f[b], static_cast<float>(N / 16)) << optName << " shared float atomic_add bin " << b;
        }
        EXPECT_EQ(h_out_shared_cas_winners, kNumBlocks) << optName << " shared CAS winners (1 per block)";
        EXPECT_EQ(h_out_shared_min, 1) << optName << " shared atomic_min";
        EXPECT_EQ(h_out_shared_max, kBlockSize) << optName << " shared atomic_max";
        EXPECT_EQ(h_out_shared_sub, 0) << optName << " shared atomic_sub (256-256=0 per block)";
        EXPECT_EQ(h_out_shared_bitwise[0], -16) << optName << " shared atomic_or (16 * 0xFFFFFFFF)";
        EXPECT_EQ(h_out_shared_bitwise[1], 0) << optName << " shared atomic_and";
        EXPECT_EQ(h_out_shared_bitwise[2], static_cast<int32_t>(16u * 0x55555555u)) << optName << " shared atomic_xor";
        EXPECT_EQ(h_out_shared_exch, 16 * (888 + 777)) << optName << " shared atomic_exch (new + old value)";

        // Cleanup
        CHECK_CUDA(cuMemFree(d_counter));
        CHECK_CUDA(cuMemFree(d_counter_f));
        CHECK_CUDA(cuMemFree(d_hist));
        CHECK_CUDA(cuMemFree(d_tickets));
        CHECK_CUDA(cuMemFree(d_sub));
        CHECK_CUDA(cuMemFree(d_min));
        CHECK_CUDA(cuMemFree(d_max));
        CHECK_CUDA(cuMemFree(d_and));
        CHECK_CUDA(cuMemFree(d_or));
        CHECK_CUDA(cuMemFree(d_xor));
        CHECK_CUDA(cuMemFree(d_exch_i));
        CHECK_CUDA(cuMemFree(d_exch_f));
        CHECK_CUDA(cuMemFree(d_out_old_exch_i));
        CHECK_CUDA(cuMemFree(d_out_old_exch_f));
        CHECK_CUDA(cuMemFree(d_cas_winner));
        CHECK_CUDA(cuMemFree(d_cas_success));
        CHECK_CUDA(cuMemFree(d_dev_counter));
        CHECK_CUDA(cuMemFree(d_out_shared_sum));
        CHECK_CUDA(cuMemFree(d_out_shared_sum_f));
        CHECK_CUDA(cuMemFree(d_out_shared_cas_winners));
        CHECK_CUDA(cuMemFree(d_out_shared_min));
        CHECK_CUDA(cuMemFree(d_out_shared_max));
        CHECK_CUDA(cuMemFree(d_out_shared_sub));
        CHECK_CUDA(cuMemFree(d_out_shared_bitwise));
        CHECK_CUDA(cuMemFree(d_out_shared_exch));

        CHECK_CUDA(cuModuleUnload(mod));
    }
}

TEST_F(KernelE2ETest, VectorMemoryOperationsMatchCpu) {
    struct TestInt2 { int32_t x, y; };
    struct TestInt4 { int32_t x, y, z, w; };
    struct TestFloat2 { float x, y; };
    struct TestFloat4 { float x, y, z, w; };

    std::string src =
        "kernel test_saxpy_v4(float4* x, float4* y, float a, int n) {\n"
        "    int i = block_idx * block_dim + thread_idx;\n"
        "    if (i < n) {\n"
        "        float4 vx = x[i];\n"
        "        float4 vy = y[i];\n"
        "        y[i] = float4(a * vx.x + vy.x, a * vx.y + vy.y, a * vx.z + vy.z, a * vx.w + vy.w);\n"
        "    }\n"
        "}\n"
        "\n"
        "kernel test_shared_reverse_v(int4* in4, int4* out4, int2* in2, int2* out2, int n) {\n"
        "    shared int4 s4[64];\n"
        "    shared int2 s2[64];\n"
        "    int tid = thread_idx;\n"
        "    int gid = block_idx * block_dim + thread_idx;\n"
        "    if (gid < n) {\n"
        "        s4[tid] = in4[gid];\n"
        "        s2[tid] = in2[gid];\n"
        "    }\n"
        "    barrier();\n"
        "    if (gid < n) {\n"
        "        out4[gid] = s4[63 - tid];\n"
        "        out2[gid] = s2[63 - tid];\n"
        "    }\n"
        "}\n"
        "\n"
        "kernel test_component_mutation(int4* in, int4* out, int n) {\n"
        "    int i = block_idx * block_dim + thread_idx;\n"
        "    if (i < n) {\n"
        "        int4 v = in[i];\n"
        "        v.x = v.x + 10;\n"
        "        v.y = v.y + 20;\n"
        "        v.z = v.z + 30;\n"
        "        v.w = v.w * 2;\n"
        "        out[i] = v;\n"
        "        out[i].y = out[i].y + 500;\n"
        "        out[i].w = out[i].w + 1000;\n"
        "    }\n"
        "}\n"
        "\n"
        "kernel test_raw_buffer_intrinsics(int* raw_i_in, int* raw_i_out, float* raw_f_in, float* raw_f_out, int n) {\n"
        "    int i = block_idx * block_dim + thread_idx;\n"
        "    if (i < n) {\n"
        // Offsets count whole vectors: load_int4(p, i) reads p[4i .. 4i+3].
        "        int offset4 = i;\n"
        "        int4 vi4 = load_int4(raw_i_in, offset4);\n"
        "        vi4.x = vi4.x + 11;\n"
        "        vi4.y = vi4.y + 22;\n"
        "        vi4.z = vi4.z + 33;\n"
        "        vi4.w = vi4.w + 44;\n"
        "        store_int4(raw_i_out, offset4, vi4);\n"
        "\n"
        "        int offset2 = i;\n"
        "        int2 vi2 = load_int2(raw_i_in, offset2);\n"
        "        vi2.x = vi2.x + 100;\n"
        "        vi2.y = vi2.y + 200;\n"
        "        store_int2(raw_i_out, n * 2 + offset2, vi2);\n"
        "\n"
        "        float4 vf4 = load_float4(raw_f_in, offset4);\n"
        "        vf4.x = vf4.x * 2.0;\n"
        "        vf4.y = vf4.y * 3.0;\n"
        "        vf4.z = vf4.z * 4.0;\n"
        "        vf4.w = vf4.w * 5.0;\n"
        "        store_float4(raw_f_out, offset4, vf4);\n"
        "\n"
        "        float2 vf2 = load_float2(raw_f_in, offset2);\n"
        "        vf2.x = vf2.x + 1.5;\n"
        "        vf2.y = vf2.y + 2.5;\n"
        "        store_float2(raw_f_out, n * 2 + offset2, vf2);\n"
        "    }\n"
        "}\n";

    constexpr int N = 256;
    constexpr int kBlockSize = 64;
    constexpr int kGridSize = N / kBlockSize; // 4 blocks

    for (bool optimize : {false, true}) {
        std::string optName = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty()) << optName;

        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr) << optName;

        // =========================================================
        // 1. Vector SAXPY float4
        // =========================================================
        {
            CUfunction func;
            CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_saxpy_v4"));

            std::vector<TestFloat4> h_x(N), h_y(N), h_y_ref(N);
            float a = 2.5f;
            for (int i = 0; i < N; ++i) {
                h_x[i] = { i * 1.5f, i * -0.5f, (i + 1) * 2.0f, (i + 2) * 0.25f };
                h_y[i] = { (i + 3) * 0.75f, (i + 4) * 1.25f, (i + 5) * -1.0f, (i + 6) * 3.5f };
                h_y_ref[i] = {
                    a * h_x[i].x + h_y[i].x,
                    a * h_x[i].y + h_y[i].y,
                    a * h_x[i].z + h_y[i].z,
                    a * h_x[i].w + h_y[i].w
                };
            }

            CUdeviceptr d_x, d_y;
            size_t bytesF4 = N * sizeof(TestFloat4);
            CHECK_CUDA(cuMemAlloc(&d_x, bytesF4));
            CHECK_CUDA(cuMemAlloc(&d_y, bytesF4));
            CHECK_CUDA(cuMemcpyHtoD(d_x, h_x.data(), bytesF4));
            CHECK_CUDA(cuMemcpyHtoD(d_y, h_y.data(), bytesF4));

            int n_arg = N;
            void* args[] = { &d_x, &d_y, &a, &n_arg };
            CHECK_CUDA(cuLaunchKernel(func, kGridSize, 1, 1, kBlockSize, 1, 1, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());

            CHECK_CUDA(cuMemcpyDtoH(h_y.data(), d_y, bytesF4));
            for (int i = 0; i < N; ++i) {
                EXPECT_FLOAT_EQ(h_y[i].x, h_y_ref[i].x) << optName << " SAXPY x at " << i;
                EXPECT_FLOAT_EQ(h_y[i].y, h_y_ref[i].y) << optName << " SAXPY y at " << i;
                EXPECT_FLOAT_EQ(h_y[i].z, h_y_ref[i].z) << optName << " SAXPY z at " << i;
                EXPECT_FLOAT_EQ(h_y[i].w, h_y_ref[i].w) << optName << " SAXPY w at " << i;
            }

            CHECK_CUDA(cuMemFree(d_x));
            CHECK_CUDA(cuMemFree(d_y));
        }

        // =========================================================
        // 2. Vector shared memory reverse (int4 and int2)
        // =========================================================
        {
            CUfunction func;
            CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_shared_reverse_v"));

            std::vector<TestInt4> h_in4(N), h_out4(N), h_ref4(N);
            std::vector<TestInt2> h_in2(N), h_out2(N), h_ref2(N);
            for (int i = 0; i < N; ++i) {
                h_in4[i] = { i * 4 + 1, i * 4 + 2, i * 4 + 3, i * 4 + 4 };
                h_in2[i] = { i * 2 + 10, i * 2 + 20 };
            }
            for (int b = 0; b < kGridSize; ++b) {
                for (int t = 0; t < kBlockSize; ++t) {
                    int gid = b * kBlockSize + t;
                    int rev_gid = b * kBlockSize + (kBlockSize - 1 - t);
                    h_ref4[gid] = h_in4[rev_gid];
                    h_ref2[gid] = h_in2[rev_gid];
                }
            }

            CUdeviceptr d_in4, d_out4, d_in2, d_out2;
            size_t bytesI4 = N * sizeof(TestInt4);
            size_t bytesI2 = N * sizeof(TestInt2);
            CHECK_CUDA(cuMemAlloc(&d_in4, bytesI4));
            CHECK_CUDA(cuMemAlloc(&d_out4, bytesI4));
            CHECK_CUDA(cuMemAlloc(&d_in2, bytesI2));
            CHECK_CUDA(cuMemAlloc(&d_out2, bytesI2));

            CHECK_CUDA(cuMemcpyHtoD(d_in4, h_in4.data(), bytesI4));
            CHECK_CUDA(cuMemcpyHtoD(d_in2, h_in2.data(), bytesI2));

            int n_arg = N;
            void* args[] = { &d_in4, &d_out4, &d_in2, &d_out2, &n_arg };
            CHECK_CUDA(cuLaunchKernel(func, kGridSize, 1, 1, kBlockSize, 1, 1, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());

            CHECK_CUDA(cuMemcpyDtoH(h_out4.data(), d_out4, bytesI4));
            CHECK_CUDA(cuMemcpyDtoH(h_out2.data(), d_out2, bytesI2));

            for (int i = 0; i < N; ++i) {
                EXPECT_EQ(h_out4[i].x, h_ref4[i].x) << optName << " shared reverse int4.x at " << i;
                EXPECT_EQ(h_out4[i].y, h_ref4[i].y) << optName << " shared reverse int4.y at " << i;
                EXPECT_EQ(h_out4[i].z, h_ref4[i].z) << optName << " shared reverse int4.z at " << i;
                EXPECT_EQ(h_out4[i].w, h_ref4[i].w) << optName << " shared reverse int4.w at " << i;

                EXPECT_EQ(h_out2[i].x, h_ref2[i].x) << optName << " shared reverse int2.x at " << i;
                EXPECT_EQ(h_out2[i].y, h_ref2[i].y) << optName << " shared reverse int2.y at " << i;
            }

            CHECK_CUDA(cuMemFree(d_in4));
            CHECK_CUDA(cuMemFree(d_out4));
            CHECK_CUDA(cuMemFree(d_in2));
            CHECK_CUDA(cuMemFree(d_out2));
        }

        // =========================================================
        // 3. Component arithmetic and mutation
        // =========================================================
        {
            CUfunction func;
            CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_component_mutation"));

            std::vector<TestInt4> h_mut_in(N), h_mut_out(N), h_mut_ref(N);
            for (int i = 0; i < N; ++i) {
                h_mut_in[i] = { i * 10, i * 20, i * 30, i * 40 };
                h_mut_ref[i] = {
                    h_mut_in[i].x + 10,
                    h_mut_in[i].y + 20 + 500,
                    h_mut_in[i].z + 30,
                    (h_mut_in[i].w * 2) + 1000
                };
            }

            CUdeviceptr d_in, d_out;
            size_t bytesI4 = N * sizeof(TestInt4);
            CHECK_CUDA(cuMemAlloc(&d_in, bytesI4));
            CHECK_CUDA(cuMemAlloc(&d_out, bytesI4));

            CHECK_CUDA(cuMemcpyHtoD(d_in, h_mut_in.data(), bytesI4));

            int n_arg = N;
            void* args[] = { &d_in, &d_out, &n_arg };
            CHECK_CUDA(cuLaunchKernel(func, kGridSize, 1, 1, kBlockSize, 1, 1, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());

            CHECK_CUDA(cuMemcpyDtoH(h_mut_out.data(), d_out, bytesI4));

            for (int i = 0; i < N; ++i) {
                EXPECT_EQ(h_mut_out[i].x, h_mut_ref[i].x) << optName << " mutation x at " << i;
                EXPECT_EQ(h_mut_out[i].y, h_mut_ref[i].y) << optName << " mutation y at " << i;
                EXPECT_EQ(h_mut_out[i].z, h_mut_ref[i].z) << optName << " mutation z at " << i;
                EXPECT_EQ(h_mut_out[i].w, h_mut_ref[i].w) << optName << " mutation w at " << i;
            }

            CHECK_CUDA(cuMemFree(d_in));
            CHECK_CUDA(cuMemFree(d_out));
        }

        // =========================================================
        // 4. Raw scalar buffer vector loading/storing
        // =========================================================
        {
            CUfunction func;
            CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_raw_buffer_intrinsics"));

            size_t totalInts = N * 4 + N * 2;
            size_t totalFloats = N * 4 + N * 2;
            std::vector<int32_t> h_raw_i_in(totalInts), h_raw_i_out(totalInts, 0), h_raw_i_ref(totalInts, 0);
            std::vector<float> h_raw_f_in(totalFloats), h_raw_f_out(totalFloats, 0.0f), h_raw_f_ref(totalFloats, 0.0f);

            for (size_t i = 0; i < totalInts; ++i) {
                h_raw_i_in[i] = static_cast<int32_t>(i * 3 + 7);
            }
            for (size_t i = 0; i < totalFloats; ++i) {
                h_raw_f_in[i] = static_cast<float>(i) * 0.5f + 1.25f;
            }

            for (int i = 0; i < N; ++i) {
                int off4 = i * 4;
                h_raw_i_ref[off4 + 0] = h_raw_i_in[off4 + 0] + 11;
                h_raw_i_ref[off4 + 1] = h_raw_i_in[off4 + 1] + 22;
                h_raw_i_ref[off4 + 2] = h_raw_i_in[off4 + 2] + 33;
                h_raw_i_ref[off4 + 3] = h_raw_i_in[off4 + 3] + 44;

                int off2 = i * 2;
                h_raw_i_ref[N * 4 + off2 + 0] = h_raw_i_in[off2 + 0] + 100;
                h_raw_i_ref[N * 4 + off2 + 1] = h_raw_i_in[off2 + 1] + 200;

                h_raw_f_ref[off4 + 0] = h_raw_f_in[off4 + 0] * 2.0f;
                h_raw_f_ref[off4 + 1] = h_raw_f_in[off4 + 1] * 3.0f;
                h_raw_f_ref[off4 + 2] = h_raw_f_in[off4 + 2] * 4.0f;
                h_raw_f_ref[off4 + 3] = h_raw_f_in[off4 + 3] * 5.0f;

                h_raw_f_ref[N * 4 + off2 + 0] = h_raw_f_in[off2 + 0] + 1.5f;
                h_raw_f_ref[N * 4 + off2 + 1] = h_raw_f_in[off2 + 1] + 2.5f;
            }

            CUdeviceptr d_raw_i_in, d_raw_i_out, d_raw_f_in, d_raw_f_out;
            CHECK_CUDA(cuMemAlloc(&d_raw_i_in, totalInts * sizeof(int32_t)));
            CHECK_CUDA(cuMemAlloc(&d_raw_i_out, totalInts * sizeof(int32_t)));
            CHECK_CUDA(cuMemAlloc(&d_raw_f_in, totalFloats * sizeof(float)));
            CHECK_CUDA(cuMemAlloc(&d_raw_f_out, totalFloats * sizeof(float)));

            CHECK_CUDA(cuMemcpyHtoD(d_raw_i_in, h_raw_i_in.data(), totalInts * sizeof(int32_t)));
            CHECK_CUDA(cuMemcpyHtoD(d_raw_f_in, h_raw_f_in.data(), totalFloats * sizeof(float)));

            int n_arg = N;
            void* args[] = { &d_raw_i_in, &d_raw_i_out, &d_raw_f_in, &d_raw_f_out, &n_arg };
            CHECK_CUDA(cuLaunchKernel(func, kGridSize, 1, 1, kBlockSize, 1, 1, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());

            CHECK_CUDA(cuMemcpyDtoH(h_raw_i_out.data(), d_raw_i_out, totalInts * sizeof(int32_t)));
            CHECK_CUDA(cuMemcpyDtoH(h_raw_f_out.data(), d_raw_f_out, totalFloats * sizeof(float)));

            for (size_t i = 0; i < totalInts; ++i) {
                EXPECT_EQ(h_raw_i_out[i], h_raw_i_ref[i]) << optName << " raw int at " << i;
            }
            for (size_t i = 0; i < totalFloats; ++i) {
                EXPECT_FLOAT_EQ(h_raw_f_out[i], h_raw_f_ref[i]) << optName << " raw float at " << i;
            }

            CHECK_CUDA(cuMemFree(d_raw_i_in));
            CHECK_CUDA(cuMemFree(d_raw_i_out));
            CHECK_CUDA(cuMemFree(d_raw_f_in));
            CHECK_CUDA(cuMemFree(d_raw_f_out));
        }

        CHECK_CUDA(cuModuleUnload(mod));
    }
}

TEST_F(KernelE2ETest, LoopUnrollingAndSchedulingMatchCpu) {
    const std::string kernelSrc = R"(
kernel test_stencil_unroll(float* in, float* out, int n) {
    int tid = thread_idx;
    if (tid < n) {
        float sum = 0.0;
        for (int i = 0; i < 5; i = i + 1) {
            sum = sum + in[tid + i];
        }
        out[tid] = sum;
    }
}

kernel test_dot_product_unroll(int* a, int* b, int* out, int n) {
    int tid = thread_idx;
    if (tid < n) {
        int acc = 0;
        for (int i = 0; i < 8; i = i + 1) {
            acc = acc + a[tid * 8 + i] * b[tid * 8 + i];
        }
        out[tid] = acc;
    }
}

kernel test_sched_multistream(float* out, float* in_x, float* in_y, int n) {
    int tid = thread_idx;
    if (tid < n) {
        for (int i = 0; i < 4; i = i + 1) {
            float x = in_x[tid * 4 + i];
            float y = in_y[tid * 4 + i];
            out[tid * 4 + i] = x * 2.0 + y * 3.0;
        }
    }
}
)";

    for (bool optimize : {false, true}) {
        std::string optName = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(kernelSrc, optimize);
        ASSERT_FALSE(ptx.empty()) << "Compilation failed under " << optName;

        CUmodule mod = loadPTX(ptx);
        const int N = 64;
        const int kBlockSize = 64;
        const int kGridSize = 1;

        // 1. 5-point stencil filter test
        {
            CUfunction func;
            CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_stencil_unroll"));

            int inLen = N + 4;
            std::vector<float> h_in(inLen), h_out(N, 0.0f), h_ref(N, 0.0f);
            for (int i = 0; i < inLen; ++i) {
                h_in[i] = static_cast<float>(i + 1) * 0.25f;
            }
            for (int tid = 0; tid < N; ++tid) {
                float sum = 0.0f;
                for (int i = 0; i < 5; ++i) {
                    sum += h_in[tid + i];
                }
                h_ref[tid] = sum;
            }

            CUdeviceptr d_in, d_out;
            CHECK_CUDA(cuMemAlloc(&d_in, inLen * sizeof(float)));
            CHECK_CUDA(cuMemAlloc(&d_out, N * sizeof(float)));
            CHECK_CUDA(cuMemcpyHtoD(d_in, h_in.data(), inLen * sizeof(float)));

            int n_arg = N;
            void* args[] = { &d_in, &d_out, &n_arg };
            CHECK_CUDA(cuLaunchKernel(func, kGridSize, 1, 1, kBlockSize, 1, 1, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());

            CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(float)));

            for (int i = 0; i < N; ++i) {
                EXPECT_FLOAT_EQ(h_out[i], h_ref[i]) << optName << " stencil failed at " << i;
            }

            CHECK_CUDA(cuMemFree(d_in));
            CHECK_CUDA(cuMemFree(d_out));
        }

        // 2. 8-element dot-product reduction test
        {
            CUfunction func;
            CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_dot_product_unroll"));

            int totalElements = N * 8;
            std::vector<int32_t> h_a(totalElements), h_b(totalElements);
            std::vector<int32_t> h_out(N, 0), h_ref(N, 0);

            for (int i = 0; i < totalElements; ++i) {
                h_a[i] = (i % 7) - 3;
                h_b[i] = ((i * 3) % 11) - 5;
            }
            for (int tid = 0; tid < N; ++tid) {
                int acc = 0;
                for (int i = 0; i < 8; ++i) {
                    acc += h_a[tid * 8 + i] * h_b[tid * 8 + i];
                }
                h_ref[tid] = acc;
            }

            CUdeviceptr d_a, d_b, d_out;
            CHECK_CUDA(cuMemAlloc(&d_a, totalElements * sizeof(int32_t)));
            CHECK_CUDA(cuMemAlloc(&d_b, totalElements * sizeof(int32_t)));
            CHECK_CUDA(cuMemAlloc(&d_out, N * sizeof(int32_t)));

            CHECK_CUDA(cuMemcpyHtoD(d_a, h_a.data(), totalElements * sizeof(int32_t)));
            CHECK_CUDA(cuMemcpyHtoD(d_b, h_b.data(), totalElements * sizeof(int32_t)));

            int n_arg = N;
            void* args[] = { &d_a, &d_b, &d_out, &n_arg };
            CHECK_CUDA(cuLaunchKernel(func, kGridSize, 1, 1, kBlockSize, 1, 1, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());

            CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(int32_t)));

            for (int i = 0; i < N; ++i) {
                EXPECT_EQ(h_out[i], h_ref[i]) << optName << " dot product failed at " << i;
            }

            CHECK_CUDA(cuMemFree(d_a));
            CHECK_CUDA(cuMemFree(d_b));
            CHECK_CUDA(cuMemFree(d_out));
        }

        // 3. Pipelined multi-stream load/ALU/store test
        {
            CUfunction func;
            CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_sched_multistream"));

            int totalElements = N * 4;
            std::vector<float> h_x(totalElements), h_y(totalElements);
            std::vector<float> h_out(totalElements, 0.0f), h_ref(totalElements, 0.0f);

            for (int i = 0; i < totalElements; ++i) {
                h_x[i] = static_cast<float>(i) * 1.5f + 0.25f;
                h_y[i] = static_cast<float>(i) * -0.75f + 10.0f;
            }
            for (int tid = 0; tid < N; ++tid) {
                for (int i = 0; i < 4; ++i) {
                    int idx = tid * 4 + i;
                    h_ref[idx] = h_x[idx] * 2.0f + h_y[idx] * 3.0f;
                }
            }

            CUdeviceptr d_x, d_y, d_out;
            CHECK_CUDA(cuMemAlloc(&d_x, totalElements * sizeof(float)));
            CHECK_CUDA(cuMemAlloc(&d_y, totalElements * sizeof(float)));
            CHECK_CUDA(cuMemAlloc(&d_out, totalElements * sizeof(float)));

            CHECK_CUDA(cuMemcpyHtoD(d_x, h_x.data(), totalElements * sizeof(float)));
            CHECK_CUDA(cuMemcpyHtoD(d_y, h_y.data(), totalElements * sizeof(float)));

            int n_arg = N;
            void* args[] = { &d_out, &d_x, &d_y, &n_arg };
            CHECK_CUDA(cuLaunchKernel(func, kGridSize, 1, 1, kBlockSize, 1, 1, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());

            CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, totalElements * sizeof(float)));

            for (int i = 0; i < totalElements; ++i) {
                EXPECT_FLOAT_EQ(h_out[i], h_ref[i]) << optName << " multistream sched failed at " << i;
            }

            CHECK_CUDA(cuMemFree(d_x));
            CHECK_CUDA(cuMemFree(d_y));
            CHECK_CUDA(cuMemFree(d_out));
        }

        CHECK_CUDA(cuModuleUnload(mod));
    }
}

TEST_F(KernelE2ETest, CSEMatchesUnoptimizedGPUAndCpu) {
    const char* src =
        "kernel test_cse_differential(int* out, int* a, int* b, int n) {\n"
        "    int i = block_idx * block_dim + thread_idx;\n"
        "    if (i < n) {\n"
        "        int v1 = a[i];\n"
        "        int v2 = a[i];\n"
        "        int sum = v1 + v2;\n"
        "        int c1 = v1 * 3 + v2 * 5;\n"
        "        int c2 = v2 * 5 + v1 * 3;\n"
        "        b[i] = sum + c1 + c2;\n"
        "        int v3 = a[i];\n"
        "        out[i] = v3 + b[i];\n"
        "    }\n"
        "}\n";

    const int N = 256;
    for (bool optimize : {false, true}) {
        std::string optName = optimize ? "-O1" : "-O0";
        std::string ptx = compileSourceToPTX(src, optimize);
        ASSERT_FALSE(ptx.empty()) << optName;
        CUmodule mod = loadPTX(ptx);
        ASSERT_NE(mod, nullptr) << optName;
        CUfunction func;
        CHECK_CUDA(cuModuleGetFunction(&func, mod, "test_cse_differential"));

        // Case 1: a and b are disjoint
        {
            std::vector<int32_t> h_a(N), h_b(N, 0), h_out(N, 0), h_ref(N, 0);
            for (int i = 0; i < N; ++i) {
                h_a[i] = i + 1;
                int v1 = h_a[i];
                int v2 = h_a[i];
                int sum = v1 + v2;
                int c1 = v1 * 3 + v2 * 5;
                int c2 = v2 * 5 + v1 * 3;
                int b_val = sum + c1 + c2;
                int v3 = h_a[i];
                h_ref[i] = v3 + b_val;
            }

            CUdeviceptr d_a, d_b, d_out;
            CHECK_CUDA(cuMemAlloc(&d_a, N * sizeof(int32_t)));
            CHECK_CUDA(cuMemAlloc(&d_b, N * sizeof(int32_t)));
            CHECK_CUDA(cuMemAlloc(&d_out, N * sizeof(int32_t)));

            CHECK_CUDA(cuMemcpyHtoD(d_a, h_a.data(), N * sizeof(int32_t)));
            CHECK_CUDA(cuMemsetD8(d_b, 0, N * sizeof(int32_t)));

            int n_arg = N;
            void* args[] = { &d_out, &d_a, &d_b, &n_arg };
            CHECK_CUDA(cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());

            CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(int32_t)));

            for (int i = 0; i < N; ++i) {
                EXPECT_EQ(h_out[i], h_ref[i]) << optName << " disjoint failed at " << i;
            }

            CHECK_CUDA(cuMemFree(d_a));
            CHECK_CUDA(cuMemFree(d_b));
            CHECK_CUDA(cuMemFree(d_out));
        }

        // Case 2: a and b ALIAS (a == b)
        // Store to b[i] modifies a[i] before v3 = a[i] read!
        {
            std::vector<int32_t> h_a(N), h_out(N, 0), h_ref(N, 0);
            for (int i = 0; i < N; ++i) {
                h_a[i] = i + 1;
                int v1 = h_a[i];
                int v2 = h_a[i];
                int sum = v1 + v2;
                int c1 = v1 * 3 + v2 * 5;
                int c2 = v2 * 5 + v1 * 3;
                int b_val = sum + c1 + c2;
                int v3 = b_val; // Since a == b, v3 reads the stored b_val!
                h_ref[i] = v3 + b_val;
            }

            CUdeviceptr d_ab, d_out;
            CHECK_CUDA(cuMemAlloc(&d_ab, N * sizeof(int32_t)));
            CHECK_CUDA(cuMemAlloc(&d_out, N * sizeof(int32_t)));

            CHECK_CUDA(cuMemcpyHtoD(d_ab, h_a.data(), N * sizeof(int32_t)));

            int n_arg = N;
            void* args[] = { &d_out, &d_ab, &d_ab, &n_arg };
            CHECK_CUDA(cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, 0, args, 0));
            CHECK_CUDA(cuCtxSynchronize());

            CHECK_CUDA(cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(int32_t)));

            for (int i = 0; i < N; ++i) {
                EXPECT_EQ(h_out[i], h_ref[i]) << optName << " aliased failed at " << i;
            }

            CHECK_CUDA(cuMemFree(d_ab));
            CHECK_CUDA(cuMemFree(d_out));
        }

        CHECK_CUDA(cuModuleUnload(mod));
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CudaEnvironment);
    return RUN_ALL_TESTS();
}
