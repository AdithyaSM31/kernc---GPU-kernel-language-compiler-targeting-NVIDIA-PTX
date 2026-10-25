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
#include <random>
#include <iostream>

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

static std::string compileSourceToPTX(const std::string& src, bool optimize = true) {
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

    PTXEmitter emitter;
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CudaEnvironment);
    return RUN_ALL_TESTS();
}
