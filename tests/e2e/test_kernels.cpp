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

static std::string compileSourceToPTX(const std::string& src) {
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
    Optimizer::runPasses(irMod);

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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CudaEnvironment);
    return RUN_ALL_TESTS();
}
