#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#define CHECK_CUDA(call) \
    do { \
        CUresult err = (call); \
        if (err != CUDA_SUCCESS) { \
            const char* errStr = nullptr; \
            cuGetErrorString(err, &errStr); \
            std::cerr << "CUDA Error at " << __LINE__ << ": " \
                      << (errStr ? errStr : "unknown error") << "\n"; \
            return 1; \
        } \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <ptx_file> <kernel_name> <n>\n"
                  << "Example: " << argv[0] << " vecadd.ptx vecadd 1024\n";
        return 1;
    }

    std::string ptxPath = argv[1];
    std::string kernelName = argv[2];
    int n = std::stoi(argv[3]);

    std::ifstream in(ptxPath);
    if (!in.is_open()) {
        std::cerr << "Failed to open PTX file: " << ptxPath << "\n";
        return 1;
    }
    std::string ptxSource((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    in.close();

    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, nullptr, 0, dev));

    char devName[128] = {0};
    cuDeviceGetName(devName, sizeof(devName), dev);
    std::cout << "[Runner] Loaded GPU: " << devName << "\n";

    CUmodule mod;
    char errLog[4096] = {0};
    CUjit_option opts[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
    void* optVals[] = { (void*)errLog, (void*)(uintptr_t)sizeof(errLog) };
    CUresult loadRes = cuModuleLoadDataEx(&mod, ptxSource.c_str(), 2, opts, optVals);
    if (loadRes != CUDA_SUCCESS) {
        std::cerr << "JIT load error:\n" << errLog << "\n";
        return 1;
    }

    CUfunction func;
    CHECK_CUDA(cuModuleGetFunction(&func, mod, kernelName.c_str()));
    std::cout << "[Runner] Found kernel: " << kernelName << "\n";

    // Allocate test buffers assuming vecadd or saxpy layout
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    std::cout << "[Runner] Launching grid=" << gridSize << " block=" << blockSize << " n=" << n << "\n";

    // Clean up
    CHECK_CUDA(cuModuleUnload(mod));
    CHECK_CUDA(cuCtxDestroy(ctx));
    std::cout << "[Runner] Success!\n";
    return 0;
}
