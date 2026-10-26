#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <cerrno>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <filesystem>

template <typename F>
class ScopeExit {
    F fn_;
    bool active_;
public:
    explicit ScopeExit(F fn) : fn_(std::move(fn)), active_(true) {}
    ~ScopeExit() { if (active_) fn_(); }
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ScopeExit(ScopeExit&& o) noexcept : fn_(std::move(o.fn_)), active_(o.active_) { o.active_ = false; }
    void dismiss() { active_ = false; }
};

template <typename F>
ScopeExit<F> makeScopeExit(F fn) {
    return ScopeExit<F>(std::move(fn));
}

static std::string getCudaErrorString(CUresult res) {
    const char* str = nullptr;
    const char* name = nullptr;
    cuGetErrorName(res, &name);
    cuGetErrorString(res, &str);
    std::string out;
    if (name) out += name;
    if (str) {
        if (!out.empty()) out += ": ";
        out += str;
    }
    if (out.empty()) out = "CUDA error code " + std::to_string(static_cast<int>(res));
    return out;
}

static std::string stripPtxComments(const std::string& src) {
    std::string clean;
    clean.reserve(src.size());
    size_t i = 0;
    while (i < src.size()) {
        if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '/') {
            i += 2;
            while (i < src.size() && src[i] != '\n') {
                ++i;
            }
            if (i < src.size() && src[i] == '\n') {
                clean += '\n';
                ++i;
            }
        } else if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '*') {
            i += 2;
            bool hasNewline = false;
            while (i + 1 < src.size() && !(src[i] == '*' && src[i + 1] == '/')) {
                if (src[i] == '\n') hasNewline = true;
                ++i;
            }
            if (i + 1 < src.size()) {
                i += 2;
            } else {
                i = src.size();
            }
            clean += (hasNewline ? '\n' : ' ');
        } else if (src[i] == '"') {
            clean += ' ';
            ++i;
            while (i < src.size() && src[i] != '"') {
                if (src[i] == '\\' && i + 1 < src.size()) {
                    clean += ' ';
                    ++i;
                }
                clean += (src[i] == '\n' ? '\n' : ' ');
                ++i;
            }
            if (i < src.size()) {
                clean += ' ';
                ++i;
            }
        } else {
            clean += src[i++];
        }
    }
    return clean;
}

static bool getExpectedParamCountFromPTX(const std::string& rawPtx, const std::string& kernelName, size_t& count) {
    std::string ptx = stripPtxComments(rawPtx);
    size_t pos = 0;
    while (true) {
        pos = ptx.find(".entry", pos);
        if (pos == std::string::npos) return false;

        if (pos > 0 && !std::isspace(static_cast<unsigned char>(ptx[pos - 1]))) {
            pos += 6;
            continue;
        }

        size_t p = pos + 6;
        if (p >= ptx.size() || !std::isspace(static_cast<unsigned char>(ptx[p]))) {
            pos += 6;
            continue;
        }

        while (p < ptx.size() && std::isspace(static_cast<unsigned char>(ptx[p]))) {
            ++p;
        }
        if (p >= ptx.size()) return false;

        size_t nameStart = p;
        while (p < ptx.size() && (std::isalnum(static_cast<unsigned char>(ptx[p])) || ptx[p] == '_' || ptx[p] == '$')) {
            ++p;
        }
        std::string foundName = ptx.substr(nameStart, p - nameStart);

        if (foundName != kernelName) {
            pos = p;
            continue;
        }

        while (p < ptx.size() && std::isspace(static_cast<unsigned char>(ptx[p]))) {
            ++p;
        }

        if (p >= ptx.size() || ptx[p] != '(') {
            count = 0;
            return true;
        }

        size_t openParen = p;
        size_t closeParen = ptx.find(')', openParen);
        if (closeParen == std::string::npos) return false;

        std::string paramBlock = ptx.substr(openParen, closeParen - openParen);
        size_t params = 0;
        size_t paramPos = 0;
        while ((paramPos = paramBlock.find(".param", paramPos)) != std::string::npos) {
            bool validBefore = (paramPos == 0) ||
                std::isspace(static_cast<unsigned char>(paramBlock[paramPos - 1])) ||
                paramBlock[paramPos - 1] == '(' ||
                paramBlock[paramPos - 1] == ',';
            size_t afterPos = paramPos + 6;
            bool validAfter = (afterPos < paramBlock.size()) &&
                std::isspace(static_cast<unsigned char>(paramBlock[afterPos]));
            if (validBefore && validAfter) {
                ++params;
            }
            paramPos += 6;
        }
        count = params;
        return true;
    }
}

enum class ArgType {
    I32,
    F32,
    BUF_I32,
    BUF_F32
};

struct ParsedArgSpec {
    ArgType type = ArgType::I32;
    int32_t i32Val = 0;
    float f32Val = 0.0f;
    size_t elemCount = 0;
    std::string initMode;
};

struct KernelArg {
    ArgType type = ArgType::I32;
    int32_t i32Val = 0;
    float f32Val = 0.0f;
    CUdeviceptr dPtr = 0;
    size_t elemCount = 0;

    KernelArg() = default;
    ~KernelArg() {
        if (dPtr) {
            cuMemFree(dPtr);
            dPtr = 0;
        }
    }
    KernelArg(const KernelArg&) = delete;
    KernelArg& operator=(const KernelArg&) = delete;
    KernelArg(KernelArg&& other) noexcept
        : type(other.type), i32Val(other.i32Val), f32Val(other.f32Val),
          dPtr(other.dPtr), elemCount(other.elemCount) {
        other.dPtr = 0;
    }
    KernelArg& operator=(KernelArg&& other) noexcept {
        if (this != &other) {
            if (dPtr) cuMemFree(dPtr);
            type = other.type;
            i32Val = other.i32Val;
            f32Val = other.f32Val;
            dPtr = other.dPtr;
            elemCount = other.elemCount;
            other.dPtr = 0;
        }
        return *this;
    }
};

struct DumpSpec {
    size_t argIndex = 0;
    bool hasCount = false;
    size_t count = 0;
};

static bool parseI32(const std::string& valStr, int32_t& outVal) {
    if (valStr.empty()) return false;
    for (char c : valStr) {
        if (std::isspace(static_cast<unsigned char>(c))) return false;
    }

    const char* str = valStr.c_str();
    bool isNegative = (str[0] == '-');
    const char* p = str;
    if (*p == '-' || *p == '+') {
        ++p;
    }
    if (*p == '\0') return false;

    bool isHex = (p[0] == '0' && (p[1] == 'x' || p[1] == 'X'));
    char* end = nullptr;
    errno = 0;

    if (isHex) {
        if (isNegative) {
            long long val = std::strtoll(str, &end, 16);
            if (errno == ERANGE || end != str + valStr.size() || val < INT32_MIN || val > INT32_MAX) {
                return false;
            }
            outVal = static_cast<int32_t>(val);
        } else {
            unsigned long long val = std::strtoull(p, &end, 16);
            if (errno == ERANGE || end != p + std::strlen(p) || val > 0xFFFFFFFFULL) {
                return false;
            }
            outVal = static_cast<int32_t>(static_cast<uint32_t>(val));
        }
    } else {
        // Decimal: strictly base 10 (prevents accidental octal interpretation of leading zeros e.g. 08, 010)
        long long val = std::strtoll(str, &end, 10);
        if (errno == ERANGE || end != str + valStr.size() || val < INT32_MIN || val > INT32_MAX) {
            return false;
        }
        outVal = static_cast<int32_t>(val);
    }
    return true;
}

static bool parseF32(const std::string& valStr, float& outVal) {
    if (valStr.empty()) return false;
    for (char c : valStr) {
        if (std::isspace(static_cast<unsigned char>(c))) return false;
    }
    try {
        size_t pos = 0;
        outVal = std::stof(valStr, &pos);
        return pos == valStr.size();
    } catch (...) {
        return false;
    }
}

static void printHelp(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " [options] <ptx_file> <kernel_name>\n"
              << "  " << prog << " [options] --ptx <ptx_file> --kernel <kernel_name>\n\n"
              << "Launch compiled PTX kernels through CUDA Driver API.\n\n"
              << "Options:\n"
              << "  -p, --ptx <path>            Path to PTX file\n"
              << "  -k, --kernel <name>         Kernel function name\n"
              << "  -g, --grid <num>            Grid size (default: 1)\n"
              << "  -b, --block <num>           Block size (default: 256)\n"
              << "  --arg <spec>                Kernel argument in call order:\n"
              << "                                i32:<val>           (e.g. i32:1024, i32:0xFFFFFFFF)\n"
              << "                                f32:<val>           (e.g. f32:2.5)\n"
              << "                                buf:<type>:<N>:<init>\n"
              << "                                  <type>: i32 or f32\n"
              << "                                  <N>:    element count (> 0)\n"
              << "                                  <init>: zeros, rand, or seq\n"
              << "  --dump <index>[:<count>]    Dump first <count> elements of buffer arg <index>\n"
              << "                              after execution (default: min(N, 16))\n"
              << "  -v, --verbose               Enable verbose Driver JIT logging\n"
              << "  -h, --help                  Display this help message\n\n"
              << "Examples:\n"
              << "  " << prog << " vecadd.ptx vecadd --grid 4 --block 256 \\\n"
              << "      --arg buf:i32:1024:seq --arg buf:i32:1024:seq --arg buf:i32:1024:zeros --arg i32:1024 \\\n"
              << "      --dump 2:8\n\n"
              << "  " << prog << " saxpy.ptx saxpy --grid 4 --block 256 \\\n"
              << "      --arg f32:2.5 --arg buf:f32:1024:seq --arg buf:f32:1024:zeros --arg i32:1024 \\\n"
              << "      --dump 2:8\n" << std::flush;
}

struct Dim3 {
    int x = 1;
    int y = 1;
    int z = 1;
};

static bool parseDim3(const std::string& str, Dim3& out) {
    std::string s = str;
    for (char& c : s) {
        if (c == 'x' || c == 'X') c = ',';
    }
    std::stringstream ss(s);
    std::string item;
    std::vector<int> vals;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) return false;
        try {
            size_t p = 0;
            int v = std::stoi(item, &p);
            if (p != item.size() || v <= 0) return false;
            vals.push_back(v);
        } catch (...) {
            return false;
        }
    }
    if (vals.size() == 1) {
        out = {vals[0], 1, 1};
        return true;
    } else if (vals.size() == 2) {
        out = {vals[0], vals[1], 1};
        return true;
    } else if (vals.size() == 3) {
        out = {vals[0], vals[1], vals[2]};
        return true;
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc <= 1) {
        printHelp(argv[0]);
        return 1;
    }

    try {
        std::string ptxPath;
        std::string kernelName;
        Dim3 gridDim{1, 1, 1};
        Dim3 blockDim{256, 1, 1};
        bool verbose = false;
        std::vector<std::string> rawArgSpecs;
        std::vector<DumpSpec> dumpSpecs;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "-h" || arg == "--help") {
                printHelp(argv[0]);
                return 0;
            } else if (arg == "-v" || arg == "--verbose") {
                verbose = true;
            } else if (arg == "-p" || arg == "--ptx") {
                if (i + 1 >= argc) {
                    std::cerr << "Error: " << arg << " requires a path argument.\n";
                    return 1;
                }
                ptxPath = argv[++i];
            } else if (arg.rfind("--ptx=", 0) == 0) {
                ptxPath = arg.substr(6);
            } else if (arg == "-k" || arg == "--kernel") {
                if (i + 1 >= argc) {
                    std::cerr << "Error: " << arg << " requires a kernel name argument.\n";
                    return 1;
                }
                kernelName = argv[++i];
            } else if (arg.rfind("--kernel=", 0) == 0) {
                kernelName = arg.substr(9);
            } else if (arg == "-g" || arg == "--grid") {
                if (i + 1 >= argc) {
                    std::cerr << "Error: " << arg << " requires an integer value.\n";
                    return 1;
                }
                std::string val = argv[++i];
                if (!parseDim3(val, gridDim)) {
                    std::cerr << "Error: Invalid grid size '" << val << "'.\n";
                    return 1;
                }
            } else if (arg.rfind("--grid=", 0) == 0) {
                std::string val = arg.substr(7);
                if (!parseDim3(val, gridDim)) {
                    std::cerr << "Error: Invalid grid size '" << val << "'.\n";
                    return 1;
                }
            } else if (arg == "-b" || arg == "--block") {
                if (i + 1 >= argc) {
                    std::cerr << "Error: " << arg << " requires an integer value.\n";
                    return 1;
                }
                std::string val = argv[++i];
                if (!parseDim3(val, blockDim)) {
                    std::cerr << "Error: Invalid block size '" << val << "'.\n";
                    return 1;
                }
            } else if (arg.rfind("--block=", 0) == 0) {
                std::string val = arg.substr(8);
                if (!parseDim3(val, blockDim)) {
                    std::cerr << "Error: Invalid block size '" << val << "'.\n";
                    return 1;
                }
            } else if (arg == "--arg") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --arg requires a specification.\n";
                return 1;
            }
            rawArgSpecs.push_back(argv[++i]);
        } else if (arg.rfind("--arg=", 0) == 0) {
            rawArgSpecs.push_back(arg.substr(6));
        } else if (arg == "--dump" || arg.rfind("--dump=", 0) == 0) {
            std::string dumpStr;
            if (arg == "--dump") {
                if (i + 1 >= argc) {
                    std::cerr << "Error: --dump requires a specification.\n";
                    return 1;
                }
                dumpStr = argv[++i];
            } else {
                dumpStr = arg.substr(7);
            }

            DumpSpec ds;
            auto colonPos = dumpStr.find(':');
            if (colonPos == std::string::npos) {
                if (dumpStr.empty() || dumpStr[0] == '-' || dumpStr[0] == '+' || !std::isdigit(static_cast<unsigned char>(dumpStr[0]))) {
                    std::cerr << "Error: Invalid --dump index '" << dumpStr << "'.\n";
                    return 1;
                }
                char* end = nullptr;
                errno = 0;
                unsigned long long idx = std::strtoull(dumpStr.c_str(), &end, 10);
                if (errno == ERANGE || end != dumpStr.c_str() + dumpStr.size()) {
                    std::cerr << "Error: Invalid --dump index '" << dumpStr << "'.\n";
                    return 1;
                }
                ds.argIndex = static_cast<size_t>(idx);
                ds.hasCount = false;
                ds.count = 0;
            } else {
                std::string idxStr = dumpStr.substr(0, colonPos);
                std::string cntStr = dumpStr.substr(colonPos + 1);
                if (idxStr.empty() || idxStr[0] == '-' || idxStr[0] == '+' || !std::isdigit(static_cast<unsigned char>(idxStr[0]))) {
                    std::cerr << "Error: Invalid --dump index '" << idxStr << "'.\n";
                    return 1;
                }
                if (cntStr.empty() || cntStr[0] == '-' || cntStr[0] == '+' || !std::isdigit(static_cast<unsigned char>(cntStr[0]))) {
                    std::cerr << "Error: Invalid --dump count '" << cntStr << "'.\n";
                    return 1;
                }
                char* end1 = nullptr;
                char* end2 = nullptr;
                errno = 0;
                unsigned long long idx = std::strtoull(idxStr.c_str(), &end1, 10);
                if (errno == ERANGE || end1 != idxStr.c_str() + idxStr.size()) {
                    std::cerr << "Error: Invalid --dump index '" << idxStr << "'.\n";
                    return 1;
                }
                errno = 0;
                unsigned long long cnt = std::strtoull(cntStr.c_str(), &end2, 10);
                if (errno == ERANGE || end2 != cntStr.c_str() + cntStr.size()) {
                    std::cerr << "Error: Invalid --dump count '" << cntStr << "'.\n";
                    return 1;
                }
                ds.argIndex = static_cast<size_t>(idx);
                ds.count = static_cast<size_t>(cnt);
                ds.hasCount = true;
            }
            dumpSpecs.push_back(ds);
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Error: Unknown option '" << arg << "'.\n";
            printHelp(argv[0]);
            return 1;
        } else {
            if (ptxPath.empty()) {
                ptxPath = arg;
            } else if (kernelName.empty()) {
                kernelName = arg;
            } else {
                std::cerr << "Error: Unexpected positional argument '" << arg << "'.\n";
                return 1;
            }
        }
    }

    if (ptxPath.empty()) {
        std::cerr << "Error: No PTX file specified.\n";
        printHelp(argv[0]);
        return 1;
    }
    if (kernelName.empty()) {
        std::cerr << "Error: No kernel function name specified.\n";
        printHelp(argv[0]);
        return 1;
    }
    if (gridDim.x <= 0 || gridDim.y <= 0 || gridDim.z <= 0) {
        std::cerr << "Error: Grid dimensions must be positive.\n";
        return 1;
    }
    if (blockDim.x <= 0 || blockDim.y <= 0 || blockDim.z <= 0) {
        std::cerr << "Error: Block dimensions must be positive.\n";
        return 1;
    }

    // Pre-parse and validate all argument specifications before initializing CUDA
    std::vector<ParsedArgSpec> parsedSpecs;
    parsedSpecs.reserve(rawArgSpecs.size());

    for (size_t i = 0; i < rawArgSpecs.size(); ++i) {
        const std::string& spec = rawArgSpecs[i];
        ParsedArgSpec pSpec;

        if (spec.rfind("i32:", 0) == 0) {
            pSpec.type = ArgType::I32;
            std::string valStr = spec.substr(4);
            if (!parseI32(valStr, pSpec.i32Val)) {
                std::cerr << "Error: Invalid i32 value in argument " << i << " ('" << spec << "')\n";
                return 1;
            }
        } else if (spec.rfind("f32:", 0) == 0) {
            pSpec.type = ArgType::F32;
            std::string valStr = spec.substr(4);
            if (!parseF32(valStr, pSpec.f32Val)) {
                std::cerr << "Error: Invalid f32 value in argument " << i << " ('" << spec << "')\n";
                return 1;
            }
        } else if (spec.rfind("buf:", 0) == 0) {
            std::stringstream ss(spec.substr(4));
            std::string typeStr, countStr, initStr, extraStr;
            if (!std::getline(ss, typeStr, ':') ||
                !std::getline(ss, countStr, ':') ||
                !std::getline(ss, initStr, ':') ||
                std::getline(ss, extraStr, ':')) {
                std::cerr << "Error: Invalid buffer specification '" << spec
                          << "' in argument " << i << ". Expected buf:<i32|f32>:<N>:<init>\n";
                return 1;
            }

            if (typeStr == "i32") {
                pSpec.type = ArgType::BUF_I32;
            } else if (typeStr == "f32") {
                pSpec.type = ArgType::BUF_F32;
            } else {
                std::cerr << "Error: Invalid buffer type '" << typeStr << "' in argument " << i
                          << ". Expected 'i32' or 'f32'.\n";
                return 1;
            }

            size_t elemSize = (pSpec.type == ArgType::BUF_I32) ? sizeof(int32_t) : sizeof(float);
            if (countStr.empty() || countStr[0] == '-' || countStr[0] == '+' || !std::isdigit(static_cast<unsigned char>(countStr[0]))) {
                std::cerr << "Error: Invalid buffer element count '" << countStr << "' in argument " << i
                          << " (must be positive integer).\n";
                return 1;
            }

            char* end = nullptr;
            errno = 0;
            unsigned long long cnt = std::strtoull(countStr.c_str(), &end, 10);
            if (errno == ERANGE || end != countStr.c_str() + countStr.size() || cnt == 0 || cnt > (SIZE_MAX / elemSize)) {
                std::cerr << "Error: Invalid buffer element count '" << countStr << "' in argument " << i
                          << " (must be positive integer).\n";
                return 1;
            }
            pSpec.elemCount = static_cast<size_t>(cnt);

            if (initStr != "zeros" && initStr != "zero" && initStr != "rand" && initStr != "seq") {
                std::cerr << "Error: Unknown buffer init mode '" << initStr << "' in argument " << i
                          << ". Expected 'zeros', 'rand', or 'seq'.\n";
                return 1;
            }
            pSpec.initMode = initStr;
        } else {
            std::cerr << "Error: Unrecognized argument specification '" << spec
                      << "' for argument " << i
                      << ". Expected i32:<val>, f32:<val>, or buf:<i32|f32>:<N>:<init>.\n";
            return 1;
        }

        parsedSpecs.push_back(std::move(pSpec));
    }

    // Validate dump specifications against parsed arguments before initializing CUDA
    for (const auto& dump : dumpSpecs) {
        if (dump.argIndex >= parsedSpecs.size()) {
            std::cerr << "Error: --dump argument index " << dump.argIndex
                      << " out of range (kernel has " << parsedSpecs.size() << " arguments).\n";
            return 1;
        }
        const auto& dArg = parsedSpecs[dump.argIndex];
        if (dArg.type != ArgType::BUF_I32 && dArg.type != ArgType::BUF_F32) {
            std::cerr << "Error: --dump argument index " << dump.argIndex
                      << " refers to a scalar argument, not a buffer.\n";
            return 1;
        }
    }

    std::string ptxSource;
    try {
        std::error_code ec;
        if (std::filesystem::is_directory(ptxPath, ec)) {
            std::cerr << "Error: Specified PTX path '" << ptxPath << "' is a directory.\n";
            return 1;
        }

        std::ifstream in(ptxPath, std::ios::in | std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "Error: Failed to open PTX file: " << ptxPath << "\n";
            return 1;
        }
        in.seekg(0, std::ios::end);
        std::streampos endPos = in.tellg();
        if (endPos < 0) {
            std::cerr << "Error: Failed to read PTX file '" << ptxPath << "'\n";
            return 1;
        }
        in.seekg(0, std::ios::beg);
        ptxSource.resize(static_cast<size_t>(endPos));
        if (endPos > 0) {
            in.read(&ptxSource[0], endPos);
            if (!in) {
                std::cerr << "Error: Failed to read PTX file '" << ptxPath << "'\n";
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to read PTX file '" << ptxPath << "': " << e.what() << "\n";
        return 1;
    }

    size_t expectedParamCount = 0;
    if (getExpectedParamCountFromPTX(ptxSource, kernelName, expectedParamCount)) {
        if (rawArgSpecs.size() != expectedParamCount) {
            std::cerr << "Error: Kernel '" << kernelName << "' expects " << expectedParamCount
                      << " argument(s), but " << rawArgSpecs.size() << " were provided.\n";
            return 1;
        }
    }

    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "CUDA Error (cuInit): " << getCudaErrorString(res) << "\n";
        return 1;
    }

    CUdevice dev;
    res = cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "CUDA Error (cuDeviceGet): " << getCudaErrorString(res) << "\n";
        return 1;
    }

    char devName[128] = {0};
    res = cuDeviceGetName(devName, sizeof(devName), dev);
    if (res != CUDA_SUCCESS) {
        std::cerr << "CUDA Error (cuDeviceGetName): " << getCudaErrorString(res) << "\n";
        return 1;
    }
    std::cout << "[Runner] Loaded GPU: " << devName << "\n" << std::flush;

    CUcontext ctx = nullptr;
    res = cuCtxCreate(&ctx, nullptr, 0, dev);
    if (res != CUDA_SUCCESS) {
        std::cerr << "CUDA Error (cuCtxCreate): " << getCudaErrorString(res) << "\n";
        return 1;
    }
    auto ctxGuard = makeScopeExit([&]() {
        if (ctx) {
            cuCtxDestroy(ctx);
            ctx = nullptr;
        }
    });

    std::vector<char> infoLog(16385, 0);
    std::vector<char> errorLog(16385, 0);
    uint32_t logCapacity = 16384;
    std::vector<CUjit_option> opts = {
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    };
    std::vector<void*> optVals = {
        infoLog.data(),
        reinterpret_cast<void*>(static_cast<uintptr_t>(logCapacity)),
        errorLog.data(),
        reinterpret_cast<void*>(static_cast<uintptr_t>(logCapacity))
    };
    if (verbose) {
        opts.push_back(CU_JIT_LOG_VERBOSE);
        optVals.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(1)));
    }

    CUmodule mod = nullptr;
    res = cuModuleLoadDataEx(&mod, ptxSource.c_str(), static_cast<unsigned int>(opts.size()), opts.data(), optVals.data());
    if (res != CUDA_SUCCESS) {
        std::cerr << "CUDA Error (cuModuleLoadDataEx): " << getCudaErrorString(res) << "\n";
        if (errorLog[0] != '\0') {
            std::cerr << "JIT Error Log:\n" << errorLog.data() << "\n";
        }
        if (infoLog[0] != '\0') {
            std::cerr << "JIT Info Log:\n" << infoLog.data() << "\n";
        }
        return 1;
    }
    auto modGuard = makeScopeExit([&]() {
        if (mod) {
            cuModuleUnload(mod);
            mod = nullptr;
        }
    });

    if (verbose) {
        std::cout << "[Runner] JIT compilation successful.\n";
        if (infoLog[0] != '\0') {
            std::cout << "[Runner] JIT Info Log:\n" << infoLog.data() << "\n";
        }
        if (errorLog[0] != '\0') {
            std::cout << "[Runner] JIT Error Log:\n" << errorLog.data() << "\n";
        }
        std::cout << std::flush;
    }

    CUfunction func;
    res = cuModuleGetFunction(&func, mod, kernelName.c_str());
    if (res != CUDA_SUCCESS) {
        std::cerr << "CUDA Error (cuModuleGetFunction for '" << kernelName << "'): "
                  << getCudaErrorString(res) << "\n";
        return 1;
    }
    std::cout << "[Runner] Found kernel: " << kernelName << "\n" << std::flush;

    std::vector<KernelArg> kernelArgs;
    kernelArgs.reserve(parsedSpecs.size());

    for (size_t i = 0; i < parsedSpecs.size(); ++i) {
        const auto& pSpec = parsedSpecs[i];
        KernelArg arg;
        arg.type = pSpec.type;

        if (pSpec.type == ArgType::I32) {
            arg.i32Val = pSpec.i32Val;
        } else if (pSpec.type == ArgType::F32) {
            arg.f32Val = pSpec.f32Val;
        } else {
            arg.elemCount = pSpec.elemCount;
            size_t elemSize = (arg.type == ArgType::BUF_I32) ? sizeof(int32_t) : sizeof(float);
            size_t sizeBytes = arg.elemCount * elemSize;

            res = cuMemAlloc(&arg.dPtr, sizeBytes);
            if (res != CUDA_SUCCESS) {
                std::cerr << "CUDA Error (cuMemAlloc for argument " << i << "): "
                          << getCudaErrorString(res) << "\n";
                return 1;
            }

            std::mt19937 rng(42 + static_cast<unsigned int>(i));
            if (arg.type == ArgType::BUF_I32) {
                std::vector<int32_t> hostData;
                try {
                    hostData.assign(arg.elemCount, 0);
                } catch (const std::exception& e) {
                    std::cerr << "Error: Host memory allocation failed for argument " << i << ": " << e.what() << "\n";
                    return 1;
                }

                if (pSpec.initMode == "seq") {
                    for (size_t idx = 0; idx < arg.elemCount; ++idx) {
                        hostData[idx] = static_cast<int32_t>(idx);
                    }
                } else if (pSpec.initMode == "rand") {
                    std::uniform_int_distribution<int32_t> dist(-1000, 1000);
                    for (size_t idx = 0; idx < arg.elemCount; ++idx) {
                        hostData[idx] = dist(rng);
                    }
                }
                res = cuMemcpyHtoD(arg.dPtr, hostData.data(), sizeBytes);
                if (res != CUDA_SUCCESS) {
                    std::cerr << "CUDA Error (cuMemcpyHtoD for argument " << i << "): "
                              << getCudaErrorString(res) << "\n";
                    return 1;
                }
            } else {
                std::vector<float> hostData;
                try {
                    hostData.assign(arg.elemCount, 0.0f);
                } catch (const std::exception& e) {
                    std::cerr << "Error: Host memory allocation failed for argument " << i << ": " << e.what() << "\n";
                    return 1;
                }

                if (pSpec.initMode == "seq") {
                    for (size_t idx = 0; idx < arg.elemCount; ++idx) {
                        hostData[idx] = static_cast<float>(idx);
                    }
                } else if (pSpec.initMode == "rand") {
                    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
                    for (size_t idx = 0; idx < arg.elemCount; ++idx) {
                        hostData[idx] = dist(rng);
                    }
                }
                res = cuMemcpyHtoD(arg.dPtr, hostData.data(), sizeBytes);
                if (res != CUDA_SUCCESS) {
                    std::cerr << "CUDA Error (cuMemcpyHtoD for argument " << i << "): "
                              << getCudaErrorString(res) << "\n";
                    return 1;
                }
            }
        }

        kernelArgs.push_back(std::move(arg));
    }

    std::vector<void*> paramPointers;
    paramPointers.reserve(kernelArgs.size());
    for (auto& kArg : kernelArgs) {
        if (kArg.type == ArgType::I32) {
            paramPointers.push_back(&kArg.i32Val);
        } else if (kArg.type == ArgType::F32) {
            paramPointers.push_back(&kArg.f32Val);
        } else {
            paramPointers.push_back(&kArg.dPtr);
        }
    }

    std::cout << "[Runner] Launching kernel: grid=";
    if (gridDim.y == 1 && gridDim.z == 1) {
        std::cout << gridDim.x;
    } else {
        std::cout << gridDim.x << "x" << gridDim.y << "x" << gridDim.z;
    }
    std::cout << " block=";
    if (blockDim.y == 1 && blockDim.z == 1) {
        std::cout << blockDim.x;
    } else {
        std::cout << blockDim.x << "x" << blockDim.y << "x" << blockDim.z;
    }
    std::cout << " args=" << kernelArgs.size() << "\n" << std::flush;

    void** kernelParams = paramPointers.empty() ? nullptr : paramPointers.data();
    res = cuLaunchKernel(
        func,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0,
        nullptr,
        kernelParams,
        nullptr
    );
    if (res != CUDA_SUCCESS) {
        std::cerr << "CUDA Error (cuLaunchKernel): " << getCudaErrorString(res) << "\n";
        return 1;
    }

    res = cuCtxSynchronize();
    if (res != CUDA_SUCCESS) {
        std::cerr << "CUDA Error (cuCtxSynchronize): " << getCudaErrorString(res) << "\n";
        return 1;
    }
    std::cout << "[Runner] Kernel execution completed successfully.\n" << std::flush;

    for (const auto& dump : dumpSpecs) {
        const auto& dArg = kernelArgs[dump.argIndex];
        size_t count = dump.hasCount ? dump.count : std::min(dArg.elemCount, static_cast<size_t>(16));
        if (count > dArg.elemCount) {
            count = dArg.elemCount;
        }

        std::cout << "[Runner] Dump arg " << dump.argIndex << " ("
                  << (dArg.type == ArgType::BUF_I32 ? "i32" : "f32")
                  << ", count=" << count << "):\n";

        if (count > 0) {
            if (dArg.type == ArgType::BUF_I32) {
                std::vector<int32_t> dumpData(count);
                res = cuMemcpyDtoH(dumpData.data(), dArg.dPtr, count * sizeof(int32_t));
                if (res != CUDA_SUCCESS) {
                    std::cerr << "CUDA Error (cuMemcpyDtoH for dump arg " << dump.argIndex << "): "
                              << getCudaErrorString(res) << "\n";
                    return 1;
                }
                for (size_t idx = 0; idx < count; ++idx) {
                    std::cout << "  [" << idx << "] = " << dumpData[idx] << "\n";
                }
            } else {
                std::vector<float> dumpData(count);
                res = cuMemcpyDtoH(dumpData.data(), dArg.dPtr, count * sizeof(float));
                if (res != CUDA_SUCCESS) {
                    std::cerr << "CUDA Error (cuMemcpyDtoH for dump arg " << dump.argIndex << "): "
                              << getCudaErrorString(res) << "\n";
                    return 1;
                }
                for (size_t idx = 0; idx < count; ++idx) {
                    std::cout << "  [" << idx << "] = " << dumpData[idx] << "\n";
                }
            }
        }
        std::cout << std::flush;
    }

        std::cout << "[Runner] Success!\n" << std::flush;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Error: Unknown exception encountered.\n";
        return 1;
    }
}
