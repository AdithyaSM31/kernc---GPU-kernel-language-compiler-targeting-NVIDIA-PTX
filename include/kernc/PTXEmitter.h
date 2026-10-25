#pragma once

#include "kernc/IR.h"
#include <string>
#include <unordered_map>

namespace kernc {

struct PTXOptions {
    bool lineInfo = false;  // emit .file/.loc source line info (like nvcc -lineinfo)
    std::string sourceFile; // file name recorded by the .file directive
};

class PTXEmitter {
public:
    explicit PTXEmitter(PTXOptions options = {}) : options_(std::move(options)) {}

    std::string emitModule(const IRModule& mod);
    std::string emitFunction(const IRFunction& fn);

private:
    struct RegAllocator {
        std::unordered_map<VReg, std::string> vregToName;
        int countR = 0;
        int countF = 0;
        int countRD = 0;
        int countP = 0;

        std::string getOrAlloc(VReg vreg, IRType type);
        std::string get(VReg vreg) const;
    };

    std::string formatFloatHex(float val) const;

    PTXOptions options_;
};

} // namespace kernc
