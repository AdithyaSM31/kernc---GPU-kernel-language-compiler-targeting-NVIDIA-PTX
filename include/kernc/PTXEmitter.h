#pragma once

#include "kernc/IR.h"
#include <string>
#include <unordered_map>

namespace kernc {

class PTXEmitter {
public:
    PTXEmitter() = default;

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
};

} // namespace kernc
