#pragma once

#include "kernc/IR.h"
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace kernc {

namespace ptx {
struct MagicInfo {
    int32_t M;
    uint32_t s;
};

MagicInfo computeSignedMagic(int32_t d);
int32_t mulHiS32(int32_t a, int32_t b);
} // namespace ptx

struct CallGraph {
    std::unordered_map<std::string, std::unordered_set<std::string>> callersToCallees;

    static CallGraph build(const IRModule& mod);
    bool isRecursive(const std::string& funcName) const;
};

class Optimizer {
public:
    static bool runPasses(IRFunction& fn);
    static bool runPasses(IRModule& mod);

    static bool runInlining(IRModule& mod, int threshold = 64);

    static bool runCFGCleanup(IRFunction& fn);
    static bool runConstantFolding(IRFunction& fn);
    static bool runCSE(IRFunction& fn);
    static bool runDeadCodeElimination(IRFunction& fn);
    static bool runInstructionScheduling(IRFunction& fn);
};

} // namespace kernc

