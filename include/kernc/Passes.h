#pragma once

#include "kernc/IR.h"
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace kernc {

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

    static bool runConstantFolding(IRFunction& fn);
    static bool runCSE(IRFunction& fn);
    static bool runDeadCodeElimination(IRFunction& fn);
    static bool runInstructionScheduling(IRFunction& fn);
};

} // namespace kernc

