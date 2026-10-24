#pragma once

#include "kernc/IR.h"

namespace kernc {

class Optimizer {
public:
    static bool runPasses(IRFunction& fn);
    static bool runPasses(IRModule& mod);

    static bool runConstantFolding(IRFunction& fn);
    static bool runDeadCodeElimination(IRFunction& fn);
};

} // namespace kernc
