#include "kernc/Passes.h"
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace kernc {

struct ConstVal {
    bool isFloat = false;
    int32_t intVal = 0;
    float floatVal = 0.0f;
};

bool Optimizer::runConstantFolding(IRFunction& fn) {
    bool changed = false;

    // 1. Count definitions for each register
    std::unordered_map<VReg, int> defCount;
    for (const auto& bb : fn.blocks) {
        for (const auto& inst : bb.instructions) {
            if (inst.dst != 0) {
                defCount[inst.dst]++;
            }
        }
    }

    // 2. Identify global constants (defined exactly once by LoadImm)
    std::unordered_map<VReg, ConstVal> constTable;
    for (const auto& bb : fn.blocks) {
        for (const auto& inst : bb.instructions) {
            if (inst.op == IROp::LoadImm && inst.dst != 0 && defCount[inst.dst] == 1) {
                ConstVal cv;
                cv.isFloat = (inst.type == IRType::F32);
                cv.intVal = inst.immInt;
                cv.floatVal = inst.immFloat;
                constTable[inst.dst] = cv;
            }
        }
    }

    // 3. Fold instructions
    for (auto& bb : fn.blocks) {
        for (auto& inst : bb.instructions) {
            bool src1IsConst = (constTable.find(inst.src1) != constTable.end());
            bool src2IsConst = (constTable.find(inst.src2) != constTable.end());

            // Both sources are constant
            if (src1IsConst && src2IsConst && inst.dst != 0) {
                const auto& c1 = constTable[inst.src1];
                const auto& c2 = constTable[inst.src2];

                if (!c1.isFloat && !c2.isFloat) {
                    bool folded = true;
                    int32_t res = 0;
                    switch (inst.op) {
                        case IROp::Add: res = c1.intVal + c2.intVal; break;
                        case IROp::Sub: res = c1.intVal - c2.intVal; break;
                        case IROp::Mul: res = c1.intVal * c2.intVal; break;
                        case IROp::Div:
                            if (c2.intVal != 0) res = c1.intVal / c2.intVal;
                            else folded = false;
                            break;
                        case IROp::Mod:
                            if (c2.intVal != 0) res = c1.intVal % c2.intVal;
                            else folded = false;
                            break;
                        default:
                            folded = false;
                            break;
                    }
                    if (folded) {
                        inst.op = IROp::LoadImm;
                        inst.src1 = 0;
                        inst.src2 = 0;
                        inst.type = IRType::S32;
                        inst.immInt = res;
                        constTable[inst.dst] = ConstVal{false, res, 0.0f};
                        changed = true;
                        continue;
                    }
                } else if (c1.isFloat && c2.isFloat) {
                    bool folded = true;
                    float res = 0.0f;
                    switch (inst.op) {
                        case IROp::FAdd: res = c1.floatVal + c2.floatVal; break;
                        case IROp::FSub: res = c1.floatVal - c2.floatVal; break;
                        case IROp::FMul: res = c1.floatVal * c2.floatVal; break;
                        case IROp::FDiv:
                            if (c2.floatVal != 0.0f) res = c1.floatVal / c2.floatVal;
                            else folded = false;
                            break;
                        default:
                            folded = false;
                            break;
                    }
                    if (folded) {
                        inst.op = IROp::LoadImm;
                        inst.src1 = 0;
                        inst.src2 = 0;
                        inst.type = IRType::F32;
                        inst.immFloat = res;
                        constTable[inst.dst] = ConstVal{true, 0, res};
                        changed = true;
                        continue;
                    }
                }
            }

            // Single source constant folding (Neg, FNeg, Cvt)
            if (src1IsConst && inst.src2 == 0 && inst.dst != 0) {
                const auto& c1 = constTable[inst.src1];
                if (inst.op == IROp::Neg && !c1.isFloat) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.immInt = -c1.intVal;
                    constTable[inst.dst] = ConstVal{false, inst.immInt, 0.0f};
                    changed = true;
                    continue;
                } else if (inst.op == IROp::FNeg && c1.isFloat) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.immFloat = -c1.floatVal;
                    constTable[inst.dst] = ConstVal{true, 0, inst.immFloat};
                    changed = true;
                    continue;
                } else if (inst.op == IROp::CvtI2F && !c1.isFloat) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.type = IRType::F32;
                    inst.immFloat = static_cast<float>(c1.intVal);
                    constTable[inst.dst] = ConstVal{true, 0, inst.immFloat};
                    changed = true;
                    continue;
                } else if (inst.op == IROp::CvtF2I && c1.isFloat) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.type = IRType::S32;
                    inst.immInt = static_cast<int32_t>(c1.floatVal);
                    constTable[inst.dst] = ConstVal{false, inst.immInt, 0.0f};
                    changed = true;
                    continue;
                }
            }

            // Algebraic identities
            if (src2IsConst && !constTable[inst.src2].isFloat) {
                int32_t val = constTable[inst.src2].intVal;
                if ((inst.op == IROp::Add || inst.op == IROp::Sub) && val == 0) {
                    inst.op = IROp::Mov;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
                if (inst.op == IROp::Mul && val == 1) {
                    inst.op = IROp::Mov;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
                if (inst.op == IROp::Mul && val == 0) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.src2 = 0;
                    inst.immInt = 0;
                    constTable[inst.dst] = ConstVal{false, 0, 0.0f};
                    changed = true;
                    continue;
                }
            }
            if (src1IsConst && !constTable[inst.src1].isFloat) {
                int32_t val = constTable[inst.src1].intVal;
                if (inst.op == IROp::Add && val == 0) {
                    inst.op = IROp::Mov;
                    inst.src1 = inst.src2;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
                if (inst.op == IROp::Mul && val == 1) {
                    inst.op = IROp::Mov;
                    inst.src1 = inst.src2;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
                if (inst.op == IROp::Mul && val == 0) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.src2 = 0;
                    inst.immInt = 0;
                    constTable[inst.dst] = ConstVal{false, 0, 0.0f};
                    changed = true;
                    continue;
                }
            }
        }
    }

    return changed;
}

bool Optimizer::runDeadCodeElimination(IRFunction& fn) {
    bool changedOverall = false;

    // First: remove unreachable instructions within blocks (after Bra or Ret)
    for (auto& bb : fn.blocks) {
        for (size_t i = 0; i < bb.instructions.size(); ++i) {
            if (bb.instructions[i].op == IROp::Bra || bb.instructions[i].op == IROp::Ret) {
                if (i + 1 < bb.instructions.size()) {
                    bb.instructions.erase(bb.instructions.begin() + i + 1, bb.instructions.end());
                    changedOverall = true;
                }
                break;
            }
        }
    }

    // Iterative dead instruction removal
    bool changed = true;
    while (changed) {
        changed = false;

        // Count uses of all registers
        std::unordered_map<VReg, int> useCount;
        for (const auto& bb : fn.blocks) {
            for (const auto& inst : bb.instructions) {
                if (inst.src1 != 0) useCount[inst.src1]++;
                if (inst.src2 != 0) useCount[inst.src2]++;
            }
        }

        for (auto& bb : fn.blocks) {
            for (auto it = bb.instructions.begin(); it != bb.instructions.end(); ) {
                // Side-effecting operations
                if (it->op == IROp::Store || it->op == IROp::Bra ||
                    it->op == IROp::BraCond || it->op == IROp::Ret ||
                    it->op == IROp::Bar) {
                    ++it;
                    continue;
                }

                if (it->dst != 0 && useCount[it->dst] == 0) {
                    // Instruction defines a register that is never used
                    it = bb.instructions.erase(it);
                    changed = true;
                    changedOverall = true;
                } else {
                    ++it;
                }
            }
        }
    }

    return changedOverall;
}

bool Optimizer::runPasses(IRFunction& fn) {
    bool changed = false;
    // Iterate passes until convergence
    for (int iter = 0; iter < 10; ++iter) {
        bool cfChanged = runConstantFolding(fn);
        bool dceChanged = runDeadCodeElimination(fn);
        if (cfChanged || dceChanged) {
            changed = true;
        } else {
            break;
        }
    }
    return changed;
}

bool Optimizer::runPasses(IRModule& mod) {
    bool changed = false;
    for (auto& fn : mod.functions) {
        if (runPasses(fn)) {
            changed = true;
        }
    }
    return changed;
}

} // namespace kernc
