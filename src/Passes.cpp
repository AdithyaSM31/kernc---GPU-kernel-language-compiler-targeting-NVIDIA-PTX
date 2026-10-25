#include "kernc/Passes.h"
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace kernc {

// Folding must reproduce what the PTX instruction computes on the GPU, not what C++
// computes on the host. These helpers are the target semantics, one per instruction.
namespace ptx {

// add/sub/mul.lo/neg .s32 wrap modulo 2^32. Computing in uint32_t avoids C++ signed-overflow
// UB; converting back to int32_t is modular on GCC/Clang (and guaranteed from C++20).
static int32_t wrap(uint32_t v) { return static_cast<int32_t>(v); }
static int32_t addS32(int32_t a, int32_t b) { return wrap(static_cast<uint32_t>(a) + static_cast<uint32_t>(b)); }
static int32_t subS32(int32_t a, int32_t b) { return wrap(static_cast<uint32_t>(a) - static_cast<uint32_t>(b)); }
static int32_t mulLoS32(int32_t a, int32_t b) { return wrap(static_cast<uint32_t>(a) * static_cast<uint32_t>(b)); }
static int32_t negS32(int32_t a) { return wrap(0u - static_cast<uint32_t>(a)); }

// div.s32/rem.s32 truncate toward zero like C++, but division by zero and INT_MIN / -1 give
// an unspecified result on the GPU (and trap on x86), so those are never folded.
static bool canFoldDivS32(int32_t a, int32_t b) { return b != 0 && !(a == INT32_MIN && b == -1); }

// cvt.rzi.s32.f32 rounds toward zero, saturates to the s32 range, and maps NaN to 0.
// A plain static_cast is UB for out-of-range values and NaN.
static int32_t cvtRziS32F32(float f) {
    if (std::isnan(f)) return 0;
    if (f >= 2147483648.0f) return INT32_MAX;
    if (f <= -2147483648.0f) return INT32_MIN;
    return static_cast<int32_t>(f);
}

} // namespace ptx

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
                        case IROp::Add: res = ptx::addS32(c1.intVal, c2.intVal); break;
                        case IROp::Sub: res = ptx::subS32(c1.intVal, c2.intVal); break;
                        case IROp::Mul: res = ptx::mulLoS32(c1.intVal, c2.intVal); break;
                        case IROp::Div:
                            if (ptx::canFoldDivS32(c1.intVal, c2.intVal)) res = c1.intVal / c2.intVal;
                            else folded = false;
                            break;
                        case IROp::Mod:
                            if (ptx::canFoldDivS32(c1.intVal, c2.intVal)) res = c1.intVal % c2.intVal;
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
                    inst.immInt = ptx::negS32(c1.intVal);
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
                    inst.immInt = ptx::cvtRziS32F32(c1.floatVal);
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
