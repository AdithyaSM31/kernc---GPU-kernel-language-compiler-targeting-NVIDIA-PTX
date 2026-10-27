#include "kernc/Passes.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

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
            for (VReg d : inst.dsts) {
                if (d != 0) {
                    defCount[d]++;
                }
            }
        }
    }

    // 2. Identify global constants (defined exactly once by LoadImm)
    std::unordered_map<VReg, ConstVal> constTable;
    for (const auto& bb : fn.blocks) {
        for (const auto& inst : bb.instructions) {
            // Predicate immediates are not tracked: no folding rule here consumes a .pred.
            if (inst.op == IROp::LoadImm && inst.dst != 0 && defCount[inst.dst] == 1 &&
                inst.type != IRType::Pred) {
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
        for (size_t i = 0; i < bb.instructions.size(); ++i) {
            auto& inst = bb.instructions[i];
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
                        case IROp::And: res = c1.intVal & c2.intVal; break;
                        case IROp::Or:  res = c1.intVal | c2.intVal; break;
                        case IROp::Xor: res = c1.intVal ^ c2.intVal; break;
                        case IROp::Shl:
                            res = ptx::wrap(static_cast<uint32_t>(c1.intVal) << (static_cast<uint32_t>(c2.intVal) & 31u));
                            break;
                        case IROp::ShrS32:
                            res = c1.intVal >> (static_cast<uint32_t>(c2.intVal) & 31u);
                            break;
                        case IROp::ShrU32:
                            res = ptx::wrap(static_cast<uint32_t>(c1.intVal) >> (static_cast<uint32_t>(c2.intVal) & 31u));
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

            // MulWide constant folding (dst:s64 = s32 * s32)
            if (inst.op == IROp::MulWide && src1IsConst && src2IsConst && inst.dst != 0) {
                const auto& c1 = constTable[inst.src1];
                const auto& c2 = constTable[inst.src2];
                if (!c1.isFloat && !c2.isFloat) {
                    int64_t prod = static_cast<int64_t>(c1.intVal) * static_cast<int64_t>(c2.intVal);
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.src2 = 0;
                    inst.type = IRType::S64;
                    inst.immInt = static_cast<int32_t>(prod);
                    constTable[inst.dst] = ConstVal{false, inst.immInt, 0.0f};
                    changed = true;
                    continue;
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
                } else if (inst.op == IROp::Not && !c1.isFloat) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.immInt = ~c1.intVal;
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

            // Algebraic identities and strength reduction
            if (inst.op == IROp::AddS64) {
                if (src2IsConst && !constTable[inst.src2].isFloat && constTable[inst.src2].intVal == 0) {
                    inst.op = IROp::Mov;
                    inst.type = IRType::S64;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
                if (src1IsConst && !constTable[inst.src1].isFloat && constTable[inst.src1].intVal == 0) {
                    inst.op = IROp::Mov;
                    inst.type = IRType::S64;
                    inst.src1 = inst.src2;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
            }
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
                if (inst.op == IROp::Div && val == 1) {
                    inst.op = IROp::Mov;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
                if (inst.op == IROp::Mod && val == 1) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.src2 = 0;
                    inst.immInt = 0;
                    constTable[inst.dst] = ConstVal{false, 0, 0.0f};
                    changed = true;
                    continue;
                }
                if (inst.op == IROp::And && val == 0) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.src2 = 0;
                    inst.immInt = 0;
                    constTable[inst.dst] = ConstVal{false, 0, 0.0f};
                    changed = true;
                    continue;
                }
                if (inst.op == IROp::And && val == -1) {
                    inst.op = IROp::Mov;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
                if ((inst.op == IROp::Or || inst.op == IROp::Xor || inst.op == IROp::Shl ||
                     inst.op == IROp::ShrS32 || inst.op == IROp::ShrU32) && val == 0) {
                    inst.op = IROp::Mov;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
                if (inst.op == IROp::Or && val == -1) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.src2 = 0;
                    inst.immInt = -1;
                    constTable[inst.dst] = ConstVal{false, -1, 0.0f};
                    changed = true;
                    continue;
                }
                if ((inst.op == IROp::Div || inst.op == IROp::Mod) && val > 1 && ((val & (val - 1)) == 0)) {
                    SourceLocation loc = inst.srcLoc;
                    VReg x = inst.src1;
                    VReg dst = inst.dst;
                    int k = __builtin_ctz(static_cast<uint32_t>(val));
                    std::vector<IRInst> newInsts;

                    if (k == 1) {
                        // k = 1: divisor is 2
                        VReg r_sign = fn.freshReg(IRType::S32);
                        VReg imm31 = fn.freshReg(IRType::S32);
                        IRInst li31(IROp::LoadImm, imm31, 0, 0, IRType::S32); li31.immInt = 31; li31.srcLoc = loc;
                        IRInst shr_sign(IROp::ShrU32, r_sign, x, imm31, IRType::S32); shr_sign.srcLoc = loc;

                        VReg r_biased = fn.freshReg(IRType::S32);
                        IRInst add_bias(IROp::Add, r_biased, x, r_sign, IRType::S32); add_bias.srcLoc = loc;

                        newInsts.push_back(li31);
                        newInsts.push_back(shr_sign);
                        newInsts.push_back(add_bias);

                        if (inst.op == IROp::Div) {
                            VReg imm1 = fn.freshReg(IRType::S32);
                            IRInst li1(IROp::LoadImm, imm1, 0, 0, IRType::S32); li1.immInt = 1; li1.srcLoc = loc;
                            IRInst shr_out(IROp::ShrS32, dst, r_biased, imm1, IRType::S32); shr_out.srcLoc = loc;
                            newInsts.push_back(li1);
                            newInsts.push_back(shr_out);
                        } else {
                            VReg r_masked = fn.freshReg(IRType::S32);
                            VReg immMask = fn.freshReg(IRType::S32);
                            IRInst liMask(IROp::LoadImm, immMask, 0, 0, IRType::S32); liMask.immInt = -2; liMask.srcLoc = loc;
                            IRInst and_inst(IROp::And, r_masked, r_biased, immMask, IRType::S32); and_inst.srcLoc = loc;
                            IRInst sub_inst(IROp::Sub, dst, x, r_masked, IRType::S32); sub_inst.srcLoc = loc;
                            newInsts.push_back(liMask);
                            newInsts.push_back(and_inst);
                            newInsts.push_back(sub_inst);
                        }
                    } else {
                        // k in [2, 30]: divisor is 2^k
                        VReg r_mask = fn.freshReg(IRType::S32);
                        VReg imm31 = fn.freshReg(IRType::S32);
                        IRInst li31(IROp::LoadImm, imm31, 0, 0, IRType::S32); li31.immInt = 31; li31.srcLoc = loc;
                        IRInst shr_mask(IROp::ShrS32, r_mask, x, imm31, IRType::S32); shr_mask.srcLoc = loc;

                        VReg r_bias = fn.freshReg(IRType::S32);
                        VReg imm32MinusK = fn.freshReg(IRType::S32);
                        IRInst li32(IROp::LoadImm, imm32MinusK, 0, 0, IRType::S32); li32.immInt = 32 - k; li32.srcLoc = loc;
                        IRInst shr_bias(IROp::ShrU32, r_bias, r_mask, imm32MinusK, IRType::S32); shr_bias.srcLoc = loc;

                        VReg r_biased = fn.freshReg(IRType::S32);
                        IRInst add_bias(IROp::Add, r_biased, x, r_bias, IRType::S32); add_bias.srcLoc = loc;

                        newInsts.push_back(li31);
                        newInsts.push_back(shr_mask);
                        newInsts.push_back(li32);
                        newInsts.push_back(shr_bias);
                        newInsts.push_back(add_bias);

                        if (inst.op == IROp::Div) {
                            VReg immK = fn.freshReg(IRType::S32);
                            IRInst liK(IROp::LoadImm, immK, 0, 0, IRType::S32); liK.immInt = k; liK.srcLoc = loc;
                            IRInst shr_out(IROp::ShrS32, dst, r_biased, immK, IRType::S32); shr_out.srcLoc = loc;
                            newInsts.push_back(liK);
                            newInsts.push_back(shr_out);
                        } else {
                            VReg r_masked = fn.freshReg(IRType::S32);
                            VReg immMask = fn.freshReg(IRType::S32);
                            IRInst liMask(IROp::LoadImm, immMask, 0, 0, IRType::S32);
                            liMask.immInt = ptx::wrap(0u - (1u << k));
                            liMask.srcLoc = loc;
                            IRInst and_inst(IROp::And, r_masked, r_biased, immMask, IRType::S32); and_inst.srcLoc = loc;
                            IRInst sub_inst(IROp::Sub, dst, x, r_masked, IRType::S32); sub_inst.srcLoc = loc;
                            newInsts.push_back(liMask);
                            newInsts.push_back(and_inst);
                            newInsts.push_back(sub_inst);
                        }
                    }

                    bb.instructions.erase(bb.instructions.begin() + i);
                    bb.instructions.insert(bb.instructions.begin() + i, newInsts.begin(), newInsts.end());
                    i += newInsts.size() - 1;
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
                if (inst.op == IROp::And && val == 0) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.src2 = 0;
                    inst.immInt = 0;
                    constTable[inst.dst] = ConstVal{false, 0, 0.0f};
                    changed = true;
                    continue;
                }
                if (inst.op == IROp::And && val == -1) {
                    inst.op = IROp::Mov;
                    inst.src1 = inst.src2;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
                if ((inst.op == IROp::Or || inst.op == IROp::Xor) && val == 0) {
                    inst.op = IROp::Mov;
                    inst.src1 = inst.src2;
                    inst.src2 = 0;
                    changed = true;
                    continue;
                }
                if (inst.op == IROp::Or && val == -1) {
                    inst.op = IROp::LoadImm;
                    inst.src1 = 0;
                    inst.src2 = 0;
                    inst.immInt = -1;
                    constTable[inst.dst] = ConstVal{false, -1, 0.0f};
                    changed = true;
                    continue;
                }
            }

            // Self-identities: x ^ x = 0, x - x = 0
            if (inst.src1 != 0 && inst.src1 == inst.src2 && inst.dst != 0) {
                if (inst.op == IROp::Xor || inst.op == IROp::Sub) {
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
                if (inst.src3 != 0) useCount[inst.src3]++;
                for (VReg arg : inst.args) {
                    if (arg != 0) useCount[arg]++;
                }
            }
        }

        for (auto& bb : fn.blocks) {
            for (auto it = bb.instructions.begin(); it != bb.instructions.end(); ) {
                // Side-effecting operations
                if (it->op == IROp::Store || it->op == IROp::StoreV2 ||
                    it->op == IROp::StoreV4 || it->op == IROp::Bra ||
                    it->op == IROp::BraCond || it->op == IROp::Ret ||
                    it->op == IROp::Bar || it->op == IROp::Call ||
                    it->op == IROp::StoreRetval ||
                    it->op == IROp::ShflIdx || it->op == IROp::ShflDown ||
                    it->op == IROp::ShflUp || it->op == IROp::ShflBfly ||
                    it->op == IROp::VoteBallot ||
                    it->op == IROp::AtomicAdd || it->op == IROp::AtomicSub ||
                    it->op == IROp::AtomicMin || it->op == IROp::AtomicMax ||
                    it->op == IROp::AtomicAnd || it->op == IROp::AtomicOr ||
                    it->op == IROp::AtomicXor || it->op == IROp::AtomicExch ||
                    it->op == IROp::AtomicCas) {
                    ++it;
                    continue;
                }

                if (it->op == IROp::LoadV2 || it->op == IROp::LoadV4) {
                    bool allDead = true;
                    for (VReg d : it->dsts) {
                        if (d != 0 && useCount[d] > 0) {
                            allDead = false;
                            break;
                        }
                    }
                    if (allDead) {
                        it = bb.instructions.erase(it);
                        changed = true;
                        changedOverall = true;
                    } else {
                        ++it;
                    }
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

namespace {

static bool isFenceOrBarrier(IROp op) {
    switch (op) {
        case IROp::Bar:
        case IROp::AtomicAdd:
        case IROp::AtomicSub:
        case IROp::AtomicMin:
        case IROp::AtomicMax:
        case IROp::AtomicAnd:
        case IROp::AtomicOr:
        case IROp::AtomicXor:
        case IROp::AtomicExch:
        case IROp::AtomicCas:
        case IROp::ShflIdx:
        case IROp::ShflDown:
        case IROp::ShflUp:
        case IROp::ShflBfly:
        case IROp::VoteBallot:
        case IROp::Activemask:
        case IROp::Call:
        case IROp::StoreRetval:
        case IROp::Bra:
        case IROp::BraCond:
        case IROp::Ret:
            return true;
        default:
            return false;
    }
}

static void getDefs(const IRInst& inst, std::vector<VReg>& defs) {
    if (inst.dst != 0) defs.push_back(inst.dst);
    for (VReg d : inst.dsts) {
        if (d != 0) defs.push_back(d);
    }
}

static void getUses(const IRInst& inst, std::vector<VReg>& uses) {
    if (inst.src1 != 0) uses.push_back(inst.src1);
    if (inst.src2 != 0) uses.push_back(inst.src2);
    if (inst.src3 != 0) uses.push_back(inst.src3);
    for (VReg a : inst.args) {
        if (a != 0) uses.push_back(a);
    }
}

static bool isLoadOp(IROp op) {
    return op == IROp::Load || op == IROp::LoadV2 || op == IROp::LoadV4;
}

static bool isStoreOp(IROp op) {
    return op == IROp::Store || op == IROp::StoreV2 || op == IROp::StoreV4;
}

static bool isMemOp(IROp op) {
    return isLoadOp(op) || isStoreOp(op);
}

static int getInstBaseLatency(const IRInst& inst) {
    if (isLoadOp(inst.op)) {
        return (inst.space == AddrSpace::Global) ? 100 : 80;
    }
    if (inst.op == IROp::ReadSReg || inst.op == IROp::LoadParam || inst.op == IROp::LoadImm) {
        return 50;
    }
    if (inst.op == IROp::MulWide || inst.op == IROp::AddS64 || inst.op == IROp::CvtaGlobal || inst.op == IROp::SharedAddr) {
        return 40;
    }
    if (isStoreOp(inst.op)) {
        return 1;
    }
    return 20;
}

static std::unordered_map<VReg, VReg> computePointerBases(const IRFunction& fn) {
    std::unordered_map<VReg, VReg> pointerBase;
    for (int iter = 0; iter < 4; ++iter) {
        for (const auto& bb : fn.blocks) {
            for (const auto& inst : bb.instructions) {
                if (inst.dst == 0) continue;
                if (inst.op == IROp::LoadParam || inst.op == IROp::SharedAddr) {
                    pointerBase[inst.dst] = inst.dst;
                } else if (inst.op == IROp::CvtaGlobal) {
                    if (pointerBase.count(inst.src1)) pointerBase[inst.dst] = pointerBase[inst.src1];
                } else if (inst.op == IROp::Mov) {
                    if (pointerBase.count(inst.src1)) pointerBase[inst.dst] = pointerBase[inst.src1];
                } else if (inst.op == IROp::AddS64 || inst.op == IROp::Add) {
                    if (pointerBase.count(inst.src1)) pointerBase[inst.dst] = pointerBase[inst.src1];
                    else if (pointerBase.count(inst.src2)) pointerBase[inst.dst] = pointerBase[inst.src2];
                }
            }
        }
    }
    return pointerBase;
}

static std::vector<IRInst> scheduleWindow(const std::vector<IRInst>& window,
                                          const std::unordered_map<VReg, VReg>& pointerBases) {
    const size_t N = window.size();
    if (N <= 1) return window;

    std::vector<std::vector<int>> succs(N);
    std::vector<std::vector<int>> preds(N);
    std::vector<int> inDegree(N, 0);

    std::vector<std::vector<VReg>> defs(N);
    std::vector<std::vector<VReg>> uses(N);
    std::vector<VReg> baseRegs(N, 0);

    for (size_t i = 0; i < N; ++i) {
        getDefs(window[i], defs[i]);
        getUses(window[i], uses[i]);
        if (isMemOp(window[i].op)) {
            auto it = pointerBases.find(window[i].src1);
            if (it != pointerBases.end()) {
                baseRegs[i] = it->second;
            }
        }
    }

    auto addEdge = [&](int from, int to) {
        for (int s : succs[from]) {
            if (s == to) return;
        }
        succs[from].push_back(to);
        preds[to].push_back(from);
        inDegree[to]++;
    };

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            bool dependent = false;
            // RAW: instruction i defines a register read by instruction j
            for (VReg d : defs[i]) {
                for (VReg u : uses[j]) {
                    if (d == u) { dependent = true; break; }
                }
                if (dependent) break;
            }
            // WAR: instruction i reads a register overwritten by instruction j
            if (!dependent) {
                for (VReg u : uses[i]) {
                    for (VReg d : defs[j]) {
                        if (u == d) { dependent = true; break; }
                    }
                    if (dependent) break;
                }
            }
            // WAW: instruction i defines a register overwritten by instruction j
            if (!dependent) {
                for (VReg d1 : defs[i]) {
                    for (VReg d2 : defs[j]) {
                        if (d1 == d2) { dependent = true; break; }
                    }
                    if (dependent) break;
                }
            }
            // Memory dependencies
            if (!dependent && isMemOp(window[i].op) && isMemOp(window[j].op)) {
                const auto& m1 = window[i];
                const auto& m2 = window[j];
                if (m1.space == m2.space) {
                    if (isStoreOp(m1.op) && isStoreOp(m2.op)) {
                        // Preserve relative store order
                        dependent = true;
                    } else if (isStoreOp(m1.op) || isStoreOp(m2.op)) {
                        // Aliased store-load or load-store hazard
                        VReg b1 = baseRegs[i];
                        VReg b2 = baseRegs[j];
                        if (b1 == 0 || b2 == 0 || b1 == b2) {
                            dependent = true;
                        }
                    }
                }
            }
            if (dependent) {
                addEdge(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    // Critical path depth computation
    std::vector<int> priority(N, 0);
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        int maxSucc = 0;
        for (int succ : succs[i]) {
            maxSucc = std::max(maxSucc, priority[succ]);
        }
        priority[i] = getInstBaseLatency(window[i]) + maxSucc;
    }

    // List scheduling ready queue
    std::vector<int> ready;
    for (int i = 0; i < static_cast<int>(N); ++i) {
        if (inDegree[i] == 0) {
            ready.push_back(i);
        }
    }

    std::vector<IRInst> scheduled;
    scheduled.reserve(N);

    while (!ready.empty()) {
        auto bestIt = ready.begin();
        for (auto it = ready.begin() + 1; it != ready.end(); ++it) {
            int bestIdx = *bestIt;
            int curIdx = *it;
            if (priority[curIdx] > priority[bestIdx]) {
                bestIt = it;
            } else if (priority[curIdx] == priority[bestIdx] && curIdx < bestIdx) {
                bestIt = it;
            }
        }
        int u = *bestIt;
        ready.erase(bestIt);
        scheduled.push_back(window[u]);

        for (int v : succs[u]) {
            inDegree[v]--;
            if (inDegree[v] == 0) {
                ready.push_back(v);
            }
        }
    }

    return scheduled;
}

} // anonymous namespace

bool Optimizer::runInstructionScheduling(IRFunction& fn) {
    bool changedOverall = false;
    auto pointerBases = computePointerBases(fn);

    for (auto& bb : fn.blocks) {
        std::vector<IRInst> newInstructions;
        size_t idx = 0;
        while (idx < bb.instructions.size()) {
            if (isFenceOrBarrier(bb.instructions[idx].op)) {
                newInstructions.push_back(bb.instructions[idx]);
                ++idx;
                continue;
            }
            size_t start = idx;
            while (idx < bb.instructions.size() && !isFenceOrBarrier(bb.instructions[idx].op)) {
                ++idx;
            }
            size_t count = idx - start;
            if (count <= 1) {
                newInstructions.push_back(bb.instructions[start]);
                continue;
            }
            std::vector<IRInst> window(bb.instructions.begin() + start, bb.instructions.begin() + idx);
            auto scheduled = scheduleWindow(window, pointerBases);
            for (size_t k = 0; k < count; ++k) {
                if (window[k].op != scheduled[k].op || window[k].dst != scheduled[k].dst || window[k].src1 != scheduled[k].src1) {
                    changedOverall = true;
                    break;
                }
            }
            newInstructions.insert(newInstructions.end(), scheduled.begin(), scheduled.end());
        }
        bb.instructions = std::move(newInstructions);
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
    if (runInstructionScheduling(fn)) {
        changed = true;
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
