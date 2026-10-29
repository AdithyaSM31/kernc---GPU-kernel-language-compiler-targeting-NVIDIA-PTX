#include "kernc/Passes.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
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

int32_t mulHiS32(int32_t a, int32_t b) {
    int64_t prod = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    return static_cast<int32_t>(prod >> 32);
}

// Granlund-Montgomery / Hacker's Delight magic number generation for signed 32-bit division
// ad = |d| must satisfy 2 <= ad <= 2^31 - 1 (i.e. d != 0, 1, -1, INT32_MIN).
MagicInfo computeSignedMagic(int32_t d) {
    uint32_t ad = (d < 0) ? wrap(0u - static_cast<uint32_t>(d)) : static_cast<uint32_t>(d);
    const uint32_t two31 = 0x80000000u;
    uint32_t t = two31 + (ad >> 31);
    uint32_t anc = t - 1 - (t % ad);
    uint32_t p = 31;
    uint32_t q1 = two31 / anc;
    uint32_t r1 = two31 - q1 * anc;
    uint32_t q2 = two31 / ad;
    uint32_t r2 = two31 - q2 * ad;
    uint32_t delta;
    do {
        p++;
        q1 = 2 * q1;
        r1 = 2 * r1;
        if (r1 >= anc) { q1++; r1 -= anc; }
        q2 = 2 * q2;
        r2 = 2 * r2;
        if (r2 >= ad) { q2++; r2 -= ad; }
        delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));

    uint32_t M_u = q2 + 1;
    int32_t M = static_cast<int32_t>(M_u);
    uint32_t s = p - 32;
    return {M, s};
}

// shl.b32 / shr.u32 / shr.s32 clamp the shift count (read as unsigned) to 32, so a count of
// 32 or more shifts every bit out: 0, or all sign bits for shr.s32. In C++ such shifts are
// undefined, and x86 masks the count to 5 bits, so neither matches the GPU.
static int32_t shlB32(int32_t a, int32_t n) {
    uint32_t c = static_cast<uint32_t>(n);
    return c >= 32 ? 0 : wrap(static_cast<uint32_t>(a) << c);
}
static int32_t shrU32(int32_t a, int32_t n) {
    uint32_t c = static_cast<uint32_t>(n);
    return c >= 32 ? 0 : wrap(static_cast<uint32_t>(a) >> c);
}
static int32_t shrS32(int32_t a, int32_t n) {
    uint32_t c = static_cast<uint32_t>(n);
    if (c >= 32) return a < 0 ? -1 : 0;
    return a >> c; // arithmetic on GCC/Clang (guaranteed from C++20)
}

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
                        case IROp::MulHi: res = ptx::mulHiS32(c1.intVal, c2.intVal); break;
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
                        case IROp::Shl:    res = ptx::shlB32(c1.intVal, c2.intVal); break;
                        case IROp::ShrS32: res = ptx::shrS32(c1.intVal, c2.intVal); break;
                        case IROp::ShrU32: res = ptx::shrU32(c1.intVal, c2.intVal); break;
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
                } else if ((inst.op == IROp::Div || inst.op == IROp::Mod) &&
                           val != 0 && val != 1 && val != -1 && val != INT32_MIN) {
                    SourceLocation loc = inst.srcLoc;
                    VReg x = inst.src1;
                    VReg dst = inst.dst;
                    int32_t d = val;
                    ptx::MagicInfo m = ptx::computeSignedMagic(d);
                    std::vector<IRInst> newInsts;

                    // 1. Load magic multiplier M into register
                    VReg immM = fn.freshReg(IRType::S32);
                    IRInst liM(IROp::LoadImm, immM, 0, 0, IRType::S32);
                    liM.immInt = m.M;
                    liM.srcLoc = loc;
                    newInsts.push_back(liM);

                    // 2. High 32 bits of product: q0 = mul.hi.s32(x, M)
                    VReg q0 = fn.freshReg(IRType::S32);
                    IRInst mulHi(IROp::MulHi, q0, x, immM, IRType::S32);
                    mulHi.srcLoc = loc;
                    newInsts.push_back(mulHi);

                    // 3. If M < 0: q1 = q0 + x (compensate for negative signed multiplier)
                    VReg q1 = q0;
                    if (m.M < 0) {
                        q1 = fn.freshReg(IRType::S32);
                        IRInst addN(IROp::Add, q1, q0, x, IRType::S32);
                        addN.srcLoc = loc;
                        newInsts.push_back(addN);
                    }

                    // 4. Arithmetic right shift by post-shift s: q2 = q1 >> s
                    VReg q2 = q1;
                    if (m.s > 0) {
                        VReg immS = fn.freshReg(IRType::S32);
                        IRInst liS(IROp::LoadImm, immS, 0, 0, IRType::S32);
                        liS.immInt = static_cast<int32_t>(m.s);
                        liS.srcLoc = loc;
                        newInsts.push_back(liS);

                        q2 = fn.freshReg(IRType::S32);
                        IRInst shrS(IROp::ShrS32, q2, q1, immS, IRType::S32);
                        shrS.srcLoc = loc;
                        newInsts.push_back(shrS);
                    }

                    // 5. Sign correction: add sign bit of x (1 if x < 0, 0 if x >= 0)
                    VReg imm31 = fn.freshReg(IRType::S32);
                    IRInst li31(IROp::LoadImm, imm31, 0, 0, IRType::S32);
                    li31.immInt = 31;
                    li31.srcLoc = loc;
                    newInsts.push_back(li31);

                    VReg rSign = fn.freshReg(IRType::S32);
                    IRInst shrSign(IROp::ShrU32, rSign, x, imm31, IRType::S32);
                    shrSign.srcLoc = loc;
                    newInsts.push_back(shrSign);

                    // 6. Quotient for positive |d|: qPos = q2 + rSign
                    VReg qPos = (d > 0 && inst.op == IROp::Div) ? dst : fn.freshReg(IRType::S32);
                    IRInst addSign(IROp::Add, qPos, q2, rSign, IRType::S32);
                    addSign.srcLoc = loc;
                    newInsts.push_back(addSign);

                    // 7. If d < 0: negate quotient (trunc(x / d) = -trunc(x / |d|))
                    VReg q = qPos;
                    if (d < 0) {
                        q = (inst.op == IROp::Div) ? dst : fn.freshReg(IRType::S32);
                        IRInst negQ(IROp::Neg, q, qPos, 0, IRType::S32);
                        negQ.srcLoc = loc;
                        newInsts.push_back(negQ);
                    }

                    // 8. If Mod: r = x - q * d
                    if (inst.op == IROp::Mod) {
                        VReg immD = fn.freshReg(IRType::S32);
                        IRInst liD(IROp::LoadImm, immD, 0, 0, IRType::S32);
                        liD.immInt = d;
                        liD.srcLoc = loc;
                        newInsts.push_back(liD);

                        VReg prod = fn.freshReg(IRType::S32);
                        IRInst mulD(IROp::Mul, prod, q, immD, IRType::S32);
                        mulD.srcLoc = loc;
                        newInsts.push_back(mulD);

                        IRInst subR(IROp::Sub, dst, x, prod, IRType::S32);
                        subR.srcLoc = loc;
                        newInsts.push_back(subR);
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
                if (inst.src4 != 0) useCount[inst.src4]++;
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
    if (inst.src4 != 0) uses.push_back(inst.src4);
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

// Maps every address register derived from a SharedAddr to that shared array's symbol. Each
// access computes its own base register, so the symbol (not the register) identifies the array.
static std::unordered_map<VReg, std::string> computeSharedObjects(
        const IRFunction& fn, const std::unordered_map<VReg, VReg>& pointerBases) {
    std::unordered_map<VReg, std::string> symbolOfBase;
    for (const auto& bb : fn.blocks)
        for (const auto& inst : bb.instructions)
            if (inst.op == IROp::SharedAddr && inst.dst != 0) symbolOfBase[inst.dst] = inst.symbol;

    std::unordered_map<VReg, std::string> objectOf;
    for (const auto& [reg, base] : pointerBases) {
        auto it = symbolOfBase.find(base);
        if (it != symbolOfBase.end()) objectOf[reg] = it->second;
    }
    return objectOf;
}

static std::vector<IRInst> scheduleWindow(const std::vector<IRInst>& window,
                                          const std::unordered_map<VReg, std::string>& sharedObjects) {
    const size_t N = window.size();
    if (N <= 1) return window;

    std::vector<std::vector<int>> succs(N);
    std::vector<std::vector<int>> preds(N);
    std::vector<int> inDegree(N, 0);

    std::vector<std::vector<VReg>> defs(N);
    std::vector<std::vector<VReg>> uses(N);
    std::vector<std::string> sharedObject(N); // shared array accessed, or empty if unknown

    for (size_t i = 0; i < N; ++i) {
        getDefs(window[i], defs[i]);
        getUses(window[i], uses[i]);
        if (isMemOp(window[i].op) && window[i].space == AddrSpace::Shared) {
            auto it = sharedObjects.find(window[i].src1);
            if (it != sharedObjects.end()) {
                sharedObject[i] = it->second;
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
                        // A load and a store may only be reordered if they provably touch
                        // different memory. The one provable case is two different shared
                        // arrays. Global pointers may alias each other (as in C/CUDA without
                        // __restrict__), and two accesses to one shared array may overlap.
                        bool disjoint = !sharedObject[i].empty() && !sharedObject[j].empty() &&
                                        sharedObject[i] != sharedObject[j];
                        if (!disjoint) {
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

static bool isCommutativeOp(IROp op) {
    switch (op) {
        case IROp::Add:
        case IROp::Mul:
        case IROp::MulHi:
        case IROp::And:
        case IROp::Or:
        case IROp::Xor:
        case IROp::FAdd:
        case IROp::FMul:
        case IROp::AddS64:
        case IROp::MulWide:
        case IROp::SetpEq:
        case IROp::SetpNe:
        case IROp::SetpFEq:
        case IROp::SetpFNe:
        case IROp::XorPred:
            return true;
        default:
            return false;
    }
}

static bool isCSECandidate(const IRInst& inst) {
    switch (inst.op) {
        case IROp::Add:
        case IROp::Sub:
        case IROp::Mul:
        case IROp::MulHi:
        case IROp::Div:
        case IROp::Mod:
        case IROp::And:
        case IROp::Or:
        case IROp::Xor:
        case IROp::Not:
        case IROp::Shl:
        case IROp::ShrS32:
        case IROp::ShrU32:
        case IROp::FAdd:
        case IROp::FSub:
        case IROp::FMul:
        case IROp::FDiv:
        case IROp::Neg:
        case IROp::FNeg:
        case IROp::CvtF2I:
        case IROp::CvtI2F:
        case IROp::CvtPredToS32:
        case IROp::CvtPredToF32:
        case IROp::NotPred:
        case IROp::XorPred:
        case IROp::SetpLt:
        case IROp::SetpGt:
        case IROp::SetpLe:
        case IROp::SetpGe:
        case IROp::SetpEq:
        case IROp::SetpNe:
        case IROp::SetpFLt:
        case IROp::SetpFGt:
        case IROp::SetpFLe:
        case IROp::SetpFGe:
        case IROp::SetpFEq:
        case IROp::SetpFNe:
        case IROp::MulWide:
        case IROp::AddS64:
        case IROp::CvtaGlobal:
        case IROp::SharedAddr:
        case IROp::LoadImm:
        case IROp::LoadParam:
        case IROp::ReadSReg:
        case IROp::Load:
            return true;
        default:
            return false;
    }
}

struct ExprKey {
    IROp op = IROp::Mov;
    IRType type = IRType::S32;
    VReg src1 = 0;
    VReg src2 = 0;
    int32_t immInt = 0;
    float immFloat = 0.0f;
    std::string symbol;
    std::string specialReg;
    int paramIndex = 0;
    AddrSpace space = AddrSpace::Global;

    bool operator==(const ExprKey& o) const {
        if (op != o.op || type != o.type || src1 != o.src1 || src2 != o.src2)
            return false;
        if (immInt != o.immInt) return false;
        uint32_t fb1 = 0, fb2 = 0;
        std::memcpy(&fb1, &immFloat, sizeof(fb1));
        std::memcpy(&fb2, &o.immFloat, sizeof(fb2));
        if (fb1 != fb2) return false;
        if (symbol != o.symbol || specialReg != o.specialReg || paramIndex != o.paramIndex)
            return false;
        if (space != o.space) return false;
        return true;
    }
};

struct ExprKeyHash {
    size_t operator()(const ExprKey& k) const {
        size_t h = 0;
        auto hashCombine = [&h](size_t val) {
            h ^= val + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        hashCombine(static_cast<size_t>(k.op));
        hashCombine(static_cast<size_t>(k.type));
        hashCombine(k.src1);
        hashCombine(k.src2);
        hashCombine(static_cast<uint32_t>(k.immInt));
        uint32_t fb = 0;
        std::memcpy(&fb, &k.immFloat, sizeof(fb));
        hashCombine(fb);
        hashCombine(std::hash<std::string>{}(k.symbol));
        hashCombine(std::hash<std::string>{}(k.specialReg));
        hashCombine(static_cast<size_t>(k.paramIndex));
        hashCombine(static_cast<size_t>(k.space));
        return h;
    }
};

static ExprKey makeExprKey(const IRInst& inst, const std::unordered_map<VReg, std::string>& sharedObjects) {
    ExprKey k;
    k.op = inst.op;
    k.type = inst.type;
    k.src1 = inst.src1;
    k.src2 = inst.src2;
    k.immInt = inst.immInt;
    k.immFloat = inst.immFloat;
    k.symbol = inst.symbol;
    k.specialReg = inst.specialReg;
    k.paramIndex = inst.paramIndex;
    k.space = inst.space;

    if (inst.op == IROp::Load && inst.space == AddrSpace::Shared) {
        auto it = sharedObjects.find(inst.src1);
        if (it != sharedObjects.end()) {
            k.symbol = it->second;
        }
    }
    return k;
}

} // anonymous namespace

bool Optimizer::runCSE(IRFunction& fn) {
    bool changedOverall = false;
    auto sharedObjects = computeSharedObjects(fn, computePointerBases(fn));

    for (auto& bb : fn.blocks) {
        std::unordered_map<VReg, VReg> copyMap;
        std::unordered_map<ExprKey, VReg, ExprKeyHash> availExprs;

        auto getRoot = [&](VReg r) -> VReg {
            if (r == 0) return 0;
            VReg cur = r;
            std::unordered_set<VReg> visited;
            while (true) {
                auto it = copyMap.find(cur);
                if (it == copyMap.end()) break;
                if (visited.count(it->second)) break;
                visited.insert(cur);
                cur = it->second;
            }
            return cur;
        };

        auto killRegister = [&](VReg reg) {
            if (reg == 0) return;
            copyMap.erase(reg);
            for (auto it = copyMap.begin(); it != copyMap.end(); ) {
                if (it->second == reg) {
                    it = copyMap.erase(it);
                } else {
                    ++it;
                }
            }
            for (auto it = availExprs.begin(); it != availExprs.end(); ) {
                if (it->second == reg || it->first.src1 == reg || it->first.src2 == reg) {
                    it = availExprs.erase(it);
                } else {
                    ++it;
                }
            }
        };

        auto killGlobalLoads = [&]() {
            for (auto it = availExprs.begin(); it != availExprs.end(); ) {
                if (it->first.op == IROp::Load && it->first.space == AddrSpace::Global) {
                    it = availExprs.erase(it);
                } else {
                    ++it;
                }
            }
        };

        auto killSharedLoads = [&](const std::string& targetArraySym) {
            for (auto it = availExprs.begin(); it != availExprs.end(); ) {
                if (it->first.op == IROp::Load && it->first.space == AddrSpace::Shared) {
                    if (targetArraySym.empty() || it->first.symbol.empty() ||
                        targetArraySym == it->first.symbol) {
                        it = availExprs.erase(it);
                    } else {
                        ++it;
                    }
                } else {
                    ++it;
                }
            }
        };

        auto killAllLoads = [&]() {
            for (auto it = availExprs.begin(); it != availExprs.end(); ) {
                if (it->first.op == IROp::Load) {
                    it = availExprs.erase(it);
                } else {
                    ++it;
                }
            }
        };

        for (size_t i = 0; i < bb.instructions.size(); ++i) {
            auto& inst = bb.instructions[i];

            // 1. Copy propagate into operands
            if (inst.src1 != 0) {
                VReg root = getRoot(inst.src1);
                if (root != inst.src1 && fn.regType(root) == fn.regType(inst.src1)) {
                    inst.src1 = root;
                    changedOverall = true;
                }
            }
            if (inst.src2 != 0) {
                VReg root = getRoot(inst.src2);
                if (root != inst.src2 && fn.regType(root) == fn.regType(inst.src2)) {
                    inst.src2 = root;
                    changedOverall = true;
                }
            }
            if (inst.src3 != 0) {
                VReg root = getRoot(inst.src3);
                if (root != inst.src3 && fn.regType(root) == fn.regType(inst.src3)) {
                    inst.src3 = root;
                    changedOverall = true;
                }
            }
            if (inst.src4 != 0) {
                VReg root = getRoot(inst.src4);
                if (root != inst.src4 && fn.regType(root) == fn.regType(inst.src4)) {
                    inst.src4 = root;
                    changedOverall = true;
                }
            }
            for (auto& a : inst.args) {
                if (a != 0) {
                    VReg root = getRoot(a);
                    if (root != a && fn.regType(root) == fn.regType(a)) {
                        a = root;
                        changedOverall = true;
                    }
                }
            }

            // 2. Canonicalize commutative operands
            if (isCommutativeOp(inst.op) && inst.src1 != 0 && inst.src2 != 0) {
                if (inst.src1 > inst.src2) {
                    std::swap(inst.src1, inst.src2);
                }
            }

            // 3. Handle Mov
            if (inst.op == IROp::Mov) {
                if (inst.dst == inst.src1 && inst.dst != 0) {
                    bb.instructions.erase(bb.instructions.begin() + i);
                    --i;
                    changedOverall = true;
                    continue;
                }
                if (inst.dst != 0) {
                    killRegister(inst.dst);
                    if (inst.src1 != 0) {
                        copyMap[inst.dst] = getRoot(inst.src1);
                    }
                }
                continue;
            }

            // 4. Check if CSE candidate
            bool cseCandidate = isCSECandidate(inst);
            if (cseCandidate && inst.dst != 0) {
                ExprKey key = makeExprKey(inst, sharedObjects);
                auto it = availExprs.find(key);
                if (it != availExprs.end() && it->second != 0) {
                    VReg availReg = it->second;
                    killRegister(inst.dst);
                    if (inst.dst == availReg) {
                        bb.instructions.erase(bb.instructions.begin() + i);
                        --i;
                        changedOverall = true;
                        continue;
                    }
                    inst.op = IROp::Mov;
                    inst.src1 = availReg;
                    inst.src2 = 0;
                    inst.src3 = 0;
                    inst.src4 = 0;
                    inst.immInt = 0;
                    inst.immFloat = 0.0f;
                    inst.symbol.clear();
                    inst.specialReg.clear();
                    copyMap[inst.dst] = getRoot(availReg);
                    changedOverall = true;
                    continue;
                } else {
                    killRegister(inst.dst);
                    if (inst.dst != inst.src1 && inst.dst != inst.src2 && inst.dst != inst.src3 && inst.dst != inst.src4) {
                        availExprs[key] = inst.dst;
                    }
                }
            } else {
                std::vector<VReg> defs;
                getDefs(inst, defs);
                for (VReg d : defs) {
                    killRegister(d);
                }
            }

            // 5. Memory fence and store clobber handling
            if (isStoreOp(inst.op)) {
                if (inst.space == AddrSpace::Global) {
                    killGlobalLoads();
                } else if (inst.space == AddrSpace::Shared) {
                    std::string sym;
                    auto sit = sharedObjects.find(inst.src1);
                    if (sit != sharedObjects.end()) sym = sit->second;
                    killSharedLoads(sym);
                }
            } else if (isFenceOrBarrier(inst.op)) {
                killAllLoads();
            }
        }
    }

    return changedOverall;
}

bool Optimizer::runCFGCleanup(IRFunction& fn) {
    if (fn.blocks.empty()) return false;
    bool changedOverall = false;

    bool changed = true;
    while (changed) {
        changed = false;

        // Step 1: Remove Unreachable Blocks
        // Traverse CFG reachability starting from fn.blocks[0]
        std::unordered_set<std::string> reachable;
        std::vector<std::string> worklist;
        reachable.insert(fn.blocks[0].label);
        worklist.push_back(fn.blocks[0].label);

        std::unordered_map<std::string, const BasicBlock*> blockMap;
        for (const auto& bb : fn.blocks) {
            blockMap[bb.label] = &bb;
        }

        while (!worklist.empty()) {
            std::string curLabel = worklist.back();
            worklist.pop_back();

            auto it = blockMap.find(curLabel);
            if (it == blockMap.end()) continue;
            const auto& bb = *it->second;

            if (bb.instructions.empty()) {
                for (size_t idx = 0; idx < fn.blocks.size(); ++idx) {
                    if (fn.blocks[idx].label == curLabel && idx + 1 < fn.blocks.size()) {
                        const std::string& nextLabel = fn.blocks[idx + 1].label;
                        if (reachable.insert(nextLabel).second) {
                            worklist.push_back(nextLabel);
                        }
                        break;
                    }
                }
            } else {
                const auto& last = bb.instructions.back();
                if (last.op == IROp::Ret) {
                    // No successors
                } else if (last.op == IROp::Bra) {
                    if (!last.trueTarget.empty() && reachable.insert(last.trueTarget).second) {
                        worklist.push_back(last.trueTarget);
                    }
                } else if (last.op == IROp::BraCond) {
                    if (!last.trueTarget.empty() && reachable.insert(last.trueTarget).second) {
                        worklist.push_back(last.trueTarget);
                    }
                    if (!last.falseTarget.empty() && reachable.insert(last.falseTarget).second) {
                        worklist.push_back(last.falseTarget);
                    }
                } else {
                    for (size_t idx = 0; idx < fn.blocks.size(); ++idx) {
                        if (fn.blocks[idx].label == curLabel && idx + 1 < fn.blocks.size()) {
                            const std::string& nextLabel = fn.blocks[idx + 1].label;
                            if (reachable.insert(nextLabel).second) {
                                worklist.push_back(nextLabel);
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Delete unreachable blocks
        for (auto it = fn.blocks.begin(); it != fn.blocks.end(); ) {
            if (it != fn.blocks.begin() && !reachable.count(it->label)) {
                it = fn.blocks.erase(it);
                changed = true;
                changedOverall = true;
            } else {
                ++it;
            }
        }
        if (changed) continue;

        // Step 2: Fold unconditional Bra targeting the immediate next block in layout
        for (size_t i = 0; i + 1 < fn.blocks.size(); ++i) {
            auto& bb = fn.blocks[i];
            const auto& nextBb = fn.blocks[i + 1];
            if (!bb.instructions.empty()) {
                if (bb.instructions.back().op == IROp::Bra &&
                    bb.instructions.back().trueTarget == nextBb.label) {
                    bb.instructions.pop_back();
                    changed = true;
                    changedOverall = true;
                    break;
                }
            }
        }
        if (changed) continue;

        // Step 3: Fold Jump Trampolines (blocks that contain only `bra target`)
        for (size_t i = 1; i < fn.blocks.size(); ++i) {
            const auto& bb = fn.blocks[i];
            if (bb.instructions.size() == 1 && bb.instructions[0].op == IROp::Bra) {
                const std::string& target = bb.instructions[0].trueTarget;
                if (!target.empty() && target != bb.label) {
                    for (auto& other : fn.blocks) {
                        for (auto& inst : other.instructions) {
                            if (inst.op == IROp::Bra && inst.trueTarget == bb.label) {
                                inst.trueTarget = target;
                                changed = true;
                                changedOverall = true;
                            } else if (inst.op == IROp::BraCond) {
                                if (inst.trueTarget == bb.label) {
                                    inst.trueTarget = target;
                                    changed = true;
                                    changedOverall = true;
                                }
                                if (inst.falseTarget == bb.label) {
                                    inst.falseTarget = target;
                                    changed = true;
                                    changedOverall = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (changed) continue;

        // Step 4: Merge Single-Predecessor Blocks
        std::unordered_map<std::string, std::vector<std::string>> preds;
        for (size_t i = 0; i < fn.blocks.size(); ++i) {
            const auto& bb = fn.blocks[i];
            if (bb.instructions.empty()) {
                if (i + 1 < fn.blocks.size()) {
                    preds[fn.blocks[i + 1].label].push_back(bb.label);
                }
            } else {
                const auto& last = bb.instructions.back();
                if (last.op == IROp::Ret) {
                    // No successors
                } else if (last.op == IROp::Bra) {
                    preds[last.trueTarget].push_back(bb.label);
                } else if (last.op == IROp::BraCond) {
                    preds[last.trueTarget].push_back(bb.label);
                    preds[last.falseTarget].push_back(bb.label);
                } else {
                    if (i + 1 < fn.blocks.size()) {
                        preds[fn.blocks[i + 1].label].push_back(bb.label);
                    }
                }
            }
        }

        for (size_t i = 0; i + 1 < fn.blocks.size(); ++i) {
            auto& bbA = fn.blocks[i];
            const auto& bbB = fn.blocks[i + 1];

            bool aOnlyGoesToB = false;
            if (bbA.instructions.empty()) {
                aOnlyGoesToB = true;
            } else {
                const auto& last = bbA.instructions.back();
                if (last.op == IROp::Bra && last.trueTarget == bbB.label) {
                    aOnlyGoesToB = true;
                } else if (last.op != IROp::Ret && last.op != IROp::Bra && last.op != IROp::BraCond) {
                    aOnlyGoesToB = true;
                }
            }

            if (aOnlyGoesToB) {
                auto pIt = preds.find(bbB.label);
                if (pIt != preds.end() && pIt->second.size() == 1 && pIt->second[0] == bbA.label) {
                    if (!bbA.instructions.empty() && bbA.instructions.back().op == IROp::Bra &&
                        bbA.instructions.back().trueTarget == bbB.label) {
                        bbA.instructions.pop_back();
                    }
                    bbA.instructions.insert(bbA.instructions.end(),
                                           bbB.instructions.begin(),
                                           bbB.instructions.end());
                    fn.blocks.erase(fn.blocks.begin() + i + 1);
                    changed = true;
                    changedOverall = true;
                    break;
                }
            }
        }
    }

    return changedOverall;
}

bool Optimizer::runInstructionScheduling(IRFunction& fn) {
    bool changedOverall = false;
    auto sharedObjects = computeSharedObjects(fn, computePointerBases(fn));

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
            auto scheduled = scheduleWindow(window, sharedObjects);
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
        bool cseChanged = runCSE(fn);
        bool dceChanged = runDeadCodeElimination(fn);
        bool cfgChanged = runCFGCleanup(fn);
        if (cfChanged || cseChanged || dceChanged || cfgChanged) {
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

CallGraph CallGraph::build(const IRModule& mod) {
    CallGraph cg;
    std::unordered_set<std::string> knownFuncs;
    for (const auto& fn : mod.functions) {
        knownFuncs.insert(fn.name);
        cg.callersToCallees[fn.name]; // ensure key exists
    }
    for (const auto& fn : mod.functions) {
        for (const auto& bb : fn.blocks) {
            for (const auto& inst : bb.instructions) {
                if (inst.op == IROp::Call && knownFuncs.count(inst.symbol)) {
                    cg.callersToCallees[fn.name].insert(inst.symbol);
                }
            }
        }
    }
    return cg;
}

bool CallGraph::isRecursive(const std::string& funcName) const {
    auto it = callersToCallees.find(funcName);
    if (it == callersToCallees.end()) return false;

    // BFS from direct callees of funcName; if funcName is reachable, it participates in a cycle.
    std::unordered_set<std::string> visited;
    std::vector<std::string> queue;
    for (const auto& callee : it->second) {
        if (callee == funcName) return true; // Direct self-recursion
        queue.push_back(callee);
        visited.insert(callee);
    }

    while (!queue.empty()) {
        std::string curr = queue.back();
        queue.pop_back();

        auto cit = callersToCallees.find(curr);
        if (cit != callersToCallees.end()) {
            for (const auto& next : cit->second) {
                if (next == funcName) return true; // Path back to funcName (recursion/cycle)
                if (visited.insert(next).second) {
                    queue.push_back(next);
                }
            }
        }
    }
    return false;
}

namespace {

static int functionInstructionCount(const IRFunction& fn) {
    int count = 0;
    for (const auto& bb : fn.blocks) {
        count += static_cast<int>(bb.instructions.size());
    }
    return count;
}

static bool inlineCall(IRFunction& caller, size_t blockIdx, size_t instIdx, const IRFunction& callee) {
    auto& callerBB = caller.blocks[blockIdx];
    const auto callInst = callerBB.instructions[instIdx];
    assert(callInst.op == IROp::Call && callInst.symbol == callee.name);

    // 1. Check if caller has bool return ABI conversion following callInst:
    //    call abiReg = callee(...)
    //    ldimm zeroReg = 0
    //    setp.ne predReg = abiReg, zeroReg
    bool calleeReturnsBool = callee.returnType.isBool();
    VReg callerPredDst = 0;
    size_t extraConsumed = 0;
    if (calleeReturnsBool && callInst.dst != 0) {
        if (instIdx + 2 < callerBB.instructions.size()) {
            const auto& inst1 = callerBB.instructions[instIdx + 1];
            const auto& inst2 = callerBB.instructions[instIdx + 2];
            if (inst1.op == IROp::LoadImm && inst1.immInt == 0 &&
                inst2.op == IROp::SetpNe && inst2.src1 == callInst.dst && inst2.src2 == inst1.dst) {
                callerPredDst = inst2.dst;
                extraConsumed = 2;
            }
        }
    }

    // 2. Map callee virtual registers to fresh virtual registers in caller
    std::unordered_map<VReg, VReg> vregMap;
    vregMap[0] = 0;
    for (size_t r = 1; r < callee.regTypes.size(); ++r) {
        VReg fresh = caller.freshReg(callee.regTypes[r]);
        vregMap[static_cast<VReg>(r)] = fresh;
    }

    // 3. Map callee basic block labels to fresh labels in caller
    std::unordered_map<Label, Label> labelMap;
    for (const auto& bb : callee.blocks) {
        std::string hint = callee.name + "_" + (!bb.label.empty() && bb.label.front() == '$' ? bb.label.substr(1) : bb.label);
        labelMap[bb.label] = caller.freshLabel(hint);
    }
    Label contLabel = caller.freshLabel("inlined_cont");

    // 4. Identify parameter mappings and bool argument conversions
    std::unordered_map<size_t, VReg> calleeParamReg;
    std::unordered_map<size_t, VReg> calleeBoolParamPredReg;
    std::unordered_set<size_t> calleeSkippedEntryInsts;

    if (!callee.blocks.empty()) {
        const auto& entryInsts = callee.blocks[0].instructions;
        for (size_t p = 0; p < callee.params.size(); ++p) {
            for (size_t k = 0; k < entryInsts.size(); ++k) {
                const auto& ci = entryInsts[k];
                if (ci.op == IROp::LoadParam && ci.paramIndex == static_cast<int>(p)) {
                    calleeParamReg[p] = ci.dst;
                    calleeSkippedEntryInsts.insert(k);

                    if (callee.params[p].type.isBool() && k + 2 < entryInsts.size()) {
                        const auto& zInst = entryInsts[k + 1];
                        const auto& sInst = entryInsts[k + 2];
                        if (zInst.op == IROp::LoadImm && zInst.immInt == 0 &&
                            sInst.op == IROp::SetpNe && sInst.src1 == ci.dst && sInst.src2 == zInst.dst) {
                            calleeBoolParamPredReg[p] = sInst.dst;
                            calleeSkippedEntryInsts.insert(k + 1);
                            calleeSkippedEntryInsts.insert(k + 2);
                        }
                    }
                    break;
                }
            }
        }
    }

    // Check caller for cvt.pred2i for each bool param
    std::unordered_map<size_t, VReg> callerOrigPredArg;
    std::unordered_set<const IRInst*> callerSkippedArgCvts;
    for (size_t p = 0; p < callee.params.size(); ++p) {
        if (callee.params[p].type.isBool() && p < callInst.args.size()) {
            VReg abiArg = callInst.args[p];
            for (size_t k = 0; k < instIdx; ++k) {
                const auto& prev = callerBB.instructions[k];
                if (prev.op == IROp::CvtPredToS32 && prev.dst == abiArg) {
                    callerOrigPredArg[p] = prev.src1;
                    callerSkippedArgCvts.insert(&prev);
                    break;
                }
            }
        }
    }

    // 5. Generate parameter move instructions
    std::vector<IRInst> paramMoves;
    for (size_t p = 0; p < callee.params.size(); ++p) {
        if (p >= callInst.args.size()) break;
        VReg argReg = callInst.args[p];
        if (callee.params[p].type.isBool() && calleeBoolParamPredReg.count(p) && callerOrigPredArg.count(p)) {
            // Direct predicate copy without ABI roundtrip!
            VReg calleePred = calleeBoolParamPredReg[p];
            IRInst movInst{IROp::Mov, vregMap[calleePred], callerOrigPredArg[p], 0, IRType::Pred};
            movInst.srcLoc = callInst.srcLoc;
            paramMoves.push_back(movInst);
        } else if (calleeParamReg.count(p)) {
            VReg calleeP = calleeParamReg[p];
            IRType pType = callee.regType(calleeP);
            IRInst movInst{IROp::Mov, vregMap[calleeP], argReg, 0, pType};
            movInst.srcLoc = callInst.srcLoc;
            paramMoves.push_back(movInst);

            if (callee.params[p].type.isBool() && calleeBoolParamPredReg.count(p) && !callerOrigPredArg.count(p)) {
                VReg zeroR = caller.freshReg(IRType::S32);
                IRInst zInst{IROp::LoadImm, zeroR, 0, 0, IRType::S32};
                zInst.immInt = 0;
                zInst.srcLoc = callInst.srcLoc;
                paramMoves.push_back(zInst);
                VReg calleePred = calleeBoolParamPredReg[p];
                IRInst sInst{IROp::SetpNe, vregMap[calleePred], vregMap[calleeP], zeroR, IRType::S32};
                sInst.srcLoc = callInst.srcLoc;
                paramMoves.push_back(sInst);
            }
        }
    }

    // 6. Find bool return cvt.pred2i in callee
    std::unordered_map<VReg, VReg> calleeRetvalToPred;
    std::unordered_set<const IRInst*> calleeSkippedRetCvts;
    if (calleeReturnsBool && callerPredDst != 0) {
        for (const auto& bb : callee.blocks) {
            for (size_t k = 0; k + 1 < bb.instructions.size(); ++k) {
                const auto& cvt = bb.instructions[k];
                const auto& stRet = bb.instructions[k + 1];
                if (cvt.op == IROp::CvtPredToS32 && stRet.op == IROp::StoreRetval && stRet.src1 == cvt.dst) {
                    calleeRetvalToPred[stRet.src1] = cvt.src1;
                    calleeSkippedRetCvts.insert(&cvt);
                }
            }
        }
    }

    // 7. Transform callee basic blocks
    std::vector<BasicBlock> inlinedBlocks;
    for (size_t b = 0; b < callee.blocks.size(); ++b) {
        const auto& cbb = callee.blocks[b];
        BasicBlock newBB;
        newBB.label = labelMap[cbb.label];

        for (size_t k = 0; k < cbb.instructions.size(); ++k) {
            const auto& cInst = cbb.instructions[k];

            // Skip parameter load instructions in entry block
            if (b == 0 && calleeSkippedEntryInsts.count(k)) {
                continue;
            }

            // Skip cvt.pred2i before return in callee
            if (calleeSkippedRetCvts.count(&cInst)) {
                continue;
            }

            // Return value store
            if (cInst.op == IROp::StoreRetval) {
                if (calleeReturnsBool && callerPredDst != 0 && calleeRetvalToPred.count(cInst.src1)) {
                    VReg calleePred = calleeRetvalToPred[cInst.src1];
                    IRInst movInst{IROp::Mov, callerPredDst, vregMap[calleePred], 0, IRType::Pred};
                    movInst.srcLoc = cInst.srcLoc;
                    newBB.instructions.push_back(movInst);
                } else if (callInst.dst != 0) {
                    IRType retType = caller.regType(callInst.dst);
                    IRInst movInst{IROp::Mov, callInst.dst, vregMap[cInst.src1], 0, retType};
                    movInst.srcLoc = cInst.srcLoc;
                    newBB.instructions.push_back(movInst);
                }
                continue;
            }

            // Return instruction becomes branch to continuation
            if (cInst.op == IROp::Ret) {
                IRInst braInst{IROp::Bra, 0, 0, 0, IRType::S32};
                braInst.trueTarget = contLabel;
                braInst.srcLoc = cInst.srcLoc;
                newBB.instructions.push_back(braInst);
                continue;
            }

            // General instruction: remap registers and labels
            IRInst newInst = cInst;
            if (newInst.dst != 0) newInst.dst = vregMap[newInst.dst];
            if (newInst.src1 != 0) newInst.src1 = vregMap[newInst.src1];
            if (newInst.src2 != 0) newInst.src2 = vregMap[newInst.src2];
            if (newInst.src3 != 0) newInst.src3 = vregMap[newInst.src3];
            if (newInst.src4 != 0) newInst.src4 = vregMap[newInst.src4];
            for (auto& d : newInst.dsts) d = vregMap[d];
            for (auto& a : newInst.args) a = vregMap[a];
            if (!newInst.trueTarget.empty()) {
                auto lit = labelMap.find(newInst.trueTarget);
                if (lit != labelMap.end()) newInst.trueTarget = lit->second;
            }
            if (!newInst.falseTarget.empty()) {
                auto lit = labelMap.find(newInst.falseTarget);
                if (lit != labelMap.end()) newInst.falseTarget = lit->second;
            }
            newBB.instructions.push_back(newInst);
        }
        inlinedBlocks.push_back(std::move(newBB));
    }

    // 8. Split callerBB into head and tail
    BasicBlock headBB;
    headBB.label = callerBB.label;
    for (size_t k = 0; k < instIdx; ++k) {
        if (!callerSkippedArgCvts.count(&callerBB.instructions[k])) {
            headBB.instructions.push_back(callerBB.instructions[k]);
        }
    }
    for (const auto& pm : paramMoves) {
        headBB.instructions.push_back(pm);
    }
    if (!inlinedBlocks.empty()) {
        IRInst braEntry{IROp::Bra, 0, 0, 0, IRType::S32};
        braEntry.trueTarget = inlinedBlocks[0].label;
        braEntry.srcLoc = callInst.srcLoc;
        headBB.instructions.push_back(braEntry);
    } else {
        IRInst braCont{IROp::Bra, 0, 0, 0, IRType::S32};
        braCont.trueTarget = contLabel;
        braCont.srcLoc = callInst.srcLoc;
        headBB.instructions.push_back(braCont);
    }

    BasicBlock tailBB;
    tailBB.label = contLabel;
    for (size_t k = instIdx + 1 + extraConsumed; k < callerBB.instructions.size(); ++k) {
        tailBB.instructions.push_back(callerBB.instructions[k]);
    }

    // 9. Reconstruct caller's block list
    std::vector<BasicBlock> newBlocks;
    for (size_t b = 0; b < caller.blocks.size(); ++b) {
        if (b == blockIdx) {
            newBlocks.push_back(std::move(headBB));
            for (auto& ib : inlinedBlocks) {
                newBlocks.push_back(std::move(ib));
            }
            newBlocks.push_back(std::move(tailBB));
        } else {
            newBlocks.push_back(std::move(caller.blocks[b]));
        }
    }
    caller.blocks = std::move(newBlocks);
    return true;
}

} // anonymous namespace

bool Optimizer::runInlining(IRModule& mod, int threshold) {
    auto cg = CallGraph::build(mod);

    std::unordered_map<std::string, const IRFunction*> funcByName;
    for (const auto& fn : mod.functions) {
        funcByName[fn.name] = &fn;
    }

    std::unordered_set<std::string> eligible;
    for (const auto& fn : mod.functions) {
        if (!fn.isKernel && !cg.isRecursive(fn.name) &&
            functionInstructionCount(fn) <= threshold && fn.sharedVars.empty()) {
            eligible.insert(fn.name);
        }
    }

    if (eligible.empty()) return false;

    bool anyInlined = false;
    for (auto& caller : mod.functions) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t b = 0; b < caller.blocks.size(); ++b) {
                for (size_t i = 0; i < caller.blocks[b].instructions.size(); ++i) {
                    if (caller.blocks[b].instructions[i].op == IROp::Call) {
                        std::string calleeName = caller.blocks[b].instructions[i].symbol;
                        if (eligible.count(calleeName)) {
                            inlineCall(caller, b, i, *funcByName[calleeName]);
                            changed = true;
                            anyInlined = true;
                            break;
                        }
                    }
                }
                if (changed) break;
            }
        }
    }

    return anyInlined;
}

} // namespace kernc
