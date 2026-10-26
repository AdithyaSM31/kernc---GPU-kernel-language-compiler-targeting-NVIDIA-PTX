#pragma once

#include "kernc/AST.h"
#include "kernc/SourceLocation.h"
#include <cstdint>
#include <string>
#include <vector>

namespace kernc {

using VReg = uint32_t;
using Label = std::string;

// Names kernc generates in the PTX. Each contains '$', which PTX identifiers allow but .kern
// identifiers do not, so a generated name can never collide with a kernel, function or array
// name from the source (e.g. a function called `entry` or `f_retval`).
namespace ptxname {
inline const char* kEntryLabel = "$entry";
inline std::string label(const std::string& hint, uint32_t n) { return "$" + hint + "_" + std::to_string(n); }
inline std::string param(const std::string& fn, size_t i) { return fn + "$param_" + std::to_string(i); }
inline std::string retval(const std::string& fn) { return fn + "$retval"; }
inline std::string shared(const std::string& fn, const std::string& array) { return fn + "$" + array; }
inline std::string callArg(size_t i) { return "$arg" + std::to_string(i); } // call-site .param
inline const char* kCallRet = "$ret";                                        // call-site .param
} // namespace ptxname

enum class IRType {
    S32,
    F32,
    S64,
    Pred
};

const char* irTypeToString(IRType type);

// PTX state space a Load/Store addresses. Global addresses are 64-bit; shared addresses are
// 32-bit offsets into the block's shared-memory window.
enum class AddrSpace {
    Global,
    Shared
};

enum class IROp {
    // Integer Arithmetic
    Add, Sub, Mul, Div, Mod,
    // Bitwise and Shifts
    And, Or, Xor, Not, Shl, ShrS32, ShrU32,
    // Float Arithmetic
    FAdd, FSub, FMul, FDiv,
    // Unary Negate
    Neg, FNeg,
    // Type Conversions
    CvtF2I, CvtI2F,
    CvtPredToS32, // dst:s32 = src1 ? 1 : 0       (selp.s32)
    CvtPredToF32, // dst:f32 = src1 ? 1.0 : 0.0   (selp.f32)
    // Predicate Logic
    NotPred,      // dst:pred = !src1
    XorPred,      // dst:pred = src1 != src2
    // Comparisons
    SetpLt, SetpGt, SetpLe, SetpGe, SetpEq, SetpNe,
    SetpFLt, SetpFGt, SetpFLe, SetpFGe, SetpFEq, SetpFNe,
    // Data Movement
    Mov,
    LoadImm,
    LoadParam,
    // Memory
    Load,
    Store,
    // Address Computation
    MulWide,    // dst:s64 = src1:s32 * src2:s32
    AddS64,     // dst:s64 = src1:s64 + src2:s64
    CvtaGlobal, // dst:s64 = cvta.to.global.u64 src:s64
    SharedAddr, // dst:s32 = address of the .shared array named by `symbol`
    // Control Flow
    Bra,
    BraCond,
    Ret,
    // Device Functions
    Call,        // dst (0 for void) = call `symbol`(args...); dst/args use ABI types (no .pred)
    StoreRetval, // st.param [<function>$retval], src1

    // GPU Specific
    ReadSReg,
    Bar
};

const char* irOpToString(IROp op);

struct IRInst {
    IROp op = IROp::Mov;
    VReg dst = 0;
    VReg src1 = 0;
    VReg src2 = 0;
    IRType type = IRType::S32;

    int32_t immInt = 0;
    float immFloat = 0.0f;

    Label trueTarget;
    Label falseTarget;

    std::string specialReg; // e.g. "%tid.x", "%ctaid.x", "%ntid.x"
    int paramIndex = 0;
    std::string paramName;

    AddrSpace space = AddrSpace::Global; // Load / Store
    std::string symbol;                  // SharedAddr: array symbol; Call: callee name
    std::vector<VReg> args;              // Call: argument registers, in order

    SourceLocation srcLoc;

    IRInst() = default;
    IRInst(IROp o, VReg d = 0, VReg s1 = 0, VReg s2 = 0, IRType t = IRType::S32)
        : op(o), dst(d), src1(s1), src2(s2), type(t) {}

    std::string toString() const;
};

struct BasicBlock {
    Label label;
    std::vector<IRInst> instructions;
};

// A statically sized array in .shared memory, emitted at module scope (as nvcc does).
struct SharedVar {
    std::string symbol; // unique within the module, e.g. "reduce_sum_sdata"
    IRType elemType;
    int64_t count;
};

struct IRFunction {
    bool isKernel = true;
    Type returnType = Type::makeVoid();
    std::string name;
    std::vector<Param> params;
    std::vector<BasicBlock> blocks;
    std::vector<SharedVar> sharedVars;

    uint32_t nextVReg = 1;
    uint32_t nextLabel = 0;

    // Type of each virtual register, indexed by VReg. Slot 0 is the "no register" sentinel.
    // A register's type is fixed when it is created; PTX declares registers per type.
    std::vector<IRType> regTypes{IRType::S32};

    VReg freshReg(IRType type) {
        regTypes.push_back(type);
        return nextVReg++;
    }

    IRType regType(VReg reg) const {
        return regTypes.at(reg);
    }

    Label freshLabel(const std::string& hint = "L") {
        return ptxname::label(hint, nextLabel++);
    }

    BasicBlock* currentBlock() {
        if (blocks.empty()) return nullptr;
        return &blocks.back();
    }

    void addBlock(const Label& label) {
        blocks.push_back(BasicBlock{label, {}});
    }

    // Source location stamped on every emitted instruction; IRGen sets it per statement.
    SourceLocation currentLoc;

    void emit(const IRInst& inst) {
        if (blocks.empty()) {
            addBlock(ptxname::kEntryLabel);
        }
        blocks.back().instructions.push_back(inst);
        blocks.back().instructions.back().srcLoc = currentLoc;
    }

    std::string dump() const;
};

struct IRModule {
    std::vector<IRFunction> functions;
    std::string dump() const;
};

} // namespace kernc
