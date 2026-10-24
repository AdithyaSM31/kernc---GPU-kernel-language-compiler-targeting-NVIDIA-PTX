#include "kernc/IR.h"
#include <sstream>

namespace kernc {

const char* irTypeToString(IRType type) {
    switch (type) {
        case IRType::S32:  return "s32";
        case IRType::F32:  return "f32";
        case IRType::S64:  return "s64";
        case IRType::Pred: return "pred";
    }
    return "?";
}

const char* irOpToString(IROp op) {
    switch (op) {
        case IROp::Add:        return "add";
        case IROp::Sub:        return "sub";
        case IROp::Mul:        return "mul";
        case IROp::Div:        return "div";
        case IROp::Mod:        return "rem";
        case IROp::FAdd:       return "fadd";
        case IROp::FSub:       return "fsub";
        case IROp::FMul:       return "fmul";
        case IROp::FDiv:       return "fdiv";
        case IROp::Neg:        return "neg";
        case IROp::FNeg:       return "fneg";
        case IROp::CvtF2I:     return "cvt.f2i";
        case IROp::CvtI2F:     return "cvt.i2f";
        case IROp::SetpLt:     return "setp.lt";
        case IROp::SetpGt:     return "setp.gt";
        case IROp::SetpLe:     return "setp.le";
        case IROp::SetpGe:     return "setp.ge";
        case IROp::SetpEq:     return "setp.eq";
        case IROp::SetpNe:     return "setp.ne";
        case IROp::SetpFLt:    return "setp.flt";
        case IROp::SetpFGt:    return "setp.fgt";
        case IROp::SetpFLe:    return "setp.fle";
        case IROp::SetpFGe:    return "setp.fge";
        case IROp::SetpFEq:    return "setp.feq";
        case IROp::SetpFNe:    return "setp.fne";
        case IROp::Mov:        return "mov";
        case IROp::LoadImm:    return "ldimm";
        case IROp::LoadParam:  return "ldparam";
        case IROp::Load:       return "load";
        case IROp::Store:      return "store";
        case IROp::MulWide:    return "mul.wide";
        case IROp::AddS64:     return "add.s64";
        case IROp::CvtaGlobal: return "cvta.to.global";
        case IROp::Bra:        return "bra";
        case IROp::BraCond:    return "bra.cond";
        case IROp::Ret:        return "ret";
        case IROp::ReadSReg:   return "readsreg";
        case IROp::Bar:        return "bar.sync";
    }
    return "??";
}

std::string IRInst::toString() const {
    std::ostringstream ss;
    ss << irOpToString(op) << " ";
    if (dst != 0) {
        ss << "%v" << dst << ":" << irTypeToString(type) << " = ";
    }
    switch (op) {
        case IROp::LoadImm:
            if (type == IRType::F32) ss << immFloat;
            else ss << immInt;
            break;
        case IROp::LoadParam:
            ss << paramName << "[" << paramIndex << "]";
            break;
        case IROp::ReadSReg:
            ss << specialReg;
            break;
        case IROp::Bra:
            ss << trueTarget;
            break;
        case IROp::BraCond:
            ss << "%v" << src1 << " ? " << trueTarget << " : " << falseTarget;
            break;
        case IROp::Store:
            ss << "[%v" << src1 << "], %v" << src2;
            break;
        case IROp::Load:
            ss << "[%v" << src1 << "]";
            break;
        case IROp::Neg:
        case IROp::FNeg:
        case IROp::CvtF2I:
        case IROp::CvtI2F:
        case IROp::CvtaGlobal:
        case IROp::Mov:
            ss << "%v" << src1;
            break;
        case IROp::Ret:
        case IROp::Bar:
            break;
        default:
            ss << "%v" << src1 << ", %v" << src2;
            break;
    }
    return ss.str();
}

std::string IRFunction::dump() const {
    std::ostringstream ss;
    ss << "function " << name << " {\n";
    for (const auto& bb : blocks) {
        ss << bb.label << ":\n";
        for (const auto& inst : bb.instructions) {
            ss << "    " << inst.toString() << "\n";
        }
    }
    ss << "}\n";
    return ss.str();
}

std::string IRModule::dump() const {
    std::ostringstream ss;
    for (const auto& fn : functions) {
        ss << fn.dump() << "\n";
    }
    return ss.str();
}

} // namespace kernc
