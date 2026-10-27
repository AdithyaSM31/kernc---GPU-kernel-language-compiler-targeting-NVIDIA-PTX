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
        case IROp::And:        return "and";
        case IROp::Or:         return "or";
        case IROp::Xor:        return "xor";
        case IROp::Not:        return "not";
        case IROp::Shl:        return "shl";
        case IROp::ShrS32:     return "shr.s32";
        case IROp::ShrU32:     return "shr.u32";
        case IROp::FAdd:       return "fadd";
        case IROp::FSub:       return "fsub";
        case IROp::FMul:       return "fmul";
        case IROp::FDiv:       return "fdiv";
        case IROp::Neg:        return "neg";
        case IROp::FNeg:       return "fneg";
        case IROp::CvtF2I:     return "cvt.f2i";
        case IROp::CvtI2F:     return "cvt.i2f";
        case IROp::CvtPredToS32: return "cvt.pred2i";
        case IROp::CvtPredToF32: return "cvt.pred2f";
        case IROp::NotPred:    return "not.pred";
        case IROp::XorPred:    return "xor.pred";
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
        case IROp::SharedAddr: return "addr.shared";
        case IROp::Bra:      return "bra";
        case IROp::BraCond:    return "bra.cond";
        case IROp::Ret:        return "ret";
        case IROp::Call:       return "call";
        case IROp::StoreRetval: return "st.retval";
        case IROp::ReadSReg:   return "readsreg";
        case IROp::Bar:        return "bar.sync";
        case IROp::ShflIdx:    return "shfl.sync.idx";
        case IROp::ShflDown:   return "shfl.sync.down";
        case IROp::ShflUp:     return "shfl.sync.up";
        case IROp::ShflBfly:   return "shfl.sync.bfly";
        case IROp::VoteBallot: return "vote.sync.ballot";
        case IROp::Activemask: return "activemask";
        case IROp::AtomicAdd:  return "atom.add";
        case IROp::AtomicSub:  return "atom.sub";
        case IROp::AtomicMin:  return "atom.min";
        case IROp::AtomicMax:  return "atom.max";
        case IROp::AtomicAnd:  return "atom.and";
        case IROp::AtomicOr:   return "atom.or";
        case IROp::AtomicXor:  return "atom.xor";
        case IROp::AtomicExch: return "atom.exch";
        case IROp::AtomicCas:  return "atom.cas";
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
            ss << (space == AddrSpace::Shared ? "shared" : "global") << " [%v" << src1 << "], %v" << src2;
            break;
        case IROp::Load:
            ss << (space == AddrSpace::Shared ? "shared" : "global") << " [%v" << src1 << "]";
            break;
        case IROp::SharedAddr:
            ss << symbol;
            break;
        case IROp::Call:
            ss << symbol << "(";
            for (size_t i = 0; i < args.size(); ++i) ss << (i ? ", " : "") << "%v" << args[i];
            ss << ")";
            break;
        case IROp::StoreRetval:
            ss << "%v" << src1;
            break;
        case IROp::ShflIdx:
        case IROp::ShflDown:
        case IROp::ShflBfly:
        case IROp::ShflUp:
            ss << "%v" << src1 << ", %v" << src2 << ", " << immInt << ", %v" << src3;
            break;
        case IROp::VoteBallot:
            ss << "%v" << src1 << ", %v" << src2;
            break;
        case IROp::Activemask:
            break;
        case IROp::AtomicAdd:
        case IROp::AtomicSub:
        case IROp::AtomicMin:
        case IROp::AtomicMax:
        case IROp::AtomicAnd:
        case IROp::AtomicOr:
        case IROp::AtomicXor:
        case IROp::AtomicExch:
            ss << (space == AddrSpace::Shared ? "shared" : "global") << " [%v" << src1 << "], %v" << src2;
            break;
        case IROp::AtomicCas:
            ss << (space == AddrSpace::Shared ? "shared" : "global") << " [%v" << src1 << "], %v" << src2 << ", %v" << src3;
            break;
        case IROp::Neg:
        case IROp::FNeg:
        case IROp::Not:
        case IROp::CvtF2I:
        case IROp::CvtI2F:
        case IROp::CvtPredToS32:
        case IROp::CvtPredToF32:
        case IROp::NotPred:
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
