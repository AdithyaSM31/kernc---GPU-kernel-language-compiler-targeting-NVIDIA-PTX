#include "kernc/IRGen.h"

namespace kernc {

// Maps a source-language type to the IR register type that holds its values.
static IRType toIRType(const Type& t) {
    switch (t.kind) {
        case Type::Float: return IRType::F32;
        case Type::Ptr:   return IRType::S64;
        case Type::Bool:  return IRType::Pred;
        case Type::Int:
        case Type::Void:  break;
        case Type::SharedArray: break; // lives in memory, never in a register
    }
    return IRType::S32;
}

IRGen::IRGen(DiagnosticsEngine& diag) : diag_(diag) {}

IRModule IRGen::emitModule(const Program& program) {
    funcSignatures_.clear();
    for (const auto& func : program.functions) {
        std::vector<Type> pt;
        for (const auto& p : func->params) pt.push_back(p.type);
        funcSignatures_[func->name] = FuncInfo{func->isKernel, func->returnType, std::move(pt)};
    }
    IRModule mod;
    for (const auto& kernel : program.functions) {
        mod.functions.push_back(emitFunction(*kernel));
    }
    return mod;
}

IRFunction IRGen::emitFunction(const FunctionDecl& kernel) {
    if (funcSignatures_.find(kernel.name) == funcSignatures_.end()) {
        std::vector<Type> pt;
        for (const auto& p : kernel.params) pt.push_back(p.type);
        funcSignatures_[kernel.name] = FuncInfo{kernel.isKernel, kernel.returnType, std::move(pt)};
    }

    IRFunction fn;
    fn.isKernel = kernel.isKernel;
    fn.returnType = kernel.returnType;
    fn.name = kernel.name;
    fn.params = kernel.params;

    fn.addBlock(ptxname::kEntryLabel);
    enterScope();

    // Load parameters in prologue; attributed to the kernel's own line.
    fn.currentLoc = kernel.range.begin;
    for (size_t i = 0; i < kernel.params.size(); ++i) {
        const auto& param = kernel.params[i];
        std::string paramPtxName = ptxname::param(kernel.name, i);

        if (param.type.isPtr()) {
            if (kernel.isKernel) {
                VReg rawReg = fn.freshReg(IRType::S64);
                IRInst ldParam{IROp::LoadParam, rawReg, 0, 0, IRType::S64};
                ldParam.paramIndex = static_cast<int>(i);
                ldParam.paramName = paramPtxName;
                fn.emit(ldParam);

                VReg baseReg = fn.freshReg(IRType::S64);
                IRInst cvta{IROp::CvtaGlobal, baseReg, rawReg, 0, IRType::S64};
                fn.emit(cvta);

                setVar(param.name, baseReg, param.type);
            } else {
                VReg baseReg = fn.freshReg(IRType::S64);
                IRInst ldParam{IROp::LoadParam, baseReg, 0, 0, IRType::S64};
                ldParam.paramIndex = static_cast<int>(i);
                ldParam.paramName = paramPtxName;
                fn.emit(ldParam);

                setVar(param.name, baseReg, param.type);
            }
        } else if (param.type.isFloat()) {
            VReg pReg = fn.freshReg(IRType::F32);
            IRInst ldParam{IROp::LoadParam, pReg, 0, 0, IRType::F32};
            ldParam.paramIndex = static_cast<int>(i);
            ldParam.paramName = paramPtxName;
            fn.emit(ldParam);

            setVar(param.name, pReg, param.type);
        } else if (param.type.isBool()) {
            VReg s32Reg = fn.freshReg(IRType::S32);
            IRInst ldParam{IROp::LoadParam, s32Reg, 0, 0, IRType::S32};
            ldParam.paramIndex = static_cast<int>(i);
            ldParam.paramName = paramPtxName;
            fn.emit(ldParam);

            VReg predReg = fn.freshReg(IRType::Pred);
            VReg zeroReg = fn.freshReg(IRType::S32);
            IRInst zeroInst{IROp::LoadImm, zeroReg, 0, 0, IRType::S32};
            zeroInst.immInt = 0;
            fn.emit(zeroInst);
            fn.emit(IRInst{IROp::SetpNe, predReg, s32Reg, zeroReg, IRType::S32});

            setVar(param.name, predReg, param.type);
        } else {
            VReg pReg = fn.freshReg(IRType::S32);
            IRInst ldParam{IROp::LoadParam, pReg, 0, 0, IRType::S32};
            ldParam.paramIndex = static_cast<int>(i);
            ldParam.paramName = paramPtxName;
            fn.emit(ldParam);

            setVar(param.name, pReg, param.type);
        }
    }

    for (const auto& stmt : kernel.body) {
        emitStmt(*stmt, fn);
    }

    // Ensure the last block has a ret; like nvcc, it belongs to the closing brace.
    fn.currentLoc = kernel.range.end;
    IRInst retInst{IROp::Ret, 0, 0, 0, IRType::S32};
    fn.emit(retInst);

    exitScope();
    return fn;
}

void IRGen::emitStmt(const Stmt& stmt, IRFunction& fn) {
    // Everything this statement emits is attributed to it. Nested statements set their own
    // location and restore this one, so e.g. the branch that closes an `if` maps to the `if`.
    SourceLocation outer = fn.currentLoc;
    fn.currentLoc = stmt.range.begin;
    emitStmtBody(stmt, fn);
    fn.currentLoc = outer;
}

void IRGen::emitStmtBody(const Stmt& stmt, IRFunction& fn) {
    if (auto* varDecl = dynamic_cast<const VarDeclStmt*>(&stmt)) {
        emitVarDecl(*varDecl, fn);
    } else if (auto* assign = dynamic_cast<const AssignStmt*>(&stmt)) {
        emitAssign(*assign, fn);
    } else if (auto* ifStmt = dynamic_cast<const IfStmt*>(&stmt)) {
        emitIf(*ifStmt, fn);
    } else if (auto* whileStmt = dynamic_cast<const WhileStmt*>(&stmt)) {
        emitWhile(*whileStmt, fn);
    } else if (auto* forStmt = dynamic_cast<const ForStmt*>(&stmt)) {
        emitFor(*forStmt, fn);
    } else if (auto* block = dynamic_cast<const BlockStmt*>(&stmt)) {
        emitBlock(*block, fn);
    } else if (auto* exprStmt = dynamic_cast<const ExprStmt*>(&stmt)) {
        emitExpr(*exprStmt->expr, fn);
    } else if (dynamic_cast<const BarrierStmt*>(&stmt)) {
        fn.emit(IRInst{IROp::Bar, 0, 0, 0, IRType::S32});
    } else if (auto* shared = dynamic_cast<const SharedDeclStmt*>(&stmt)) {
        emitSharedDecl(*shared, fn);
    } else if (auto* ret = dynamic_cast<const ReturnStmt*>(&stmt)) {
        if (ret->value) {
            VReg valReg = emitExpr(*ret->value, fn);
            if (fn.returnType.isFloat() && ret->value->resolvedType.isInt()) {
                VReg cvtReg = fn.freshReg(IRType::F32);
                fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
                valReg = cvtReg;
            } else if (fn.returnType.isBool()) {
                VReg s32Reg = fn.freshReg(IRType::S32);
                fn.emit(IRInst{IROp::CvtPredToS32, s32Reg, valReg, 0, IRType::S32});
                valReg = s32Reg;
            }
            IRInst stRet{IROp::StoreRetval, 0, valReg, 0, fn.regType(valReg)};
            stRet.symbol = ptxname::retval(fn.name);
            fn.emit(stRet);
        }
        fn.emit(IRInst{IROp::Ret, 0, 0, 0, IRType::S32});
        fn.addBlock(fn.freshLabel("L_after_ret"));
    }
}

void IRGen::emitVarDecl(const VarDeclStmt& stmt, IRFunction& fn) {
    VReg varReg = fn.freshReg(toIRType(stmt.type));
    setVar(stmt.name, varReg, stmt.type);

    if (stmt.init) {
        VReg valReg = emitExpr(*stmt.init, fn);
        if (stmt.type.isFloat() && stmt.init->resolvedType.isInt()) {
            VReg cvtReg = fn.freshReg(IRType::F32);
            fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
            valReg = cvtReg;
        }
        fn.emit(IRInst{IROp::Mov, varReg, valReg, 0, toIRType(stmt.type)});
    }
}

void IRGen::emitAssign(const AssignStmt& stmt, IRFunction& fn) {
    if (auto* ident = dynamic_cast<const IdentExpr*>(stmt.target.get())) {
        const VarInfo* vInfo = getVar(ident->name);
        VReg valReg = emitExpr(*stmt.value, fn);
        if (vInfo->type.isFloat() && stmt.value->resolvedType.isInt()) {
            VReg cvtReg = fn.freshReg(IRType::F32);
            fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
            valReg = cvtReg;
        }
        fn.emit(IRInst{IROp::Mov, vInfo->reg, valReg, 0, toIRType(vInfo->type)});
    } else if (auto* idx = dynamic_cast<const IndexExpr*>(stmt.target.get())) {
        AddrSpace space;
        IRType elemType;
        VReg addrReg = emitElementAddress(*idx, fn, space, elemType);

        VReg valReg = emitExpr(*stmt.value, fn);
        if (elemType == IRType::F32 && stmt.value->resolvedType.isInt()) {
            VReg cvtReg = fn.freshReg(IRType::F32);
            fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
            valReg = cvtReg;
        }
        IRInst store{IROp::Store, 0, addrReg, valReg, elemType};
        store.space = space;
        fn.emit(store);
    }
}

void IRGen::emitSharedDecl(const SharedDeclStmt& stmt, IRFunction& fn) {
    // Module-scope symbol <kernel>$<array>. The '$' keeps it apart from every source name and
    // from other kernels' arrays (kernel names cannot contain '$'). A second array with the
    // same name in another scope of this kernel gets a "$<n>" suffix.
    std::string symbol = ptxname::shared(fn.name, stmt.name);
    auto taken = [&](const std::string& s) {
        for (const auto& sv : fn.sharedVars)
            if (sv.symbol == s) return true;
        return false;
    };
    for (int n = 1; taken(symbol); ++n) {
        symbol = ptxname::shared(fn.name, stmt.name) + "$" + std::to_string(n);
    }
    fn.sharedVars.push_back({symbol, toIRType(stmt.elemType), stmt.size});
    setVar(stmt.name, 0, Type::makeSharedArray(stmt.elemType.kind), symbol);
}

VReg IRGen::emitElementAddress(const IndexExpr& idx, IRFunction& fn, AddrSpace& space, IRType& elemType) {
    auto* baseIdent = dynamic_cast<const IdentExpr*>(idx.base.get());
    const VarInfo* baseInfo = getVar(baseIdent->name);
    elemType = baseInfo->type.pointeeKind == Type::Float ? IRType::F32 : IRType::S32;

    VReg idxReg = emitExpr(*idx.index, fn);
    VReg szReg = fn.freshReg(IRType::S32);
    IRInst ldimm{IROp::LoadImm, szReg, 0, 0, IRType::S32};
    ldimm.immInt = 4;
    fn.emit(ldimm);

    if (baseInfo->type.isSharedArray()) {
        // 32-bit arithmetic: shared addresses are offsets into the block's shared window.
        space = AddrSpace::Shared;
        VReg baseReg = fn.freshReg(IRType::S32);
        IRInst addrOf{IROp::SharedAddr, baseReg, 0, 0, IRType::S32};
        addrOf.symbol = baseInfo->sharedSymbol;
        fn.emit(addrOf);
        VReg offReg = fn.freshReg(IRType::S32);
        fn.emit(IRInst{IROp::Mul, offReg, idxReg, szReg, IRType::S32});
        VReg addrReg = fn.freshReg(IRType::S32);
        fn.emit(IRInst{IROp::Add, addrReg, baseReg, offReg, IRType::S32});
        return addrReg;
    }

    // Global pointer: widen the 32-bit index while scaling it, then add the 64-bit base.
    space = AddrSpace::Global;
    VReg offReg = fn.freshReg(IRType::S64);
    fn.emit(IRInst{IROp::MulWide, offReg, idxReg, szReg, IRType::S64});
    VReg addrReg = fn.freshReg(IRType::S64);
    fn.emit(IRInst{IROp::AddS64, addrReg, baseInfo->reg, offReg, IRType::S64});
    return addrReg;
}

void IRGen::emitIf(const IfStmt& stmt, IRFunction& fn) {
    Label thenLabel = fn.freshLabel("L_then");
    Label elseLabel = stmt.elseBranch ? fn.freshLabel("L_else") : "";
    Label endLabel = fn.freshLabel("L_if_end");

    emitCondition(*stmt.cond, thenLabel, stmt.elseBranch ? elseLabel : endLabel, fn);

    fn.addBlock(thenLabel);
    emitStmt(*stmt.thenBranch, fn);
    IRInst braEnd{IROp::Bra, 0, 0, 0, IRType::S32};
    braEnd.trueTarget = endLabel;
    fn.emit(braEnd);

    if (stmt.elseBranch) {
        fn.addBlock(elseLabel);
        emitStmt(*stmt.elseBranch, fn);
        IRInst braElseEnd{IROp::Bra, 0, 0, 0, IRType::S32};
        braElseEnd.trueTarget = endLabel;
        fn.emit(braElseEnd);
    }

    fn.addBlock(endLabel);
}

void IRGen::emitWhile(const WhileStmt& stmt, IRFunction& fn) {
    Label condLabel = fn.freshLabel("L_while_cond");
    Label bodyLabel = fn.freshLabel("L_while_body");
    Label endLabel = fn.freshLabel("L_while_end");

    IRInst braCond{IROp::Bra, 0, 0, 0, IRType::S32};
    braCond.trueTarget = condLabel;
    fn.emit(braCond);

    fn.addBlock(bodyLabel);
    emitStmt(*stmt.body, fn);
    IRInst braLoop{IROp::Bra, 0, 0, 0, IRType::S32};
    braLoop.trueTarget = condLabel;
    fn.emit(braLoop);

    fn.addBlock(condLabel);
    SourceLocation loopLoc = fn.currentLoc;
    fn.currentLoc = stmt.cond->range.begin;
    emitCondition(*stmt.cond, bodyLabel, endLabel, fn);
    fn.currentLoc = loopLoc;

    fn.addBlock(endLabel);
}

void IRGen::emitFor(const ForStmt& stmt, IRFunction& fn) {
    enterScope();
    if (stmt.init) {
        emitStmt(*stmt.init, fn);
    }

    Label condLabel = fn.freshLabel("L_for_cond");
    Label bodyLabel = fn.freshLabel("L_for_body");
    Label endLabel = fn.freshLabel("L_for_end");

    IRInst braCond{IROp::Bra, 0, 0, 0, IRType::S32};
    braCond.trueTarget = condLabel;
    fn.emit(braCond);

    fn.addBlock(bodyLabel);
    emitStmt(*stmt.body, fn);
    if (stmt.step) {
        emitStmt(*stmt.step, fn);
    }
    IRInst braLoop{IROp::Bra, 0, 0, 0, IRType::S32};
    braLoop.trueTarget = condLabel;
    fn.emit(braLoop);

    fn.addBlock(condLabel);
    if (stmt.cond) {
        // The condition is emitted after the body, so give it its own location back.
        SourceLocation loopLoc = fn.currentLoc;
        fn.currentLoc = stmt.cond->range.begin;
        emitCondition(*stmt.cond, bodyLabel, endLabel, fn);
        fn.currentLoc = loopLoc;
    } else {
        IRInst braInf{IROp::Bra, 0, 0, 0, IRType::S32};
        braInf.trueTarget = bodyLabel;
        fn.emit(braInf);
    }

    fn.addBlock(endLabel);
    exitScope();
}

void IRGen::emitBlock(const BlockStmt& stmt, IRFunction& fn) {
    enterScope();
    for (const auto& s : stmt.stmts) {
        emitStmt(*s, fn);
    }
    exitScope();
}

void IRGen::emitCondition(const Expr& cond, const Label& trueLabel, const Label& falseLabel, IRFunction& fn) {
    if (auto* bin = dynamic_cast<const BinaryExpr*>(&cond)) {
        if (bin->op == BinaryOp::And) {
            Label rhsLabel = fn.freshLabel("L_and_rhs");
            emitCondition(*bin->lhs, rhsLabel, falseLabel, fn);
            fn.addBlock(rhsLabel);
            emitCondition(*bin->rhs, trueLabel, falseLabel, fn);
            return;
        }
        if (bin->op == BinaryOp::Or) {
            Label rhsLabel = fn.freshLabel("L_or_rhs");
            emitCondition(*bin->lhs, trueLabel, rhsLabel, fn);
            fn.addBlock(rhsLabel);
            emitCondition(*bin->rhs, trueLabel, falseLabel, fn);
            return;
        }

        if (isComparison(bin->op)) {
            emitBranch(emitCompare(*bin, fn), trueLabel, falseLabel, fn);
            return;
        }
        // Arithmetic (e.g. `if (n % 2)`) is an ordinary scalar condition, handled below.
    }

    if (auto* unary = dynamic_cast<const UnaryExpr*>(&cond)) {
        if (unary->op == UnaryOp::Not) {
            emitCondition(*unary->operand, falseLabel, trueLabel, fn);
            return;
        }
    }

    // Any other scalar: a bool is already a predicate, an int is tested against zero.
    emitBranch(emitAsPred(cond, fn), trueLabel, falseLabel, fn);
}

void IRGen::emitBranch(VReg pred, const Label& trueLabel, const Label& falseLabel, IRFunction& fn) {
    IRInst braCond{IROp::BraCond, 0, pred, 0, IRType::Pred};
    braCond.trueTarget = trueLabel;
    braCond.falseTarget = falseLabel;
    fn.emit(braCond);
}

bool IRGen::isComparison(BinaryOp op) {
    switch (op) {
        case BinaryOp::Lt: case BinaryOp::Gt: case BinaryOp::Le:
        case BinaryOp::Ge: case BinaryOp::Eq: case BinaryOp::Ne:
            return true;
        default:
            return false;
    }
}

// Evaluates a comparison into a fresh .pred register.
VReg IRGen::emitCompare(const BinaryExpr& bin, IRFunction& fn) {
    VReg r1 = emitExpr(*bin.lhs, fn);
    VReg r2 = emitExpr(*bin.rhs, fn);

    // bool == bool / bool != bool: setp has no .pred form, so use xor (and not for ==).
    if (bin.lhs->resolvedType.isBool()) {
        VReg differ = fn.freshReg(IRType::Pred);
        fn.emit(IRInst{IROp::XorPred, differ, r1, r2, IRType::Pred});
        if (bin.op == BinaryOp::Ne) return differ;
        VReg same = fn.freshReg(IRType::Pred);
        fn.emit(IRInst{IROp::NotPred, same, differ, 0, IRType::Pred});
        return same;
    }

    bool isFloat = false;
    if (bin.lhs->resolvedType.isFloat() || bin.rhs->resolvedType.isFloat()) {
        isFloat = true;
        if (bin.lhs->resolvedType.isInt()) {
            VReg cvt = fn.freshReg(IRType::F32);
            fn.emit(IRInst{IROp::CvtI2F, cvt, r1, 0, IRType::F32});
            r1 = cvt;
        }
        if (bin.rhs->resolvedType.isInt()) {
            VReg cvt = fn.freshReg(IRType::F32);
            fn.emit(IRInst{IROp::CvtI2F, cvt, r2, 0, IRType::F32});
            r2 = cvt;
        }
    }

    IROp setpOp;
    if (isFloat) {
        switch (bin.op) {
            case BinaryOp::Lt: setpOp = IROp::SetpFLt; break;
            case BinaryOp::Gt: setpOp = IROp::SetpFGt; break;
            case BinaryOp::Le: setpOp = IROp::SetpFLe; break;
            case BinaryOp::Ge: setpOp = IROp::SetpFGe; break;
            case BinaryOp::Eq: setpOp = IROp::SetpFEq; break;
            default:           setpOp = IROp::SetpFNe; break;
        }
    } else {
        switch (bin.op) {
            case BinaryOp::Lt: setpOp = IROp::SetpLt; break;
            case BinaryOp::Gt: setpOp = IROp::SetpGt; break;
            case BinaryOp::Le: setpOp = IROp::SetpLe; break;
            case BinaryOp::Ge: setpOp = IROp::SetpGe; break;
            case BinaryOp::Eq: setpOp = IROp::SetpEq; break;
            default:           setpOp = IROp::SetpNe; break;
        }
    }

    VReg pred = fn.freshReg(IRType::Pred);
    fn.emit(IRInst{setpOp, pred, r1, r2, IRType::Pred});
    return pred;
}

// "Truthiness": returns a .pred register that is true when the scalar is non-zero.
VReg IRGen::emitAsPred(const Expr& expr, IRFunction& fn) {
    VReg r = emitExpr(expr, fn);
    const Type& t = expr.resolvedType;
    if (t.isBool()) return r;

    bool isFloat = t.isFloat();
    VReg zero = fn.freshReg(isFloat ? IRType::F32 : IRType::S32);
    IRInst ldimm{IROp::LoadImm, zero, 0, 0, isFloat ? IRType::F32 : IRType::S32};
    fn.emit(ldimm);
    VReg pred = fn.freshReg(IRType::Pred);
    // SetpFNe lowers to setp.neu, so NaN counts as true, as in C.
    fn.emit(IRInst{isFloat ? IROp::SetpFNe : IROp::SetpNe, pred, r, zero, IRType::Pred});
    return pred;
}

// && and || used as values: reuse the short-circuit branches, then write true/false into the
// result on each path. and.pred/or.pred would evaluate both sides, so `i < n && x[i] > 0`
// would load x[i] out of bounds.
VReg IRGen::emitShortCircuitValue(const BinaryExpr& expr, IRFunction& fn) {
    VReg dst = fn.freshReg(IRType::Pred);
    Label trueLabel = fn.freshLabel("L_bool_true");
    Label falseLabel = fn.freshLabel("L_bool_false");
    Label endLabel = fn.freshLabel("L_bool_end");

    emitCondition(expr, trueLabel, falseLabel, fn);
    for (const auto& [label, value] : {std::pair{trueLabel, 1}, std::pair{falseLabel, 0}}) {
        fn.addBlock(label);
        IRInst set{IROp::LoadImm, dst, 0, 0, IRType::Pred};
        set.immInt = value;
        fn.emit(set);
        IRInst bra{IROp::Bra, 0, 0, 0, IRType::S32};
        bra.trueTarget = endLabel;
        fn.emit(bra);
    }
    fn.addBlock(endLabel);
    return dst;
}

VReg IRGen::emitExpr(const Expr& expr, IRFunction& fn) {
    if (auto* intLit = dynamic_cast<const IntLiteralExpr*>(&expr)) {
        VReg r = fn.freshReg(IRType::S32);
        IRInst inst{IROp::LoadImm, r, 0, 0, IRType::S32};
        inst.immInt = static_cast<int32_t>(intLit->value);
        fn.emit(inst);
        return r;
    }
    if (auto* floatLit = dynamic_cast<const FloatLiteralExpr*>(&expr)) {
        VReg r = fn.freshReg(IRType::F32);
        IRInst inst{IROp::LoadImm, r, 0, 0, IRType::F32};
        inst.immFloat = static_cast<float>(floatLit->value);
        fn.emit(inst);
        return r;
    }
    if (auto* boolLit = dynamic_cast<const BoolLiteralExpr*>(&expr)) {
        VReg r = fn.freshReg(IRType::Pred);
        IRInst inst{IROp::LoadImm, r, 0, 0, IRType::Pred};
        inst.immInt = boolLit->value ? 1 : 0;
        fn.emit(inst);
        return r;
    }
    if (auto* builtin = dynamic_cast<const BuiltinExpr*>(&expr)) {
        return emitBuiltin(*builtin, fn);
    }
    if (auto* ident = dynamic_cast<const IdentExpr*>(&expr)) {
        const VarInfo* vInfo = getVar(ident->name);
        return vInfo->reg;
    }
    if (auto* bin = dynamic_cast<const BinaryExpr*>(&expr)) {
        return emitBinaryExpr(*bin, fn);
    }
    if (auto* unary = dynamic_cast<const UnaryExpr*>(&expr)) {
        return emitUnaryExpr(*unary, fn);
    }
    if (auto* idx = dynamic_cast<const IndexExpr*>(&expr)) {
        return emitIndexExpr(*idx, fn);
    }
    if (auto* cast = dynamic_cast<const CastExpr*>(&expr)) {
        return emitCastExpr(*cast, fn);
    }
    if (auto* call = dynamic_cast<const CallExpr*>(&expr)) {
        return emitCallExpr(*call, fn);
    }
    return 0;
}

VReg IRGen::emitBuiltin(const BuiltinExpr& expr, IRFunction& fn) {
    VReg r = fn.freshReg(IRType::S32);
    IRInst inst{IROp::ReadSReg, r, 0, 0, IRType::S32};
    switch (expr.builtin) {
        case BuiltinKind::ThreadIdxX: inst.specialReg = "%tid.x"; break;
        case BuiltinKind::ThreadIdxY: inst.specialReg = "%tid.y"; break;
        case BuiltinKind::ThreadIdxZ: inst.specialReg = "%tid.z"; break;
        case BuiltinKind::BlockIdxX:  inst.specialReg = "%ctaid.x"; break;
        case BuiltinKind::BlockIdxY:  inst.specialReg = "%ctaid.y"; break;
        case BuiltinKind::BlockIdxZ:  inst.specialReg = "%ctaid.z"; break;
        case BuiltinKind::BlockDimX:  inst.specialReg = "%ntid.x"; break;
        case BuiltinKind::BlockDimY:  inst.specialReg = "%ntid.y"; break;
        case BuiltinKind::BlockDimZ:  inst.specialReg = "%ntid.z"; break;
        case BuiltinKind::GridDimX:   inst.specialReg = "%nctaid.x"; break;
        case BuiltinKind::GridDimY:   inst.specialReg = "%nctaid.y"; break;
        case BuiltinKind::GridDimZ:   inst.specialReg = "%nctaid.z"; break;
    }
    fn.emit(inst);
    return r;
}

VReg IRGen::emitBinaryExpr(const BinaryExpr& expr, IRFunction& fn) {
    if (expr.op == BinaryOp::And || expr.op == BinaryOp::Or) {
        return emitShortCircuitValue(expr, fn);
    }
    if (isComparison(expr.op)) {
        return emitCompare(expr, fn);
    }

    VReg r1 = emitExpr(*expr.lhs, fn);
    VReg r2 = emitExpr(*expr.rhs, fn);

    bool isFloat = (expr.resolvedType.isFloat());
    if (isFloat) {
        if (expr.lhs->resolvedType.isInt()) {
            VReg cvt = fn.freshReg(IRType::F32);
            fn.emit(IRInst{IROp::CvtI2F, cvt, r1, 0, IRType::F32});
            r1 = cvt;
        }
        if (expr.rhs->resolvedType.isInt()) {
            VReg cvt = fn.freshReg(IRType::F32);
            fn.emit(IRInst{IROp::CvtI2F, cvt, r2, 0, IRType::F32});
            r2 = cvt;
        }
    }

    IRType t = isFloat ? IRType::F32 : IRType::S32;
    VReg dst = fn.freshReg(t);
    IROp op;

    if (isFloat) {
        switch (expr.op) {
            case BinaryOp::Add: op = IROp::FAdd; break;
            case BinaryOp::Sub: op = IROp::FSub; break;
            case BinaryOp::Mul: op = IROp::FMul; break;
            case BinaryOp::Div: op = IROp::FDiv; break;
            default: op = IROp::FAdd; break;
        }
    } else {
        switch (expr.op) {
            case BinaryOp::Add:    op = IROp::Add; break;
            case BinaryOp::Sub:    op = IROp::Sub; break;
            case BinaryOp::Mul:    op = IROp::Mul; break;
            case BinaryOp::Div:    op = IROp::Div; break;
            case BinaryOp::Mod:    op = IROp::Mod; break;
            case BinaryOp::BitAnd: op = IROp::And; break;
            case BinaryOp::BitOr:  op = IROp::Or; break;
            case BinaryOp::BitXor: op = IROp::Xor; break;
            case BinaryOp::Shl:    op = IROp::Shl; break;
            case BinaryOp::Shr:    op = IROp::ShrS32; break;
            default: op = IROp::Add; break;
        }
    }

    fn.emit(IRInst{op, dst, r1, r2, t});
    return dst;
}

VReg IRGen::emitUnaryExpr(const UnaryExpr& expr, IRFunction& fn) {
    if (expr.op == UnaryOp::Not) {
        VReg operand = emitAsPred(*expr.operand, fn); // accepts bool or int
        VReg dst = fn.freshReg(IRType::Pred);
        fn.emit(IRInst{IROp::NotPred, dst, operand, 0, IRType::Pred});
        return dst;
    }
    if (expr.op == UnaryOp::BitNot) {
        VReg sub = emitExpr(*expr.operand, fn);
        VReg dst = fn.freshReg(IRType::S32);
        fn.emit(IRInst{IROp::Not, dst, sub, 0, IRType::S32});
        return dst;
    }

    VReg sub = emitExpr(*expr.operand, fn);
    VReg dst = fn.freshReg(toIRType(expr.resolvedType));
    if (expr.op == UnaryOp::Neg) {
        if (expr.resolvedType.isFloat()) {
            fn.emit(IRInst{IROp::FNeg, dst, sub, 0, IRType::F32});
        } else {
            fn.emit(IRInst{IROp::Neg, dst, sub, 0, IRType::S32});
        }
    }
    return dst;
}

VReg IRGen::emitIndexExpr(const IndexExpr& expr, IRFunction& fn) {
    AddrSpace space;
    IRType loadType;
    VReg addrReg = emitElementAddress(expr, fn, space, loadType);
    VReg dstReg = fn.freshReg(loadType);
    IRInst load{IROp::Load, dstReg, addrReg, 0, loadType};
    load.space = space;
    fn.emit(load);
    return dstReg;
}

VReg IRGen::emitCastExpr(const CastExpr& expr, IRFunction& fn) {
    if (expr.targetType.isBool()) {
        return emitAsPred(*expr.operand, fn); // x != 0 (NaN counts as true)
    }
    VReg sub = emitExpr(*expr.operand, fn);
    VReg dst = fn.freshReg(toIRType(expr.targetType));
    if (expr.operand->resolvedType.isBool()) {
        IROp op = expr.targetType.isFloat() ? IROp::CvtPredToF32 : IROp::CvtPredToS32;
        fn.emit(IRInst{op, dst, sub, 0, toIRType(expr.targetType)});
    } else if (expr.targetType.isFloat() && expr.operand->resolvedType.isInt()) {
        fn.emit(IRInst{IROp::CvtI2F, dst, sub, 0, IRType::F32});
    } else if (expr.targetType.isInt() && expr.operand->resolvedType.isFloat()) {
        fn.emit(IRInst{IROp::CvtF2I, dst, sub, 0, IRType::S32});
    } else {
        IRType t = expr.targetType.isFloat() ? IRType::F32 : IRType::S32;
        fn.emit(IRInst{IROp::Mov, dst, sub, 0, t});
    }
    return dst;
}

VReg IRGen::emitCallExpr(const CallExpr& expr, IRFunction& fn) {
    auto it = funcSignatures_.find(expr.callee);
    std::vector<VReg> argRegs;
    for (size_t i = 0; i < expr.args.size(); ++i) {
        VReg argReg = emitExpr(*expr.args[i], fn);
        if (it != funcSignatures_.end() && i < it->second.paramTypes.size()) {
            Type expectedType = it->second.paramTypes[i];
            if (expectedType.isFloat() && expr.args[i]->resolvedType.isInt()) {
                VReg cvt = fn.freshReg(IRType::F32);
                fn.emit(IRInst{IROp::CvtI2F, cvt, argReg, 0, IRType::F32});
                argReg = cvt;
            } else if (expectedType.isBool()) {
                VReg s32 = fn.freshReg(IRType::S32);
                fn.emit(IRInst{IROp::CvtPredToS32, s32, argReg, 0, IRType::S32});
                argReg = s32;
            }
        }
        argRegs.push_back(argReg);
    }

    Type retType = (it != funcSignatures_.end()) ? it->second.returnType : expr.resolvedType;
    if (retType.isVoid()) {
        IRInst callInst{IROp::Call, 0, 0, 0, IRType::S32};
        callInst.symbol = expr.callee;
        callInst.args = std::move(argRegs);
        fn.emit(callInst);
        return 0;
    }

    if (retType.isBool()) {
        VReg abiReg = fn.freshReg(IRType::S32);
        IRInst callInst{IROp::Call, abiReg, 0, 0, IRType::S32};
        callInst.symbol = expr.callee;
        callInst.args = std::move(argRegs);
        fn.emit(callInst);

        VReg predReg = fn.freshReg(IRType::Pred);
        VReg zeroReg = fn.freshReg(IRType::S32);
        IRInst zeroInst{IROp::LoadImm, zeroReg, 0, 0, IRType::S32};
        zeroInst.immInt = 0;
        fn.emit(zeroInst);
        fn.emit(IRInst{IROp::SetpNe, predReg, abiReg, zeroReg, IRType::S32});
        return predReg;
    }

    IRType irRet = retType.isFloat() ? IRType::F32 : IRType::S32;
    VReg dstReg = fn.freshReg(irRet);
    IRInst callInst{IROp::Call, dstReg, 0, 0, irRet};
    callInst.symbol = expr.callee;
    callInst.args = std::move(argRegs);
    fn.emit(callInst);
    return dstReg;
}

} // namespace kernc
