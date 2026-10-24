#include "kernc/IRGen.h"

namespace kernc {

IRGen::IRGen(DiagnosticsEngine& diag) : diag_(diag) {}

IRModule IRGen::emitModule(const Program& program) {
    IRModule mod;
    for (const auto& kernel : program.kernels) {
        mod.functions.push_back(emitKernel(*kernel));
    }
    return mod;
}

IRFunction IRGen::emitKernel(const KernelDecl& kernel) {
    IRFunction fn;
    fn.name = kernel.name;
    fn.params = kernel.params;

    fn.addBlock("entry");
    enterScope();

    // Load parameters in prologue
    for (size_t i = 0; i < kernel.params.size(); ++i) {
        const auto& param = kernel.params[i];
        std::string paramPtxName = kernel.name + "_param_" + std::to_string(i);

        if (param.type.isPtr()) {
            VReg rawReg = fn.freshReg();
            IRInst ldParam{IROp::LoadParam, rawReg, 0, 0, IRType::S64};
            ldParam.paramIndex = static_cast<int>(i);
            ldParam.paramName = paramPtxName;
            ldParam.srcLoc = param.loc;
            fn.emit(ldParam);

            VReg baseReg = fn.freshReg();
            IRInst cvta{IROp::CvtaGlobal, baseReg, rawReg, 0, IRType::S64};
            cvta.srcLoc = param.loc;
            fn.emit(cvta);

            setVar(param.name, baseReg, param.type);
        } else if (param.type.isFloat()) {
            VReg pReg = fn.freshReg();
            IRInst ldParam{IROp::LoadParam, pReg, 0, 0, IRType::F32};
            ldParam.paramIndex = static_cast<int>(i);
            ldParam.paramName = paramPtxName;
            ldParam.srcLoc = param.loc;
            fn.emit(ldParam);

            setVar(param.name, pReg, param.type);
        } else {
            VReg pReg = fn.freshReg();
            IRInst ldParam{IROp::LoadParam, pReg, 0, 0, IRType::S32};
            ldParam.paramIndex = static_cast<int>(i);
            ldParam.paramName = paramPtxName;
            ldParam.srcLoc = param.loc;
            fn.emit(ldParam);

            setVar(param.name, pReg, param.type);
        }
    }

    for (const auto& stmt : kernel.body) {
        emitStmt(*stmt, fn);
    }

    // Ensure the last block has a ret
    IRInst retInst{IROp::Ret, 0, 0, 0, IRType::S32};
    fn.emit(retInst);

    exitScope();
    return fn;
}

void IRGen::emitStmt(const Stmt& stmt, IRFunction& fn) {
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
    }
}

void IRGen::emitVarDecl(const VarDeclStmt& stmt, IRFunction& fn) {
    VReg varReg = fn.freshReg();
    setVar(stmt.name, varReg, stmt.type);

    if (stmt.init) {
        VReg valReg = emitExpr(*stmt.init, fn);
        if (stmt.type.isFloat() && stmt.init->resolvedType.isInt()) {
            VReg cvtReg = fn.freshReg();
            fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
            valReg = cvtReg;
        }
        IRType t = stmt.type.isFloat() ? IRType::F32 : IRType::S32;
        fn.emit(IRInst{IROp::Mov, varReg, valReg, 0, t});
    }
}

void IRGen::emitAssign(const AssignStmt& stmt, IRFunction& fn) {
    if (auto* ident = dynamic_cast<const IdentExpr*>(stmt.target.get())) {
        const VarInfo* vInfo = getVar(ident->name);
        VReg valReg = emitExpr(*stmt.value, fn);
        if (vInfo->type.isFloat() && stmt.value->resolvedType.isInt()) {
            VReg cvtReg = fn.freshReg();
            fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
            valReg = cvtReg;
        }
        IRType t = vInfo->type.isFloat() ? IRType::F32 : IRType::S32;
        fn.emit(IRInst{IROp::Mov, vInfo->reg, valReg, 0, t});
    } else if (auto* idx = dynamic_cast<const IndexExpr*>(stmt.target.get())) {
        auto* baseIdent = dynamic_cast<const IdentExpr*>(idx->base.get());
        const VarInfo* baseInfo = getVar(baseIdent->name);

        VReg idxReg = emitExpr(*idx->index, fn);
        VReg szReg = fn.freshReg();
        IRInst ldimm{IROp::LoadImm, szReg, 0, 0, IRType::S32};
        ldimm.immInt = 4;
        fn.emit(ldimm);

        VReg offReg = fn.freshReg();
        fn.emit(IRInst{IROp::MulWide, offReg, idxReg, szReg, IRType::S64});

        VReg addrReg = fn.freshReg();
        fn.emit(IRInst{IROp::AddS64, addrReg, baseInfo->reg, offReg, IRType::S64});

        VReg valReg = emitExpr(*stmt.value, fn);
        if (baseInfo->type.pointeeKind == Type::Float && stmt.value->resolvedType.isInt()) {
            VReg cvtReg = fn.freshReg();
            fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
            valReg = cvtReg;
        }
        IRType stType = (baseInfo->type.pointeeKind == Type::Float ? IRType::F32 : IRType::S32);
        fn.emit(IRInst{IROp::Store, 0, addrReg, valReg, stType});
    }
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
    emitCondition(*stmt.cond, bodyLabel, endLabel, fn);

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
        emitCondition(*stmt.cond, bodyLabel, endLabel, fn);
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

        // Comparison
        VReg r1 = emitExpr(*bin->lhs, fn);
        VReg r2 = emitExpr(*bin->rhs, fn);

        bool isFloat = false;
        if (bin->lhs->resolvedType.isFloat() || bin->rhs->resolvedType.isFloat()) {
            isFloat = true;
            if (bin->lhs->resolvedType.isInt()) {
                VReg cvt = fn.freshReg();
                fn.emit(IRInst{IROp::CvtI2F, cvt, r1, 0, IRType::F32});
                r1 = cvt;
            }
            if (bin->rhs->resolvedType.isInt()) {
                VReg cvt = fn.freshReg();
                fn.emit(IRInst{IROp::CvtI2F, cvt, r2, 0, IRType::F32});
                r2 = cvt;
            }
        }

        IROp setpOp;
        if (isFloat) {
            switch (bin->op) {
                case BinaryOp::Lt: setpOp = IROp::SetpFLt; break;
                case BinaryOp::Gt: setpOp = IROp::SetpFGt; break;
                case BinaryOp::Le: setpOp = IROp::SetpFLe; break;
                case BinaryOp::Ge: setpOp = IROp::SetpFGe; break;
                case BinaryOp::Eq: setpOp = IROp::SetpFEq; break;
                case BinaryOp::Ne: setpOp = IROp::SetpFNe; break;
                default: setpOp = IROp::SetpFEq; break;
            }
        } else {
            switch (bin->op) {
                case BinaryOp::Lt: setpOp = IROp::SetpLt; break;
                case BinaryOp::Gt: setpOp = IROp::SetpGt; break;
                case BinaryOp::Le: setpOp = IROp::SetpLe; break;
                case BinaryOp::Ge: setpOp = IROp::SetpGe; break;
                case BinaryOp::Eq: setpOp = IROp::SetpEq; break;
                case BinaryOp::Ne: setpOp = IROp::SetpNe; break;
                default: setpOp = IROp::SetpEq; break;
            }
        }

        VReg pred = fn.freshReg();
        fn.emit(IRInst{setpOp, pred, r1, r2, IRType::Pred});

        IRInst braCond{IROp::BraCond, 0, pred, 0, IRType::Pred};
        braCond.trueTarget = trueLabel;
        braCond.falseTarget = falseLabel;
        fn.emit(braCond);
        return;
    }

    if (auto* unary = dynamic_cast<const UnaryExpr*>(&cond)) {
        if (unary->op == UnaryOp::Not) {
            emitCondition(*unary->operand, falseLabel, trueLabel, fn);
            return;
        }
    }

    // General scalar expression as condition
    VReg r = emitExpr(cond, fn);
    VReg zero = fn.freshReg();
    IRInst ldimm{IROp::LoadImm, zero, 0, 0, IRType::S32};
    ldimm.immInt = 0;
    fn.emit(ldimm);

    VReg pred = fn.freshReg();
    fn.emit(IRInst{IROp::SetpNe, pred, r, zero, IRType::Pred});

    IRInst braCond{IROp::BraCond, 0, pred, 0, IRType::Pred};
    braCond.trueTarget = trueLabel;
    braCond.falseTarget = falseLabel;
    fn.emit(braCond);
}

VReg IRGen::emitExpr(const Expr& expr, IRFunction& fn) {
    if (auto* intLit = dynamic_cast<const IntLiteralExpr*>(&expr)) {
        VReg r = fn.freshReg();
        IRInst inst{IROp::LoadImm, r, 0, 0, IRType::S32};
        inst.immInt = static_cast<int32_t>(intLit->value);
        fn.emit(inst);
        return r;
    }
    if (auto* floatLit = dynamic_cast<const FloatLiteralExpr*>(&expr)) {
        VReg r = fn.freshReg();
        IRInst inst{IROp::LoadImm, r, 0, 0, IRType::F32};
        inst.immFloat = static_cast<float>(floatLit->value);
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
    return 0;
}

VReg IRGen::emitBuiltin(const BuiltinExpr& expr, IRFunction& fn) {
    VReg r = fn.freshReg();
    IRInst inst{IROp::ReadSReg, r, 0, 0, IRType::S32};
    switch (expr.builtin) {
        case BuiltinKind::ThreadIdx: inst.specialReg = "%tid.x"; break;
        case BuiltinKind::BlockIdx:  inst.specialReg = "%ctaid.x"; break;
        case BuiltinKind::BlockDim:  inst.specialReg = "%ntid.x"; break;
    }
    fn.emit(inst);
    return r;
}

VReg IRGen::emitBinaryExpr(const BinaryExpr& expr, IRFunction& fn) {
    VReg r1 = emitExpr(*expr.lhs, fn);
    VReg r2 = emitExpr(*expr.rhs, fn);

    bool isFloat = (expr.resolvedType.isFloat());
    if (isFloat) {
        if (expr.lhs->resolvedType.isInt()) {
            VReg cvt = fn.freshReg();
            fn.emit(IRInst{IROp::CvtI2F, cvt, r1, 0, IRType::F32});
            r1 = cvt;
        }
        if (expr.rhs->resolvedType.isInt()) {
            VReg cvt = fn.freshReg();
            fn.emit(IRInst{IROp::CvtI2F, cvt, r2, 0, IRType::F32});
            r2 = cvt;
        }
    }

    VReg dst = fn.freshReg();
    IROp op;
    IRType t = isFloat ? IRType::F32 : IRType::S32;

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
            case BinaryOp::Add: op = IROp::Add; break;
            case BinaryOp::Sub: op = IROp::Sub; break;
            case BinaryOp::Mul: op = IROp::Mul; break;
            case BinaryOp::Div: op = IROp::Div; break;
            case BinaryOp::Mod: op = IROp::Mod; break;
            default: op = IROp::Add; break;
        }
    }

    fn.emit(IRInst{op, dst, r1, r2, t});
    return dst;
}

VReg IRGen::emitUnaryExpr(const UnaryExpr& expr, IRFunction& fn) {
    VReg sub = emitExpr(*expr.operand, fn);
    VReg dst = fn.freshReg();
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
    auto* baseIdent = dynamic_cast<const IdentExpr*>(expr.base.get());
    const VarInfo* baseInfo = getVar(baseIdent->name);

    VReg idxReg = emitExpr(*expr.index, fn);
    VReg szReg = fn.freshReg();
    IRInst ldimm{IROp::LoadImm, szReg, 0, 0, IRType::S32};
    ldimm.immInt = 4;
    fn.emit(ldimm);

    VReg offReg = fn.freshReg();
    fn.emit(IRInst{IROp::MulWide, offReg, idxReg, szReg, IRType::S64});

    VReg addrReg = fn.freshReg();
    fn.emit(IRInst{IROp::AddS64, addrReg, baseInfo->reg, offReg, IRType::S64});

    VReg dstReg = fn.freshReg();
    IRType loadType = (baseInfo->type.pointeeKind == Type::Float ? IRType::F32 : IRType::S32);
    fn.emit(IRInst{IROp::Load, dstReg, addrReg, 0, loadType});
    return dstReg;
}

VReg IRGen::emitCastExpr(const CastExpr& expr, IRFunction& fn) {
    VReg sub = emitExpr(*expr.operand, fn);
    VReg dst = fn.freshReg();
    if (expr.targetType.isFloat() && expr.operand->resolvedType.isInt()) {
        fn.emit(IRInst{IROp::CvtI2F, dst, sub, 0, IRType::F32});
    } else if (expr.targetType.isInt() && expr.operand->resolvedType.isFloat()) {
        fn.emit(IRInst{IROp::CvtF2I, dst, sub, 0, IRType::S32});
    } else {
        IRType t = expr.targetType.isFloat() ? IRType::F32 : IRType::S32;
        fn.emit(IRInst{IROp::Mov, dst, sub, 0, t});
    }
    return dst;
}

} // namespace kernc
