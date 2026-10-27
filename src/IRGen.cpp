#include "kernc/IRGen.h"

namespace kernc {

// Maps a source-language type to the IR register type that holds its values.
static IRType toIRType(const Type& t) {
    switch (t.kind) {
        case Type::Float:
        case Type::Float2:
        case Type::Float4: return IRType::F32;
        case Type::Ptr:   return IRType::S64;
        case Type::Bool:  return IRType::Pred;
        case Type::Int:
        case Type::Int2:
        case Type::Int4:
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
    if (stmt.type.isVector()) {
        int width = stmt.type.vectorWidth();
        IRType elemTy = stmt.type.isFloatVector() ? IRType::F32 : IRType::S32;
        std::vector<VReg> regs;
        for (int i = 0; i < width; ++i) {
            regs.push_back(fn.freshReg(elemTy));
        }
        setVar(stmt.name, 0, stmt.type, "", regs);

        if (stmt.init) {
            std::vector<VReg> initRegs = emitVectorExpr(*stmt.init, fn);
            for (int i = 0; i < width && i < static_cast<int>(initRegs.size()); ++i) {
                fn.emit(IRInst{IROp::Mov, regs[i], initRegs[i], 0, elemTy});
            }
        }
        return;
    }

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
        if (vInfo->type.isVector()) {
            std::vector<VReg> valRegs = emitVectorExpr(*stmt.value, fn);
            IRType elemTy = vInfo->type.isFloatVector() ? IRType::F32 : IRType::S32;
            for (size_t i = 0; i < vInfo->regs.size() && i < valRegs.size(); ++i) {
                fn.emit(IRInst{IROp::Mov, vInfo->regs[i], valRegs[i], 0, elemTy});
            }
            return;
        }
        VReg valReg = emitExpr(*stmt.value, fn);
        if (vInfo->type.isFloat() && stmt.value->resolvedType.isInt()) {
            VReg cvtReg = fn.freshReg(IRType::F32);
            fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
            valReg = cvtReg;
        }
        fn.emit(IRInst{IROp::Mov, vInfo->reg, valReg, 0, toIRType(vInfo->type)});
    } else if (auto* mem = dynamic_cast<const MemberExpr*>(stmt.target.get())) {
        int compIdx = 0;
        if (mem->member == "x") compIdx = 0;
        else if (mem->member == "y") compIdx = 1;
        else if (mem->member == "z") compIdx = 2;
        else if (mem->member == "w") compIdx = 3;

        if (auto* baseIdent = dynamic_cast<const IdentExpr*>(mem->base.get())) {
            const VarInfo* vInfo = getVar(baseIdent->name);
            if (vInfo && vInfo->type.isVector()) {
                VReg valReg = emitExpr(*stmt.value, fn);
                IRType elemTy = vInfo->type.isFloatVector() ? IRType::F32 : IRType::S32;
                if (elemTy == IRType::F32 && stmt.value->resolvedType.isInt()) {
                    VReg cvtReg = fn.freshReg(IRType::F32);
                    fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
                    valReg = cvtReg;
                }
                fn.emit(IRInst{IROp::Mov, vInfo->regs[compIdx], valReg, 0, elemTy});
                return;
            }
        } else if (auto* baseIdx = dynamic_cast<const IndexExpr*>(mem->base.get())) {
            AddrSpace space;
            IRType elemType;
            VReg baseAddr = emitElementAddress(*baseIdx, fn, space, elemType);
            VReg finalAddr = baseAddr;
            if (compIdx > 0) {
                if (space == AddrSpace::Global) {
                    VReg offReg = fn.freshReg(IRType::S64);
                    VReg cReg = fn.freshReg(IRType::S32);
                    IRInst ldC{IROp::LoadImm, cReg, 0, 0, IRType::S32};
                    ldC.immInt = compIdx * 4;
                    fn.emit(ldC);
                    VReg oneReg = fn.freshReg(IRType::S32);
                    IRInst ldOne{IROp::LoadImm, oneReg, 0, 0, IRType::S32};
                    ldOne.immInt = 1;
                    fn.emit(ldOne);
                    fn.emit(IRInst{IROp::MulWide, offReg, cReg, oneReg, IRType::S64});
                    finalAddr = fn.freshReg(IRType::S64);
                    fn.emit(IRInst{IROp::AddS64, finalAddr, baseAddr, offReg, IRType::S64});
                } else {
                    VReg offReg = fn.freshReg(IRType::S32);
                    IRInst ldC{IROp::LoadImm, offReg, 0, 0, IRType::S32};
                    ldC.immInt = compIdx * 4;
                    fn.emit(ldC);
                    finalAddr = fn.freshReg(IRType::S32);
                    fn.emit(IRInst{IROp::Add, finalAddr, baseAddr, offReg, IRType::S32});
                }
            }
            VReg valReg = emitExpr(*stmt.value, fn);
            if (elemType == IRType::F32 && stmt.value->resolvedType.isInt()) {
                VReg cvtReg = fn.freshReg(IRType::F32);
                fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
                valReg = cvtReg;
            }
            IRInst store{IROp::Store, 0, finalAddr, valReg, elemType};
            store.space = space;
            fn.emit(store);
            return;
        }
    } else if (auto* idx = dynamic_cast<const IndexExpr*>(stmt.target.get())) {
        AddrSpace space;
        IRType elemType;
        VReg addrReg = emitElementAddress(*idx, fn, space, elemType);

        if (stmt.target->resolvedType.isVector()) {
            std::vector<VReg> valRegs = emitVectorExpr(*stmt.value, fn);
            int width = stmt.target->resolvedType.vectorWidth();
            if (width == 2) {
                IRInst store{IROp::StoreV2, 0, addrReg, 0, elemType};
                store.space = space;
                store.args = valRegs;
                fn.emit(store);
            } else {
                IRInst store{IROp::StoreV4, 0, addrReg, 0, elemType};
                store.space = space;
                store.args = valRegs;
                fn.emit(store);
            }
            return;
        }

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
    std::string symbol = ptxname::shared(fn.name, stmt.name);
    auto taken = [&](const std::string& s) {
        for (const auto& sv : fn.sharedVars)
            if (sv.symbol == s) return true;
        return false;
    };
    for (int n = 1; taken(symbol); ++n) {
        symbol = ptxname::shared(fn.name, stmt.name) + "$" + std::to_string(n);
    }
    int width = stmt.elemType.vectorWidth();
    // 16-byte alignment for every shared array (at most 12 bytes of padding), so the v4
    // intrinsics can also be used on scalar arrays like `shared int s[N]`.
    int align = 16;
    int64_t count = stmt.size * width;
    fn.sharedVars.push_back({symbol, toIRType(stmt.elemType), count, align});
    setVar(stmt.name, 0, Type::makeSharedArray(stmt.elemType.kind), symbol);
}

VReg IRGen::emitElementAddress(const IndexExpr& idx, IRFunction& fn, AddrSpace& space, IRType& elemType) {
    return emitMemoryAddress(*idx.base, idx.index.get(), fn, space, elemType);
}

VReg IRGen::emitMemoryAddress(const Expr& baseExpr, const Expr* indexExpr, IRFunction& fn, AddrSpace& space, IRType& elemType) {
    auto* baseIdent = dynamic_cast<const IdentExpr*>(&baseExpr);
    const VarInfo* baseInfo = getVar(baseIdent->name);
    bool isFloat = (baseInfo->type.pointeeKind == Type::Float ||
                    baseInfo->type.pointeeKind == Type::Float2 ||
                    baseInfo->type.pointeeKind == Type::Float4);
    elemType = isFloat ? IRType::F32 : IRType::S32;

    int scaleBytes = 4;
    Type pointeeType(baseInfo->type.pointeeKind);
    if (pointeeType.sizeInBytes() > 0) {
        scaleBytes = pointeeType.sizeInBytes();
    }

    if (baseInfo->type.isSharedArray()) {
        // 32-bit arithmetic: shared addresses are offsets into the block's shared window.
        space = AddrSpace::Shared;
        VReg baseReg = fn.freshReg(IRType::S32);
        IRInst addrOf{IROp::SharedAddr, baseReg, 0, 0, IRType::S32};
        addrOf.symbol = baseInfo->sharedSymbol;
        fn.emit(addrOf);

        if (indexExpr) {
            VReg idxReg = emitExpr(*indexExpr, fn);
            VReg szReg = fn.freshReg(IRType::S32);
            IRInst ldimm{IROp::LoadImm, szReg, 0, 0, IRType::S32};
            ldimm.immInt = scaleBytes;
            fn.emit(ldimm);

            VReg offReg = fn.freshReg(IRType::S32);
            fn.emit(IRInst{IROp::Mul, offReg, idxReg, szReg, IRType::S32});
            VReg addrReg = fn.freshReg(IRType::S32);
            fn.emit(IRInst{IROp::Add, addrReg, baseReg, offReg, IRType::S32});
            return addrReg;
        }
        return baseReg;
    }

    // Global pointer: widen the 32-bit index while scaling it, then add the 64-bit base.
    space = AddrSpace::Global;
    if (indexExpr) {
        VReg idxReg = emitExpr(*indexExpr, fn);
        VReg szReg = fn.freshReg(IRType::S32);
        IRInst ldimm{IROp::LoadImm, szReg, 0, 0, IRType::S32};
        ldimm.immInt = scaleBytes;
        fn.emit(ldimm);

        VReg offReg = fn.freshReg(IRType::S64);
        fn.emit(IRInst{IROp::MulWide, offReg, idxReg, szReg, IRType::S64});
        VReg addrReg = fn.freshReg(IRType::S64);
        fn.emit(IRInst{IROp::AddS64, addrReg, baseInfo->reg, offReg, IRType::S64});
        return addrReg;
    }
    return baseInfo->reg;
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

bool IRGen::extractIntConstant(const Expr* expr, int64_t& outVal) {
    if (!expr) return false;
    if (auto* lit = dynamic_cast<const IntLiteralExpr*>(expr)) {
        outVal = lit->value;
        return true;
    }
    if (auto* un = dynamic_cast<const UnaryExpr*>(expr)) {
        if (un->op == UnaryOp::Neg) {
            int64_t sub = 0;
            if (extractIntConstant(un->operand.get(), sub)) {
                outVal = -sub;
                return true;
            }
        }
    }
    return false;
}

bool IRGen::hasAssignmentTo(const Stmt& stmt, const std::string& varName) {
    if (auto* assign = dynamic_cast<const AssignStmt*>(&stmt)) {
        if (auto* id = dynamic_cast<const IdentExpr*>(assign->target.get())) {
            if (id->name == varName) return true;
        }
        return false;
    }
    if (auto* varDecl = dynamic_cast<const VarDeclStmt*>(&stmt)) {
        if (varDecl->name == varName) return true;
        return false;
    }
    if (auto* block = dynamic_cast<const BlockStmt*>(&stmt)) {
        for (const auto& s : block->stmts) {
            if (hasAssignmentTo(*s, varName)) return true;
        }
        return false;
    }
    if (auto* ifStmt = dynamic_cast<const IfStmt*>(&stmt)) {
        if (ifStmt->thenBranch && hasAssignmentTo(*ifStmt->thenBranch, varName)) return true;
        if (ifStmt->elseBranch && hasAssignmentTo(*ifStmt->elseBranch, varName)) return true;
        return false;
    }
    if (auto* whileStmt = dynamic_cast<const WhileStmt*>(&stmt)) {
        if (whileStmt->body && hasAssignmentTo(*whileStmt->body, varName)) return true;
        return false;
    }
    if (auto* forStmt = dynamic_cast<const ForStmt*>(&stmt)) {
        if (forStmt->init && hasAssignmentTo(*forStmt->init, varName)) return true;
        if (forStmt->step && hasAssignmentTo(*forStmt->step, varName)) return true;
        if (forStmt->body && hasAssignmentTo(*forStmt->body, varName)) return true;
        return false;
    }
    return false;
}

bool IRGen::tryGetConstantTripCount(const ForStmt& stmt, int64_t& tripCount, int64_t& start, int64_t& step, std::string& varName) {
    if (!stmt.init || !stmt.cond || !stmt.step || !stmt.body) {
        return false;
    }

    // 1. Init: int i = start; or i = start;
    if (auto* decl = dynamic_cast<const VarDeclStmt*>(stmt.init.get())) {
        if (!decl->type.isInt() || !decl->init) return false;
        varName = decl->name;
        if (!extractIntConstant(decl->init.get(), start)) return false;
    } else if (auto* assign = dynamic_cast<const AssignStmt*>(stmt.init.get())) {
        auto* id = dynamic_cast<const IdentExpr*>(assign->target.get());
        if (!id || !assign->value) return false;
        varName = id->name;
        if (!extractIntConstant(assign->value.get(), start)) return false;
    } else {
        return false;
    }

    // 2. Cond: i < C_end, i <= C_end, etc.
    auto* condBin = dynamic_cast<const BinaryExpr*>(stmt.cond.get());
    if (!condBin) return false;

    int64_t condConst = 0;
    BinaryOp op = condBin->op;
    bool varOnLhs = false;

    if (auto* id = dynamic_cast<const IdentExpr*>(condBin->lhs.get())) {
        if (id->name == varName && extractIntConstant(condBin->rhs.get(), condConst)) {
            varOnLhs = true;
        }
    }
    if (!varOnLhs) {
        if (auto* id = dynamic_cast<const IdentExpr*>(condBin->rhs.get())) {
            if (id->name == varName && extractIntConstant(condBin->lhs.get(), condConst)) {
                switch (op) {
                    case BinaryOp::Lt: op = BinaryOp::Gt; break;
                    case BinaryOp::Gt: op = BinaryOp::Lt; break;
                    case BinaryOp::Le: op = BinaryOp::Ge; break;
                    case BinaryOp::Ge: op = BinaryOp::Le; break;
                    case BinaryOp::Eq: break;
                    case BinaryOp::Ne: break;
                    default: return false;
                }
                varOnLhs = true;
            }
        }
    }
    if (!varOnLhs) return false;

    // 3. Step: i = i + step or i = i - step
    const AssignStmt* stepAssign = dynamic_cast<const AssignStmt*>(stmt.step.get());
    if (!stepAssign) return false;
    auto* stepTarget = dynamic_cast<const IdentExpr*>(stepAssign->target.get());
    if (!stepTarget || stepTarget->name != varName) return false;
    auto* stepBin = dynamic_cast<const BinaryExpr*>(stepAssign->value.get());
    if (!stepBin) return false;

    int64_t stepVal = 0;
    step = 0;
    if (stepBin->op == BinaryOp::Add) {
        if (auto* id = dynamic_cast<const IdentExpr*>(stepBin->lhs.get())) {
            if (id->name == varName && extractIntConstant(stepBin->rhs.get(), stepVal)) {
                step = stepVal;
            }
        }
        if (step == 0) {
            if (auto* id = dynamic_cast<const IdentExpr*>(stepBin->rhs.get())) {
                if (id->name == varName && extractIntConstant(stepBin->lhs.get(), stepVal)) {
                    step = stepVal;
                }
            }
        }
    } else if (stepBin->op == BinaryOp::Sub) {
        if (auto* id = dynamic_cast<const IdentExpr*>(stepBin->lhs.get())) {
            if (id->name == varName && extractIntConstant(stepBin->rhs.get(), stepVal)) {
                step = -stepVal;
            }
        }
    }
    if (step == 0) return false;

    // 4. Verify loop body does not mutate varName
    if (hasAssignmentTo(*stmt.body, varName)) return false;

    // 5. Trip count
    if (step > 0) {
        if (op == BinaryOp::Lt) {
            if (start >= condConst) tripCount = 0;
            else tripCount = (condConst - start + step - 1) / step;
        } else if (op == BinaryOp::Le) {
            if (start > condConst) tripCount = 0;
            else tripCount = (condConst - start) / step + 1;
        } else if (op == BinaryOp::Ne) {
            if (condConst < start || ((condConst - start) % step != 0)) return false;
            tripCount = (condConst - start) / step;
        } else {
            return false;
        }
    } else {
        int64_t posStep = -step;
        if (op == BinaryOp::Gt) {
            if (start <= condConst) tripCount = 0;
            else tripCount = (start - condConst + posStep - 1) / posStep;
        } else if (op == BinaryOp::Ge) {
            if (start < condConst) tripCount = 0;
            else tripCount = (start - condConst) / posStep + 1;
        } else if (op == BinaryOp::Ne) {
            if (start < condConst || ((start - condConst) % posStep != 0)) return false;
            tripCount = (start - condConst) / posStep;
        } else {
            return false;
        }
    }

    return (tripCount >= 0);
}

void IRGen::emitFor(const ForStmt& stmt, IRFunction& fn) {
    int64_t tripCount = 0;
    int64_t start = 0;
    int64_t step = 0;
    std::string varName;
    if (unrollLoops_ && tryGetConstantTripCount(stmt, tripCount, start, step, varName)) {
        if (tripCount > 0 && tripCount <= 16) {
            enterScope();
            bool isDecl = dynamic_cast<const VarDeclStmt*>(stmt.init.get()) != nullptr;
            for (int64_t k = 0; k < tripCount; ++k) {
                enterScope();
                int64_t curVal = start + k * step;
                VReg iterReg = fn.freshReg(IRType::S32);
                IRInst ldimm{IROp::LoadImm, iterReg, 0, 0, IRType::S32};
                ldimm.immInt = static_cast<int32_t>(curVal);
                fn.emit(ldimm);
                setVar(varName, iterReg, Type::makeInt());
                emitStmt(*stmt.body, fn);
                exitScope();
            }
            if (!isDecl) {
                const VarInfo* vInfo = getVar(varName);
                if (vInfo) {
                    VReg finalReg = fn.freshReg(IRType::S32);
                    IRInst ldimm{IROp::LoadImm, finalReg, 0, 0, IRType::S32};
                    ldimm.immInt = static_cast<int32_t>(start + tripCount * step);
                    fn.emit(ldimm);
                    fn.emit(IRInst{IROp::Mov, vInfo->reg, finalReg, 0, IRType::S32});
                }
            }
            exitScope();
            return;
        }
    }

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
    if (auto* mem = dynamic_cast<const MemberExpr*>(&expr)) {
        return emitMemberExpr(*mem, fn);
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

VReg IRGen::emitMemberExpr(const MemberExpr& expr, IRFunction& fn) {
    int compIdx = 0;
    if (expr.member == "x") compIdx = 0;
    else if (expr.member == "y") compIdx = 1;
    else if (expr.member == "z") compIdx = 2;
    else if (expr.member == "w") compIdx = 3;

    if (auto* baseIdent = dynamic_cast<const IdentExpr*>(expr.base.get())) {
        const VarInfo* vInfo = getVar(baseIdent->name);
        if (vInfo && vInfo->type.isVector()) {
            return vInfo->regs[compIdx];
        }
    } else if (auto* baseIdx = dynamic_cast<const IndexExpr*>(expr.base.get())) {
        AddrSpace space;
        IRType elemType;
        VReg baseAddr = emitElementAddress(*baseIdx, fn, space, elemType);
        VReg finalAddr = baseAddr;
        if (compIdx > 0) {
            if (space == AddrSpace::Global) {
                VReg offReg = fn.freshReg(IRType::S64);
                VReg cReg = fn.freshReg(IRType::S32);
                IRInst ldC{IROp::LoadImm, cReg, 0, 0, IRType::S32};
                ldC.immInt = compIdx * 4;
                fn.emit(ldC);
                VReg oneReg = fn.freshReg(IRType::S32);
                IRInst ldOne{IROp::LoadImm, oneReg, 0, 0, IRType::S32};
                ldOne.immInt = 1;
                fn.emit(ldOne);
                fn.emit(IRInst{IROp::MulWide, offReg, cReg, oneReg, IRType::S64});
                finalAddr = fn.freshReg(IRType::S64);
                fn.emit(IRInst{IROp::AddS64, finalAddr, baseAddr, offReg, IRType::S64});
            } else {
                VReg offReg = fn.freshReg(IRType::S32);
                IRInst ldC{IROp::LoadImm, offReg, 0, 0, IRType::S32};
                ldC.immInt = compIdx * 4;
                fn.emit(ldC);
                finalAddr = fn.freshReg(IRType::S32);
                fn.emit(IRInst{IROp::Add, finalAddr, baseAddr, offReg, IRType::S32});
            }
        }
        VReg dstReg = fn.freshReg(elemType);
        IRInst load{IROp::Load, dstReg, finalAddr, 0, elemType};
        load.space = space;
        fn.emit(load);
        return dstReg;
    }

    std::vector<VReg> vRegs = emitVectorExpr(*expr.base, fn);
    if (compIdx < static_cast<int>(vRegs.size())) {
        return vRegs[compIdx];
    }
    return 0;
}

std::vector<VReg> IRGen::emitVectorExpr(const Expr& expr, IRFunction& fn) {
    if (auto* ident = dynamic_cast<const IdentExpr*>(&expr)) {
        const VarInfo* vInfo = getVar(ident->name);
        if (vInfo && vInfo->type.isVector()) {
            return vInfo->regs;
        }
    }
    if (auto* idx = dynamic_cast<const IndexExpr*>(&expr)) {
        AddrSpace space;
        IRType elemType;
        VReg addrReg = emitElementAddress(*idx, fn, space, elemType);
        int width = expr.resolvedType.vectorWidth();
        std::vector<VReg> dsts;
        for (int i = 0; i < width; ++i) dsts.push_back(fn.freshReg(elemType));
        if (width == 2) {
            IRInst ld{IROp::LoadV2, 0, addrReg, 0, elemType};
            ld.space = space;
            ld.dsts = dsts;
            fn.emit(ld);
        } else {
            IRInst ld{IROp::LoadV4, 0, addrReg, 0, elemType};
            ld.space = space;
            ld.dsts = dsts;
            fn.emit(ld);
        }
        return dsts;
    }
    if (auto* call = dynamic_cast<const CallExpr*>(&expr)) {
        if (isVectorConstructor(call->callee)) {
            bool isFloat = expr.resolvedType.isFloatVector();
            std::vector<VReg> res;
            for (size_t i = 0; i < call->args.size(); ++i) {
                VReg a = emitExpr(*call->args[i], fn);
                if (isFloat && call->args[i]->resolvedType.isInt()) {
                    VReg cvt = fn.freshReg(IRType::F32);
                    fn.emit(IRInst{IROp::CvtI2F, cvt, a, 0, IRType::F32});
                    a = cvt;
                }
                res.push_back(a);
            }
            return res;
        }
        if (auto vecIntrin = getVectorIntrinsic(call->callee)) {
            bool is2 = (*vecIntrin == VectorIntrinsic::LoadInt2 ||
                        *vecIntrin == VectorIntrinsic::LoadFloat2);
            AddrSpace space;
            IRType elemType;
            VReg addrReg = emitVectorIntrinsicAddress(*call->args[0], *call->args[1], is2 ? 8 : 16,
                                                      fn, space, elemType);
            int width = is2 ? 2 : 4;
            std::vector<VReg> dsts;
            for (int i = 0; i < width; ++i) dsts.push_back(fn.freshReg(elemType));
            if (is2) {
                IRInst ld{IROp::LoadV2, 0, addrReg, 0, elemType};
                ld.space = space;
                ld.dsts = dsts;
                fn.emit(ld);
            } else {
                IRInst ld{IROp::LoadV4, 0, addrReg, 0, elemType};
                ld.space = space;
                ld.dsts = dsts;
                fn.emit(ld);
            }
            return dsts;
        }
    }
    return {};
}

// The offset counts whole vectors of the access width (8 bytes for v2, 16 for v4), as in
// reinterpret_cast<int4*>(p)[i]. Every access is then naturally aligned whatever the buffer's
// element type; counting scalars allowed a 16-byte ld.v4 at a 4-byte boundary, which faults
// with CUDA_ERROR_MISALIGNED_ADDRESS.
VReg IRGen::emitVectorIntrinsicAddress(const Expr& baseExpr, const Expr& indexExpr, int vectorBytes,
                                       IRFunction& fn, AddrSpace& space, IRType& elemType) {
    auto* baseIdent = dynamic_cast<const IdentExpr*>(&baseExpr);
    const VarInfo* baseInfo = getVar(baseIdent->name);
    bool isFloat = (baseInfo->type.pointeeKind == Type::Float ||
                    baseInfo->type.pointeeKind == Type::Float2 ||
                    baseInfo->type.pointeeKind == Type::Float4);
    elemType = isFloat ? IRType::F32 : IRType::S32;

    int scaleBytes = vectorBytes;

    if (baseInfo->type.isSharedArray()) {
        space = AddrSpace::Shared;
        VReg baseReg = fn.freshReg(IRType::S32);
        IRInst addrOf{IROp::SharedAddr, baseReg, 0, 0, IRType::S32};
        addrOf.symbol = baseInfo->sharedSymbol;
        fn.emit(addrOf);

        VReg idxReg = emitExpr(indexExpr, fn);
        VReg szReg = fn.freshReg(IRType::S32);
        IRInst ldimm{IROp::LoadImm, szReg, 0, 0, IRType::S32};
        ldimm.immInt = scaleBytes;
        fn.emit(ldimm);

        VReg offReg = fn.freshReg(IRType::S32);
        fn.emit(IRInst{IROp::Mul, offReg, idxReg, szReg, IRType::S32});
        VReg addrReg = fn.freshReg(IRType::S32);
        fn.emit(IRInst{IROp::Add, addrReg, baseReg, offReg, IRType::S32});
        return addrReg;
    }

    space = AddrSpace::Global;
    VReg idxReg = emitExpr(indexExpr, fn);
    VReg szReg = fn.freshReg(IRType::S32);
    IRInst ldimm{IROp::LoadImm, szReg, 0, 0, IRType::S32};
    ldimm.immInt = scaleBytes;
    fn.emit(ldimm);

    VReg offReg = fn.freshReg(IRType::S64);
    fn.emit(IRInst{IROp::MulWide, offReg, idxReg, szReg, IRType::S64});
    VReg addrReg = fn.freshReg(IRType::S64);
    fn.emit(IRInst{IROp::AddS64, addrReg, baseInfo->reg, offReg, IRType::S64});
    return addrReg;
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
    if (auto intrin = getWarpIntrinsic(expr.callee)) {
        switch (*intrin) {
            case WarpIntrinsic::Activemask: {
                VReg dst = fn.freshReg(IRType::S32);
                fn.emit(IRInst{IROp::Activemask, dst, 0, 0, IRType::S32});
                return dst;
            }
            case WarpIntrinsic::BallotSync: {
                VReg mask = emitExpr(*expr.args[0], fn);
                VReg pred = emitAsPred(*expr.args[1], fn);
                VReg dst = fn.freshReg(IRType::S32);
                fn.emit(IRInst{IROp::VoteBallot, dst, pred, mask, IRType::S32});
                return dst;
            }
            case WarpIntrinsic::ShflSync:
            case WarpIntrinsic::ShflDownSync:
            case WarpIntrinsic::ShflUpSync:
            case WarpIntrinsic::ShflXorSync: {
                VReg mask = emitExpr(*expr.args[0], fn);
                VReg val = emitExpr(*expr.args[1], fn);
                VReg b = emitExpr(*expr.args[2], fn);
                IRType valIRType = expr.args[1]->resolvedType.isFloat() ? IRType::F32 : IRType::S32;
                VReg dst = fn.freshReg(valIRType);

                IROp op = IROp::ShflIdx;
                int32_t clamp = 31;
                if (*intrin == WarpIntrinsic::ShflDownSync) {
                    op = IROp::ShflDown;
                    clamp = 31;
                } else if (*intrin == WarpIntrinsic::ShflUpSync) {
                    op = IROp::ShflUp;
                    clamp = 0;
                } else if (*intrin == WarpIntrinsic::ShflXorSync) {
                    op = IROp::ShflBfly;
                    clamp = 31;
                } else if (*intrin == WarpIntrinsic::ShflSync) {
                    op = IROp::ShflIdx;
                    clamp = 31;
                }
                IRInst inst{op, dst, val, b, mask, valIRType};
                inst.immInt = clamp;
                fn.emit(inst);
                return dst;
            }
        }
    }

    if (auto atomic = getAtomicIntrinsic(expr.callee)) {
        bool isCas = (*atomic == AtomicIntrinsic::Cas);
        const Expr* baseExpr = nullptr;
        const Expr* indexExpr = nullptr;
        const Expr* cmpExpr = nullptr;
        const Expr* valExpr = nullptr;

        if (auto* idx = dynamic_cast<const IndexExpr*>(expr.args[0].get())) {
            baseExpr = idx->base.get();
            indexExpr = idx->index.get();
            if (isCas) {
                cmpExpr = expr.args[1].get();
                valExpr = expr.args[2].get();
            } else {
                valExpr = expr.args[1].get();
            }
        } else {
            baseExpr = expr.args[0].get();
            size_t maxArgs = isCas ? 4 : 3;
            if (expr.args.size() == maxArgs) {
                indexExpr = expr.args[1].get();
                if (isCas) {
                    cmpExpr = expr.args[2].get();
                    valExpr = expr.args[3].get();
                } else {
                    valExpr = expr.args[2].get();
                }
            } else {
                indexExpr = nullptr;
                if (isCas) {
                    cmpExpr = expr.args[1].get();
                    valExpr = expr.args[2].get();
                } else {
                    valExpr = expr.args[1].get();
                }
            }
        }

        AddrSpace space;
        IRType elemType;
        VReg addrReg = emitMemoryAddress(*baseExpr, indexExpr, fn, space, elemType);

        if (isCas) {
            VReg cmpReg = emitExpr(*cmpExpr, fn);
            VReg valReg = emitExpr(*valExpr, fn);
            VReg dstReg = fn.freshReg(IRType::S32);
            IRInst inst{IROp::AtomicCas, dstReg, addrReg, cmpReg, valReg, IRType::S32};
            inst.space = space;
            fn.emit(inst);
            return dstReg;
        }

        VReg valReg = emitExpr(*valExpr, fn);
        if (elemType == IRType::F32 && valExpr->resolvedType.isInt()) {
            VReg cvtReg = fn.freshReg(IRType::F32);
            fn.emit(IRInst{IROp::CvtI2F, cvtReg, valReg, 0, IRType::F32});
            valReg = cvtReg;
        }

        IROp op;
        switch (*atomic) {
            case AtomicIntrinsic::Add:  op = IROp::AtomicAdd; break;
            case AtomicIntrinsic::Sub:  op = IROp::AtomicSub; break;
            case AtomicIntrinsic::Min:  op = IROp::AtomicMin; break;
            case AtomicIntrinsic::Max:  op = IROp::AtomicMax; break;
            case AtomicIntrinsic::And:  op = IROp::AtomicAnd; break;
            case AtomicIntrinsic::Or:   op = IROp::AtomicOr; break;
            case AtomicIntrinsic::Xor:  op = IROp::AtomicXor; break;
            case AtomicIntrinsic::Exch: op = IROp::AtomicExch; break;
            default:                    op = IROp::AtomicAdd; break;
        }

        VReg dstReg = fn.freshReg(elemType);
        IRInst inst{op, dstReg, addrReg, valReg, elemType};
        inst.space = space;
        fn.emit(inst);
        return dstReg;
    }

    if (auto vecIntrin = getVectorIntrinsic(expr.callee)) {
        bool isStore = (*vecIntrin == VectorIntrinsic::StoreInt2 ||
                        *vecIntrin == VectorIntrinsic::StoreInt4 ||
                        *vecIntrin == VectorIntrinsic::StoreFloat2 ||
                        *vecIntrin == VectorIntrinsic::StoreFloat4);
        bool is2 = (*vecIntrin == VectorIntrinsic::LoadInt2 ||
                    *vecIntrin == VectorIntrinsic::StoreInt2 ||
                    *vecIntrin == VectorIntrinsic::LoadFloat2 ||
                    *vecIntrin == VectorIntrinsic::StoreFloat2);
        AddrSpace space;
        IRType elemType;
        VReg addrReg = emitVectorIntrinsicAddress(*expr.args[0], *expr.args[1], is2 ? 8 : 16,
                                                  fn, space, elemType);

        if (isStore) {
            std::vector<VReg> valRegs = emitVectorExpr(*expr.args[2], fn);
            if (is2) {
                IRInst st{IROp::StoreV2, 0, addrReg, 0, elemType};
                st.space = space;
                st.args = valRegs;
                fn.emit(st);
            } else {
                IRInst st{IROp::StoreV4, 0, addrReg, 0, elemType};
                st.space = space;
                st.args = valRegs;
                fn.emit(st);
            }
            return 0;
        } else {
            int width = is2 ? 2 : 4;
            std::vector<VReg> dsts;
            for (int i = 0; i < width; ++i) dsts.push_back(fn.freshReg(elemType));
            if (is2) {
                IRInst ld{IROp::LoadV2, 0, addrReg, 0, elemType};
                ld.space = space;
                ld.dsts = dsts;
                fn.emit(ld);
            } else {
                IRInst ld{IROp::LoadV4, 0, addrReg, 0, elemType};
                ld.space = space;
                ld.dsts = dsts;
                fn.emit(ld);
            }
            return dsts[0];
        }
    }

    if (isVectorConstructor(expr.callee)) {
        std::vector<VReg> regs = emitVectorExpr(expr, fn);
        return regs.empty() ? 0 : regs[0];
    }

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
