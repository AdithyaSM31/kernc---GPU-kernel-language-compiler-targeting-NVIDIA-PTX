#pragma once

#include "kernc/AST.h"
#include "kernc/IR.h"
#include "kernc/Diagnostics.h"
#include <unordered_map>
#include <string>

namespace kernc {

class IRGen {
public:
    explicit IRGen(DiagnosticsEngine& diag);

    IRModule emitModule(const Program& program);
    IRFunction emitFunction(const FunctionDecl& kernel);

private:
    void emitStmt(const Stmt& stmt, IRFunction& fn);
    void emitStmtBody(const Stmt& stmt, IRFunction& fn);
    void emitVarDecl(const VarDeclStmt& stmt, IRFunction& fn);
    void emitAssign(const AssignStmt& stmt, IRFunction& fn);
    void emitIf(const IfStmt& stmt, IRFunction& fn);
    void emitWhile(const WhileStmt& stmt, IRFunction& fn);
    void emitFor(const ForStmt& stmt, IRFunction& fn);
    void emitBlock(const BlockStmt& stmt, IRFunction& fn);

    VReg emitExpr(const Expr& expr, IRFunction& fn);
    VReg emitBinaryExpr(const BinaryExpr& expr, IRFunction& fn);
    VReg emitUnaryExpr(const UnaryExpr& expr, IRFunction& fn);
    VReg emitIndexExpr(const IndexExpr& expr, IRFunction& fn);
    VReg emitMemberExpr(const MemberExpr& expr, IRFunction& fn);
    std::vector<VReg> emitVectorExpr(const Expr& expr, IRFunction& fn);
    VReg emitCastExpr(const CastExpr& expr, IRFunction& fn);
    VReg emitBuiltin(const BuiltinExpr& expr, IRFunction& fn);
    VReg emitCallExpr(const CallExpr& expr, IRFunction& fn);

    void emitCondition(const Expr& cond, const Label& trueLabel, const Label& falseLabel, IRFunction& fn);
    void emitBranch(VReg pred, const Label& trueLabel, const Label& falseLabel, IRFunction& fn);
    VReg emitCompare(const BinaryExpr& bin, IRFunction& fn);
    VReg emitAsPred(const Expr& expr, IRFunction& fn);
    VReg emitShortCircuitValue(const BinaryExpr& expr, IRFunction& fn);
    static bool isComparison(BinaryOp op);

    void emitSharedDecl(const SharedDeclStmt& stmt, IRFunction& fn);
    // Address of base[index]; sets `space` and `elemType` for the Load/Store that uses it.
    VReg emitElementAddress(const IndexExpr& idx, IRFunction& fn, AddrSpace& space, IRType& elemType);
    VReg emitMemoryAddress(const Expr& baseExpr, const Expr* indexExpr, IRFunction& fn, AddrSpace& space, IRType& elemType);
    VReg emitVectorIntrinsicAddress(const Expr& baseExpr, const Expr& indexExpr, IRFunction& fn, AddrSpace& space, IRType& elemType);

    struct VarInfo {
        VReg reg = 0;             // 0 for shared arrays and vector variables
        std::vector<VReg> regs;   // component registers for vector variables
        Type type;
        std::string sharedSymbol; // module-scope .shared symbol, for shared arrays
    };

    void enterScope() {
        varScopes_.emplace_back();
    }

    void exitScope() {
        if (!varScopes_.empty()) {
            varScopes_.pop_back();
        }
    }

    void setVar(const std::string& name, VReg reg, Type type, std::string sharedSymbol = "", std::vector<VReg> regs = {}) {
        if (!varScopes_.empty()) {
            varScopes_.back()[name] = {reg, std::move(regs), type, std::move(sharedSymbol)};
        }
    }

    const VarInfo* getVar(const std::string& name) const {
        for (auto it = varScopes_.rbegin(); it != varScopes_.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) {
                return &found->second;
            }
        }
        return nullptr;
    }

    struct FuncInfo {
        bool isKernel = true;
        Type returnType = Type::makeVoid();
        std::vector<Type> paramTypes;
    };
    std::unordered_map<std::string, FuncInfo> funcSignatures_;

    DiagnosticsEngine& diag_;
    std::vector<std::unordered_map<std::string, VarInfo>> varScopes_;
};

} // namespace kernc
