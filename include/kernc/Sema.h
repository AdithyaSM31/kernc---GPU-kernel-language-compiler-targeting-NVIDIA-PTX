#pragma once

#include "kernc/AST.h"
#include "kernc/Diagnostics.h"
#include <unordered_map>
#include <vector>
#include <string>

namespace kernc {

class SymbolTable {
public:
    void enterScope() {
        scopes_.emplace_back();
    }

    void exitScope() {
        if (!scopes_.empty()) {
            scopes_.pop_back();
        }
    }

    bool declare(const std::string& name, Type type) {
        if (scopes_.empty()) return false;
        auto& current = scopes_.back();
        if (current.find(name) != current.end()) {
            return false; // Already declared in this scope
        }
        current[name] = type;
        return true;
    }

    bool isDeclaredInCurrentScope(const std::string& name) const {
        if (scopes_.empty()) return false;
        return scopes_.back().find(name) != scopes_.back().end();
    }

    const Type* lookup(const std::string& name) const {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) {
                return &found->second;
            }
        }
        return nullptr;
    }

private:
    std::vector<std::unordered_map<std::string, Type>> scopes_;
};

class Sema {
public:
    explicit Sema(DiagnosticsEngine& diag);

    bool checkProgram(Program& program);
    bool checkKernel(KernelDecl& kernel);

private:
    void checkStmt(Stmt& stmt);
    void checkVarDecl(VarDeclStmt& stmt);
    void checkAssign(AssignStmt& stmt);
    void checkIf(IfStmt& stmt);
    void checkWhile(WhileStmt& stmt);
    void checkFor(ForStmt& stmt);
    void checkBlock(BlockStmt& stmt);

    Type checkExpr(Expr& expr);
    Type checkBinaryExpr(BinaryExpr& expr);
    Type checkUnaryExpr(UnaryExpr& expr);
    Type checkIndexExpr(IndexExpr& expr);
    Type checkCastExpr(CastExpr& expr);
    Type checkIdentExpr(IdentExpr& expr);

    bool isConditionType(const Type& t) const {
        return t.isPred() || t.isInt();
    }

    DiagnosticsEngine& diag_;
    SymbolTable symTable_;
};

} // namespace kernc
