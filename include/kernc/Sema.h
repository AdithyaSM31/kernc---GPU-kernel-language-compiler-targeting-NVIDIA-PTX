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
    bool checkFunction(FunctionDecl& kernel);

private:
    void checkStmt(Stmt& stmt);
    void checkVarDecl(VarDeclStmt& stmt);
    void checkAssign(AssignStmt& stmt);
    void checkIf(IfStmt& stmt);
    void checkWhile(WhileStmt& stmt);
    void checkFor(ForStmt& stmt);
    void checkBlock(BlockStmt& stmt);
    void checkSharedDecl(SharedDeclStmt& stmt);

    Type checkExpr(Expr& expr);
    Type checkBinaryExpr(BinaryExpr& expr);
    Type checkUnaryExpr(UnaryExpr& expr);
    Type checkIndexExpr(IndexExpr& expr);
    Type checkMemberExpr(MemberExpr& expr);
    Type checkCastExpr(CastExpr& expr);
    Type checkIdentExpr(IdentExpr& expr, bool allowSharedArray = false);
    Type checkIntLiteral(IntLiteralExpr& expr, bool negated);

    Type checkCallExpr(CallExpr& expr, bool isExprStmt);

    bool isConditionType(const Type& t) const {
        return t.isBool() || t.isInt();
    }

    struct FunctionSignature {
        std::string name;
        bool isKernel = true;
        Type returnType = Type::makeVoid();
        std::vector<Type> paramTypes;
        SourceLocation loc;
    };

    DiagnosticsEngine& diag_;
    SymbolTable symTable_;
    std::unordered_map<std::string, FunctionSignature> functions_;
    FunctionDecl* currentFunction_ = nullptr;
    std::string currentKernel_; // for diagnostics such as "kernel 'k' cannot return a value"

    // Static shared memory per block on sm_86 (ptxas reports the limit as 0xc000).
    static constexpr int64_t kStaticSharedLimit = 48 * 1024;
    int64_t sharedBytes_ = 0;   // shared memory declared so far in the current kernel
};

} // namespace kernc
