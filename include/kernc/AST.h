#pragma once

#include "kernc/SourceLocation.h"
#include <memory>
#include <string>
#include <vector>

namespace kernc {

struct Type {
    enum Kind { Void, Int, Float, Ptr, Bool, SharedArray };
    Kind kind = Void;
    Kind pointeeKind = Void; // element type for Ptr and SharedArray

    static Type makeVoid() { return {Void, Void}; }
    static Type makeInt() { return {Int, Void}; }
    static Type makeFloat() { return {Float, Void}; }
    static Type makeIntPtr() { return {Ptr, Int}; }
    static Type makeFloatPtr() { return {Ptr, Float}; }
    static Type makeBool() { return {Bool, Void}; }
    static Type makeSharedArray(Kind elem) { return {SharedArray, elem}; }

    bool isVoid() const { return kind == Void; }
    bool isInt() const { return kind == Int; }
    bool isFloat() const { return kind == Float; }
    bool isPtr() const { return kind == Ptr; }
    bool isBool() const { return kind == Bool; }
    bool isSharedArray() const { return kind == SharedArray; }

    bool operator==(const Type& o) const {
        if (kind != o.kind) return false;
        if (kind == Ptr || kind == SharedArray) return pointeeKind == o.pointeeKind;
        return true;
    }
    bool operator!=(const Type& o) const { return !(*this == o); }

    std::string toString() const {
        switch (kind) {
            case Void: return "void";
            case Int: return "int";
            case Float: return "float";
            case Bool: return "bool";
            case Ptr:
                if (pointeeKind == Int) return "int*";
                if (pointeeKind == Float) return "float*";
                return "ptr";
            case SharedArray:
                return pointeeKind == Float ? "shared float[]" : "shared int[]";
        }
        return "unknown";
    }
};

struct ASTNode {
    SourceRange range;
    virtual ~ASTNode() = default;
};

// Expressions
struct Expr : public ASTNode {
    Type resolvedType = Type::makeVoid();
};

struct IntLiteralExpr : public Expr {
    int64_t value = 0;
    std::string spelling; // source text, e.g. "0xFF"; Sema needs to know hex vs decimal
    explicit IntLiteralExpr(int64_t val, std::string spell = "")
        : value(val), spelling(std::move(spell)) {
        resolvedType = Type::makeInt();
    }
};

struct FloatLiteralExpr : public Expr {
    double value = 0.0;
    explicit FloatLiteralExpr(double val) : value(val) {
        resolvedType = Type::makeFloat();
    }
};

struct BoolLiteralExpr : public Expr {
    bool value = false;
    explicit BoolLiteralExpr(bool val) : value(val) {
        resolvedType = Type::makeBool();
    }
};

struct IdentExpr : public Expr {
    std::string name;
    explicit IdentExpr(std::string n) : name(std::move(n)) {}
};

enum class BuiltinKind {
    ThreadIdx,
    BlockIdx,
    BlockDim
};

struct BuiltinExpr : public Expr {
    BuiltinKind builtin;
    explicit BuiltinExpr(BuiltinKind b) : builtin(b) {
        resolvedType = Type::makeInt();
    }
    std::string name() const {
        switch (builtin) {
            case BuiltinKind::ThreadIdx: return "thread_idx";
            case BuiltinKind::BlockIdx:  return "block_idx";
            case BuiltinKind::BlockDim:  return "block_dim";
        }
        return "";
    }
};

enum class BinaryOp {
    Add, Sub, Mul, Div, Mod,
    Lt, Gt, Le, Ge, Eq, Ne,
    And, Or,
    BitAnd, BitOr, BitXor, Shl, Shr
};

const char* binaryOpToString(BinaryOp op);

struct BinaryExpr : public Expr {
    BinaryOp op;
    std::unique_ptr<Expr> lhs;
    std::unique_ptr<Expr> rhs;
    BinaryExpr(BinaryOp o, std::unique_ptr<Expr> l, std::unique_ptr<Expr> r)
        : op(o), lhs(std::move(l)), rhs(std::move(r)) {}
};

enum class UnaryOp {
    Neg,
    Not,
    BitNot
};

const char* unaryOpToString(UnaryOp op);

struct UnaryExpr : public Expr {
    UnaryOp op;
    std::unique_ptr<Expr> operand;
    UnaryExpr(UnaryOp o, std::unique_ptr<Expr> oper)
        : op(o), operand(std::move(oper)) {}
};

struct IndexExpr : public Expr {
    std::unique_ptr<Expr> base;
    std::unique_ptr<Expr> index;
    IndexExpr(std::unique_ptr<Expr> b, std::unique_ptr<Expr> idx)
        : base(std::move(b)), index(std::move(idx)) {}
};

struct CastExpr : public Expr {
    Type targetType;
    std::unique_ptr<Expr> operand;
    CastExpr(Type t, std::unique_ptr<Expr> op)
        : targetType(t), operand(std::move(op)) {
        resolvedType = t;
    }
};

// `f(a, b)`: a call to a device function (`func`), lowered to a PTX `call`.
struct CallExpr : public Expr {
    std::string callee;
    std::vector<std::unique_ptr<Expr>> args;
    CallExpr(std::string c, std::vector<std::unique_ptr<Expr>> a)
        : callee(std::move(c)), args(std::move(a)) {}
};

// Statements
struct Stmt : public ASTNode {};

struct VarDeclStmt : public Stmt {
    Type type;
    std::string name;
    std::unique_ptr<Expr> init; // can be nullptr
    VarDeclStmt(Type t, std::string n, std::unique_ptr<Expr> i = nullptr)
        : type(t), name(std::move(n)), init(std::move(i)) {}
};

struct AssignStmt : public Stmt {
    std::unique_ptr<Expr> target; // IdentExpr or IndexExpr
    std::unique_ptr<Expr> value;
    AssignStmt(std::unique_ptr<Expr> t, std::unique_ptr<Expr> v)
        : target(std::move(t)), value(std::move(v)) {}
};

struct BlockStmt : public Stmt {
    std::vector<std::unique_ptr<Stmt>> stmts;
    explicit BlockStmt(std::vector<std::unique_ptr<Stmt>> s = {})
        : stmts(std::move(s)) {}
};

struct IfStmt : public Stmt {
    std::unique_ptr<Expr> cond;
    std::unique_ptr<Stmt> thenBranch;
    std::unique_ptr<Stmt> elseBranch; // can be nullptr
    IfStmt(std::unique_ptr<Expr> c, std::unique_ptr<Stmt> t, std::unique_ptr<Stmt> e = nullptr)
        : cond(std::move(c)), thenBranch(std::move(t)), elseBranch(std::move(e)) {}
};

struct WhileStmt : public Stmt {
    std::unique_ptr<Expr> cond;
    std::unique_ptr<Stmt> body;
    WhileStmt(std::unique_ptr<Expr> c, std::unique_ptr<Stmt> b)
        : cond(std::move(c)), body(std::move(b)) {}
};

struct ForStmt : public Stmt {
    std::unique_ptr<Stmt> init;       // VarDeclStmt or AssignStmt or ExprStmt or nullptr
    std::unique_ptr<Expr> cond;       // can be nullptr (defaults to true)
    std::unique_ptr<Stmt> step;       // AssignStmt or ExprStmt or nullptr
    std::unique_ptr<Stmt> body;
    ForStmt(std::unique_ptr<Stmt> i, std::unique_ptr<Expr> c, std::unique_ptr<Stmt> s, std::unique_ptr<Stmt> b)
        : init(std::move(i)), cond(std::move(c)), step(std::move(s)), body(std::move(b)) {}
};

struct ExprStmt : public Stmt {
    std::unique_ptr<Expr> expr;
    explicit ExprStmt(std::unique_ptr<Expr> e) : expr(std::move(e)) {}
};

struct BarrierStmt : public Stmt {};

// Kernels return void; the value is parsed anyway so Sema can report it clearly.
struct ReturnStmt : public Stmt {
    std::unique_ptr<Expr> value; // can be nullptr
    explicit ReturnStmt(std::unique_ptr<Expr> v = nullptr) : value(std::move(v)) {}
};

// `shared float buf[256];`: a per-block array in the .shared state space, sized at compile time.
struct SharedDeclStmt : public Stmt {
    Type elemType;
    std::string name;
    int64_t size = 0;
    SharedDeclStmt(Type t, std::string n, int64_t s) : elemType(t), name(std::move(n)), size(s) {}
};

// Kernel / Top-level
struct Param {
    Type type;
    std::string name;
    SourceLocation loc;
};

// A `kernel` (PTX .entry, launched from the host, returns void) or a `func` device function
// (PTX .func, called from kernels and other functions, may return a value).
struct FunctionDecl : public ASTNode {
    bool isKernel = true;
    Type returnType = Type::makeVoid();
    std::string name;
    std::vector<Param> params;
    std::vector<std::unique_ptr<Stmt>> body;
};

struct Program : public ASTNode {
    std::vector<std::unique_ptr<FunctionDecl>> functions;
};

} // namespace kernc
