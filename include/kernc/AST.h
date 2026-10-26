#pragma once

#include "kernc/SourceLocation.h"
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
    ThreadIdxX,
    ThreadIdxY,
    ThreadIdxZ,
    BlockIdxX,
    BlockIdxY,
    BlockIdxZ,
    BlockDimX,
    BlockDimY,
    BlockDimZ,
    GridDimX,
    GridDimY,
    GridDimZ,

    // Aliases for 1D code
    ThreadIdx = ThreadIdxX,
    BlockIdx  = BlockIdxX,
    BlockDim  = BlockDimX,
    GridDim   = GridDimX
};

struct BuiltinExpr : public Expr {
    BuiltinKind builtin;
    std::string spelling;
    explicit BuiltinExpr(BuiltinKind b, std::string s = "") : builtin(b), spelling(std::move(s)) {
        resolvedType = Type::makeInt();
    }
    std::string name() const {
        if (!spelling.empty()) return spelling;
        switch (builtin) {
            case BuiltinKind::ThreadIdxX: return "thread_idx";
            case BuiltinKind::ThreadIdxY: return "thread_idx_y";
            case BuiltinKind::ThreadIdxZ: return "thread_idx_z";
            case BuiltinKind::BlockIdxX:  return "block_idx";
            case BuiltinKind::BlockIdxY:  return "block_idx_y";
            case BuiltinKind::BlockIdxZ:  return "block_idx_z";
            case BuiltinKind::BlockDimX:  return "block_dim";
            case BuiltinKind::BlockDimY:  return "block_dim_y";
            case BuiltinKind::BlockDimZ:  return "block_dim_z";
            case BuiltinKind::GridDimX:   return "grid_dim";
            case BuiltinKind::GridDimY:   return "grid_dim_y";
            case BuiltinKind::GridDimZ:   return "grid_dim_z";
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

enum class WarpIntrinsic {
    ShflSync,
    ShflDownSync,
    ShflUpSync,
    ShflXorSync,
    BallotSync,
    Activemask
};

inline std::optional<WarpIntrinsic> getWarpIntrinsic(std::string_view name) {
    if (name == "shfl_sync" || name == "__shfl_sync") return WarpIntrinsic::ShflSync;
    if (name == "shfl_down_sync" || name == "__shfl_down_sync") return WarpIntrinsic::ShflDownSync;
    if (name == "shfl_up_sync" || name == "__shfl_up_sync") return WarpIntrinsic::ShflUpSync;
    if (name == "shfl_xor_sync" || name == "__shfl_xor_sync") return WarpIntrinsic::ShflXorSync;
    if (name == "ballot_sync" || name == "__ballot_sync") return WarpIntrinsic::BallotSync;
    if (name == "activemask" || name == "__activemask") return WarpIntrinsic::Activemask;
    return std::nullopt;
}

// `f(a, b)`: a call to a device function (`func`) or warp intrinsic, lowered to PTX.
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
