#pragma once

#include "kernc/SourceLocation.h"
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace kernc {

struct Type {
    enum Kind { Void, Int, Float, Bool, Int2, Int4, Float2, Float4, Ptr, SharedArray };
    Kind kind = Void;
    Kind pointeeKind = Void; // element type for Ptr and SharedArray

    Type() = default;
    Type(Kind k, Kind p = Void) : kind(k), pointeeKind(p) {}

    static Type makeVoid() { return {Void, Void}; }
    static Type makeInt() { return {Int, Void}; }
    static Type makeFloat() { return {Float, Void}; }
    static Type makeBool() { return {Bool, Void}; }
    static Type makeInt2() { return {Int2, Void}; }
    static Type makeInt4() { return {Int4, Void}; }
    static Type makeFloat2() { return {Float2, Void}; }
    static Type makeFloat4() { return {Float4, Void}; }
    static Type makeIntPtr() { return {Ptr, Int}; }
    static Type makeFloatPtr() { return {Ptr, Float}; }
    static Type makeInt2Ptr() { return {Ptr, Int2}; }
    static Type makeInt4Ptr() { return {Ptr, Int4}; }
    static Type makeFloat2Ptr() { return {Ptr, Float2}; }
    static Type makeFloat4Ptr() { return {Ptr, Float4}; }
    static Type makePtr(Kind elem) { return {Ptr, elem}; }
    static Type makeSharedArray(Kind elem) { return {SharedArray, elem}; }

    bool isVoid() const { return kind == Void; }
    bool isInt() const { return kind == Int; }
    bool isFloat() const { return kind == Float; }
    bool isBool() const { return kind == Bool; }
    bool isInt2() const { return kind == Int2; }
    bool isInt4() const { return kind == Int4; }
    bool isFloat2() const { return kind == Float2; }
    bool isFloat4() const { return kind == Float4; }
    bool isPtr() const { return kind == Ptr; }
    bool isSharedArray() const { return kind == SharedArray; }

    bool isIntVector() const { return kind == Int2 || kind == Int4; }
    bool isFloatVector() const { return kind == Float2 || kind == Float4; }
    bool isVector() const { return isIntVector() || isFloatVector(); }
    int vectorWidth() const {
        if (kind == Int2 || kind == Float2) return 2;
        if (kind == Int4 || kind == Float4) return 4;
        return 1;
    }
    Kind vectorElementKind() const {
        if (isIntVector()) return Int;
        if (isFloatVector()) return Float;
        return Void;
    }
    Type vectorElementType() const {
        if (isIntVector()) return makeInt();
        if (isFloatVector()) return makeFloat();
        return makeVoid();
    }
    int sizeInBytes() const {
        switch (kind) {
            case Int:
            case Float: return 4;
            case Int2:
            case Float2: return 8;
            case Int4:
            case Float4: return 16;
            case Ptr: return 8;
            case Bool: return 1;
            case SharedArray:
            case Void: return 0;
        }
        return 0;
    }

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
            case Int2: return "int2";
            case Int4: return "int4";
            case Float2: return "float2";
            case Float4: return "float4";
            case Ptr:
                if (pointeeKind == Int) return "int*";
                if (pointeeKind == Float) return "float*";
                if (pointeeKind == Int2) return "int2*";
                if (pointeeKind == Int4) return "int4*";
                if (pointeeKind == Float2) return "float2*";
                if (pointeeKind == Float4) return "float4*";
                return "ptr";
            case SharedArray:
                if (pointeeKind == Float) return "shared float[]";
                if (pointeeKind == Int) return "shared int[]";
                if (pointeeKind == Int2) return "shared int2[]";
                if (pointeeKind == Int4) return "shared int4[]";
                if (pointeeKind == Float2) return "shared float2[]";
                if (pointeeKind == Float4) return "shared float4[]";
                return "shared array";
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

struct MemberExpr : public Expr {
    std::unique_ptr<Expr> base;
    std::string member;
    MemberExpr(std::unique_ptr<Expr> b, std::string m)
        : base(std::move(b)), member(std::move(m)) {}
};

struct CastExpr : public Expr {
    Type targetType;
    std::unique_ptr<Expr> operand;
    CastExpr(Type t, std::unique_ptr<Expr> op)
        : targetType(t), operand(std::move(op)) {
        resolvedType = t;
    }
};

// Parenthesized expression: preserves syntax grouping in the AST so that integer literal
// range validation and source locations adhere strictly to C99 and Java JLS 3.10.1 rules.
struct ParenExpr : public Expr {
    std::unique_ptr<Expr> subExpr;
    explicit ParenExpr(std::unique_ptr<Expr> sub)
        : subExpr(std::move(sub)) {}
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

enum class AtomicIntrinsic {
    Add,
    Sub,
    Min,
    Max,
    And,
    Or,
    Xor,
    Exch,
    Cas
};

inline std::optional<AtomicIntrinsic> getAtomicIntrinsic(std::string_view name) {
    if (name == "atomic_add" || name == "__atomic_add" || name == "atomicAdd" || name == "__atomicAdd")
        return AtomicIntrinsic::Add;
    if (name == "atomic_sub" || name == "__atomic_sub" || name == "atomicSub" || name == "__atomicSub")
        return AtomicIntrinsic::Sub;
    if (name == "atomic_min" || name == "__atomic_min" || name == "atomicMin" || name == "__atomicMin")
        return AtomicIntrinsic::Min;
    if (name == "atomic_max" || name == "__atomic_max" || name == "atomicMax" || name == "__atomicMax")
        return AtomicIntrinsic::Max;
    if (name == "atomic_and" || name == "__atomic_and" || name == "atomicAnd" || name == "__atomicAnd")
        return AtomicIntrinsic::And;
    if (name == "atomic_or" || name == "__atomic_or" || name == "atomicOr" || name == "__atomicOr")
        return AtomicIntrinsic::Or;
    if (name == "atomic_xor" || name == "__atomic_xor" || name == "atomicXor" || name == "__atomicXor")
        return AtomicIntrinsic::Xor;
    if (name == "atomic_exch" || name == "__atomic_exch" || name == "atomicExch" || name == "__atomicExch" ||
        name == "atomic_exchange" || name == "__atomic_exchange" || name == "atomicExchange" || name == "__atomicExchange")
        return AtomicIntrinsic::Exch;
    if (name == "atomic_cas" || name == "__atomic_cas" || name == "atomicCAS" || name == "__atomicCAS" ||
        name == "atomicCas" || name == "__atomicCas")
        return AtomicIntrinsic::Cas;
    return std::nullopt;
}

enum class VectorIntrinsic {
    LoadInt2,
    LoadInt4,
    LoadFloat2,
    LoadFloat4,
    StoreInt2,
    StoreInt4,
    StoreFloat2,
    StoreFloat4
};

inline std::optional<VectorIntrinsic> getVectorIntrinsic(std::string_view name) {
    if (name == "load_int2") return VectorIntrinsic::LoadInt2;
    if (name == "load_int4") return VectorIntrinsic::LoadInt4;
    if (name == "load_float2") return VectorIntrinsic::LoadFloat2;
    if (name == "load_float4") return VectorIntrinsic::LoadFloat4;
    if (name == "store_int2") return VectorIntrinsic::StoreInt2;
    if (name == "store_int4") return VectorIntrinsic::StoreInt4;
    if (name == "store_float2") return VectorIntrinsic::StoreFloat2;
    if (name == "store_float4") return VectorIntrinsic::StoreFloat4;
    return std::nullopt;
}

inline bool isVectorConstructor(std::string_view name) {
    return name == "int2" || name == "make_int2" ||
           name == "int4" || name == "make_int4" ||
           name == "float2" || name == "make_float2" ||
           name == "float4" || name == "make_float4";
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
