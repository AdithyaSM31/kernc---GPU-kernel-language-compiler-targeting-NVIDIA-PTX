#include <gtest/gtest.h>
#include "kernc/Lexer.h"
#include "kernc/Parser.h"
#include "kernc/Diagnostics.h"

using namespace kernc;

static std::unique_ptr<FunctionDecl> parseKernel(const std::string& src) {
    DiagnosticsEngine diag;
    Lexer lexer(src, &diag);
    Parser parser(lexer, diag);
    return parser.parseKernel();
}

TEST(ParserTest, KernelDeclAndParams) {
    std::string src = "kernel test(float a, float* x, int n) {}";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);
    EXPECT_EQ(k->name, "test");
    ASSERT_EQ(k->params.size(), 3);

    EXPECT_EQ(k->params[0].type.kind, Type::Float);
    EXPECT_EQ(k->params[0].name, "a");

    EXPECT_EQ(k->params[1].type.kind, Type::Ptr);
    EXPECT_EQ(k->params[1].type.pointeeKind, Type::Float);
    EXPECT_EQ(k->params[1].name, "x");

    EXPECT_EQ(k->params[2].type.kind, Type::Int);
    EXPECT_EQ(k->params[2].name, "n");
}

TEST(ParserTest, VarDeclAndInit) {
    std::string src = "kernel test() { int x = 42; float y; }";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);
    ASSERT_EQ(k->body.size(), 2);

    auto* decl1 = dynamic_cast<VarDeclStmt*>(k->body[0].get());
    ASSERT_NE(decl1, nullptr);
    EXPECT_EQ(decl1->name, "x");
    EXPECT_EQ(decl1->type.kind, Type::Int);
    ASSERT_NE(decl1->init, nullptr);

    auto* initVal = dynamic_cast<IntLiteralExpr*>(decl1->init.get());
    ASSERT_NE(initVal, nullptr);
    EXPECT_EQ(initVal->value, 42);

    auto* decl2 = dynamic_cast<VarDeclStmt*>(k->body[1].get());
    ASSERT_NE(decl2, nullptr);
    EXPECT_EQ(decl2->name, "y");
    EXPECT_EQ(decl2->type.kind, Type::Float);
    EXPECT_EQ(decl2->init, nullptr);
}

TEST(ParserTest, PrecedenceClimbing) {
    // a + b * c should parse as a + (b * c)
    std::string src = "kernel test() { int x = a + b * c; }";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);
    ASSERT_EQ(k->body.size(), 1);

    auto* decl = dynamic_cast<VarDeclStmt*>(k->body[0].get());
    ASSERT_NE(decl, nullptr);

    auto* addBin = dynamic_cast<BinaryExpr*>(decl->init.get());
    ASSERT_NE(addBin, nullptr);
    EXPECT_EQ(addBin->op, BinaryOp::Add);

    auto* lhs = dynamic_cast<IdentExpr*>(addBin->lhs.get());
    ASSERT_NE(lhs, nullptr);
    EXPECT_EQ(lhs->name, "a");

    auto* mulBin = dynamic_cast<BinaryExpr*>(addBin->rhs.get());
    ASSERT_NE(mulBin, nullptr);
    EXPECT_EQ(mulBin->op, BinaryOp::Mul);
}

TEST(ParserTest, LeftAssociativity) {
    // a - b - c should parse as (a - b) - c
    std::string src = "kernel test() { int x = a - b - c; }";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);

    auto* decl = dynamic_cast<VarDeclStmt*>(k->body[0].get());
    ASSERT_NE(decl, nullptr);

    auto* outerSub = dynamic_cast<BinaryExpr*>(decl->init.get());
    ASSERT_NE(outerSub, nullptr);
    EXPECT_EQ(outerSub->op, BinaryOp::Sub);

    auto* innerSub = dynamic_cast<BinaryExpr*>(outerSub->lhs.get());
    ASSERT_NE(innerSub, nullptr);
    EXPECT_EQ(innerSub->op, BinaryOp::Sub);

    auto* cIdent = dynamic_cast<IdentExpr*>(outerSub->rhs.get());
    ASSERT_NE(cIdent, nullptr);
    EXPECT_EQ(cIdent->name, "c");
}

TEST(ParserTest, ParenthesizedExprPreservedInAST) {
    std::string src = "kernel test() { int x = (42); }";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);
    ASSERT_EQ(k->body.size(), 1u);

    auto* decl = dynamic_cast<VarDeclStmt*>(k->body[0].get());
    ASSERT_NE(decl, nullptr);

    auto* paren = dynamic_cast<ParenExpr*>(decl->init.get());
    ASSERT_NE(paren, nullptr);

    auto* lit = dynamic_cast<IntLiteralExpr*>(paren->subExpr.get());
    ASSERT_NE(lit, nullptr);
    EXPECT_EQ(lit->value, 42);
}

TEST(ParserTest, IndexAndCastExpr) {
    std::string src = "kernel test(float* p) { float y = cast<float>(p[1]); }";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);

    auto* decl = dynamic_cast<VarDeclStmt*>(k->body[0].get());
    ASSERT_NE(decl, nullptr);

    auto* cast = dynamic_cast<CastExpr*>(decl->init.get());
    ASSERT_NE(cast, nullptr);
    EXPECT_EQ(cast->targetType.kind, Type::Float);

    auto* idx = dynamic_cast<IndexExpr*>(cast->operand.get());
    ASSERT_NE(idx, nullptr);
    auto* base = dynamic_cast<IdentExpr*>(idx->base.get());
    ASSERT_NE(base, nullptr);
    EXPECT_EQ(base->name, "p");
}

TEST(ParserTest, ControlFlowStatements) {
    std::string src = "kernel test(int n) {\n"
                      "    if (n > 0) { n = n - 1; } else { n = 0; }\n"
                      "    while (n < 10) { n = n + 1; }\n"
                      "    for (int i = 0; i < n; i = i + 1) { }\n"
                      "}\n";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);
    ASSERT_EQ(k->body.size(), 3);

    auto* ifStmt = dynamic_cast<IfStmt*>(k->body[0].get());
    ASSERT_NE(ifStmt, nullptr);
    ASSERT_NE(ifStmt->elseBranch, nullptr);

    auto* whileStmt = dynamic_cast<WhileStmt*>(k->body[1].get());
    ASSERT_NE(whileStmt, nullptr);

    auto* forStmt = dynamic_cast<ForStmt*>(k->body[2].get());
    ASSERT_NE(forStmt, nullptr);
    ASSERT_NE(forStmt->init, nullptr);
    ASSERT_NE(forStmt->cond, nullptr);
    ASSERT_NE(forStmt->step, nullptr);
}

TEST(ParserTest, DeviceFunctionsAndCalls) {
    std::string src = "func float square(float x) { return x * x; }\n"
                      "func void touch() { }\n"
                      "kernel k(float* o) { o[0] = square(2.0); touch(); }\n";
    DiagnosticsEngine diag;
    Lexer lexer(src, &diag);
    Parser parser(lexer, diag);
    auto prog = parser.parseProgram();
    ASSERT_NE(prog, nullptr) << diag.formatAll();
    ASSERT_EQ(prog->functions.size(), 3u);

    const FunctionDecl& square = *prog->functions[0];
    EXPECT_FALSE(square.isKernel);
    EXPECT_EQ(square.name, "square");
    EXPECT_EQ(square.returnType.kind, Type::Float);
    ASSERT_EQ(square.params.size(), 1u);
    EXPECT_EQ(square.params[0].type.kind, Type::Float);

    EXPECT_FALSE(prog->functions[1]->isKernel);
    EXPECT_EQ(prog->functions[1]->returnType.kind, Type::Void);

    const FunctionDecl& k = *prog->functions[2];
    EXPECT_TRUE(k.isKernel);
    ASSERT_EQ(k.body.size(), 2u);
    auto* assign = dynamic_cast<AssignStmt*>(k.body[0].get());
    ASSERT_NE(assign, nullptr);
    auto* call = dynamic_cast<CallExpr*>(assign->value.get());
    ASSERT_NE(call, nullptr);
    EXPECT_EQ(call->callee, "square");
    ASSERT_EQ(call->args.size(), 1u);
    EXPECT_NE(dynamic_cast<FloatLiteralExpr*>(call->args[0].get()), nullptr);

    // A void call is an expression statement.
    auto* stmt = dynamic_cast<ExprStmt*>(k.body[1].get());
    ASSERT_NE(stmt, nullptr);
    auto* voidCall = dynamic_cast<CallExpr*>(stmt->expr.get());
    ASSERT_NE(voidCall, nullptr);
    EXPECT_TRUE(voidCall->args.empty());
}

TEST(ParserTest, SharedArrayDecl) {
    auto k = parseKernel("kernel test() { shared float buf[256]; __shared__ int idx[8]; }");
    ASSERT_NE(k, nullptr);
    ASSERT_EQ(k->body.size(), 2u);

    auto* buf = dynamic_cast<SharedDeclStmt*>(k->body[0].get());
    ASSERT_NE(buf, nullptr);
    EXPECT_EQ(buf->elemType.kind, Type::Float);
    EXPECT_EQ(buf->name, "buf");
    EXPECT_EQ(buf->size, 256);

    auto* idx = dynamic_cast<SharedDeclStmt*>(k->body[1].get());
    ASSERT_NE(idx, nullptr);
    EXPECT_EQ(idx->elemType.kind, Type::Int);
    EXPECT_EQ(idx->size, 8);
}

TEST(ParserTest, SharedArraySizeMustBeLiteral) {
    // Static shared memory is sized at compile time, so the size is a literal.
    DiagnosticsEngine diag;
    Lexer lexer("kernel test(int n) { shared float buf[n]; }", &diag);
    Parser parser(lexer, diag);
    parser.parseKernel();
    EXPECT_NE(diag.formatAll().find("1:39: error: expected integer literal as shared array size"),
              std::string::npos) << diag.formatAll();
}

TEST(ParserTest, ReturnStatement) {
    std::string src = "kernel test(int n) {\n"
                      "    if (n > 0) return;\n"
                      "    return n;\n"
                      "}\n";
    DiagnosticsEngine diag;
    Lexer lexer(src, &diag);
    Parser parser(lexer, diag);
    auto k = parser.parseKernel();
    ASSERT_NE(k, nullptr) << diag.formatAll();
    EXPECT_FALSE(diag.hasErrors()) << diag.formatAll();
    ASSERT_EQ(k->body.size(), 2u);

    auto* ifStmt = dynamic_cast<IfStmt*>(k->body[0].get());
    ASSERT_NE(ifStmt, nullptr);
    auto* bare = dynamic_cast<ReturnStmt*>(ifStmt->thenBranch.get());
    ASSERT_NE(bare, nullptr);
    EXPECT_EQ(bare->value, nullptr);

    // A value is parsed (and later rejected by Sema), not treated as a syntax error.
    auto* withValue = dynamic_cast<ReturnStmt*>(k->body[1].get());
    ASSERT_NE(withValue, nullptr);
    EXPECT_NE(dynamic_cast<IdentExpr*>(withValue->value.get()), nullptr);
}

TEST(ParserTest, BitwiseOperatorPrecedence) {
    // a | b ^ c & d should parse as a | (b ^ (c & d))
    std::string src = "kernel test() { int x = a | b ^ c & d; }";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);
    ASSERT_EQ(k->body.size(), 1u);
    auto* decl = dynamic_cast<VarDeclStmt*>(k->body[0].get());
    ASSERT_NE(decl, nullptr);
    auto* orBin = dynamic_cast<BinaryExpr*>(decl->init.get());
    ASSERT_NE(orBin, nullptr);
    EXPECT_EQ(orBin->op, BinaryOp::BitOr);

    auto* xorBin = dynamic_cast<BinaryExpr*>(orBin->rhs.get());
    ASSERT_NE(xorBin, nullptr);
    EXPECT_EQ(xorBin->op, BinaryOp::BitXor);

    auto* andBin = dynamic_cast<BinaryExpr*>(xorBin->rhs.get());
    ASSERT_NE(andBin, nullptr);
    EXPECT_EQ(andBin->op, BinaryOp::BitAnd);
}

TEST(ParserTest, ShiftAndAdditivePrecedence) {
    // a + b << c should parse as (a + b) << c
    std::string src = "kernel test() { int x = a + b << c; }";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);
    ASSERT_EQ(k->body.size(), 1u);
    auto* decl = dynamic_cast<VarDeclStmt*>(k->body[0].get());
    ASSERT_NE(decl, nullptr);
    auto* shlBin = dynamic_cast<BinaryExpr*>(decl->init.get());
    ASSERT_NE(shlBin, nullptr);
    EXPECT_EQ(shlBin->op, BinaryOp::Shl);

    auto* addBin = dynamic_cast<BinaryExpr*>(shlBin->lhs.get());
    ASSERT_NE(addBin, nullptr);
    EXPECT_EQ(addBin->op, BinaryOp::Add);
}

TEST(ParserTest, UnaryBitwiseNot) {
    std::string src = "kernel test() { int x = ~a; }";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);
    ASSERT_EQ(k->body.size(), 1u);
    auto* decl = dynamic_cast<VarDeclStmt*>(k->body[0].get());
    ASSERT_NE(decl, nullptr);
    auto* notExpr = dynamic_cast<UnaryExpr*>(decl->init.get());
    ASSERT_NE(notExpr, nullptr);
    EXPECT_EQ(notExpr->op, UnaryOp::BitNot);
}

TEST(ParserTest, MultiDimensionalBuiltins) {
    std::string src = "kernel test() {\n"
                      "    int tx = thread_idx_x; int ty = thread_idx_y; int tz = thread_idx_z;\n"
                      "    int bx = block_idx_x;  int by = block_idx_y;  int bz = block_idx_z;\n"
                      "    int dx = block_dim_x;  int dy = block_dim_y;  int dz = block_dim_z;\n"
                      "    int gx = grid_dim_x;   int gy = grid_dim_y;   int gz = grid_dim_z;\n"
                      "    int dot_tx = thread_idx.x; int dot_ty = thread_idx.y; int dot_tz = thread_idx.z;\n"
                      "    int dot_cuda = threadIdx.x + blockIdx.y * blockDim.z + gridDim.x;\n"
                      "}\n";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);
    ASSERT_EQ(k->body.size(), 16u);

    auto checkBuiltin = [&](size_t idx, BuiltinKind expected) {
        auto* decl = dynamic_cast<VarDeclStmt*>(k->body[idx].get());
        ASSERT_NE(decl, nullptr);
        auto* b = dynamic_cast<BuiltinExpr*>(decl->init.get());
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(b->builtin, expected);
    };

    checkBuiltin(0, BuiltinKind::ThreadIdxX);
    checkBuiltin(1, BuiltinKind::ThreadIdxY);
    checkBuiltin(2, BuiltinKind::ThreadIdxZ);
    checkBuiltin(3, BuiltinKind::BlockIdxX);
    checkBuiltin(4, BuiltinKind::BlockIdxY);
    checkBuiltin(5, BuiltinKind::BlockIdxZ);
    checkBuiltin(6, BuiltinKind::BlockDimX);
    checkBuiltin(7, BuiltinKind::BlockDimY);
    checkBuiltin(8, BuiltinKind::BlockDimZ);
    checkBuiltin(9, BuiltinKind::GridDimX);
    checkBuiltin(10, BuiltinKind::GridDimY);
    checkBuiltin(11, BuiltinKind::GridDimZ);
    checkBuiltin(12, BuiltinKind::ThreadIdxX);
    checkBuiltin(13, BuiltinKind::ThreadIdxY);
    checkBuiltin(14, BuiltinKind::ThreadIdxZ);
}

TEST(ParserTest, VectorTypesAndConstructors) {
    std::string src = "kernel test(int4* in, float2* out) {\n"
                      "    int2 v2 = int2(1, 2);\n"
                      "    int4 v4 = make_int4(1, 2, 3, 4);\n"
                      "    float2 f2 = float2(1.0, 2.0);\n"
                      "    float4 f4 = make_float4(1.0, 2.0, 3.0, 4.0);\n"
                      "    shared int4 s_buf[64];\n"
                      "    int a = v4.x;\n"
                      "    int b = v4.w;\n"
                      "    float c = in[0].y;\n"
                      "    v2.x = 10;\n"
                      "    in[1].z = 20;\n"
                      "}\n";
    auto k = parseKernel(src);
    ASSERT_NE(k, nullptr);
    ASSERT_EQ(k->params.size(), 2u);
    EXPECT_TRUE(k->params[0].type.isPtr());
    EXPECT_EQ(k->params[0].type.pointeeKind, Type::Int4);
    EXPECT_TRUE(k->params[1].type.isPtr());
    EXPECT_EQ(k->params[1].type.pointeeKind, Type::Float2);

    ASSERT_EQ(k->body.size(), 10u);
    auto* d0 = dynamic_cast<VarDeclStmt*>(k->body[0].get());
    ASSERT_NE(d0, nullptr);
    EXPECT_EQ(d0->type.kind, Type::Int2);
    auto* call0 = dynamic_cast<CallExpr*>(d0->init.get());
    ASSERT_NE(call0, nullptr);
    EXPECT_EQ(call0->callee, "int2");
    EXPECT_EQ(call0->args.size(), 2u);

    auto* d5 = dynamic_cast<VarDeclStmt*>(k->body[5].get());
    ASSERT_NE(d5, nullptr);
    auto* mem5 = dynamic_cast<MemberExpr*>(d5->init.get());
    ASSERT_NE(mem5, nullptr);
    EXPECT_EQ(mem5->member, "x");

    auto* d7 = dynamic_cast<VarDeclStmt*>(k->body[7].get());
    ASSERT_NE(d7, nullptr);
    auto* mem7 = dynamic_cast<MemberExpr*>(d7->init.get());
    ASSERT_NE(mem7, nullptr);
    EXPECT_EQ(mem7->member, "y");
    auto* idx7 = dynamic_cast<IndexExpr*>(mem7->base.get());
    ASSERT_NE(idx7, nullptr);

    auto* a8 = dynamic_cast<AssignStmt*>(k->body[8].get());
    ASSERT_NE(a8, nullptr);
    auto* mem8 = dynamic_cast<MemberExpr*>(a8->target.get());
    ASSERT_NE(mem8, nullptr);
    EXPECT_EQ(mem8->member, "x");

    auto* a9 = dynamic_cast<AssignStmt*>(k->body[9].get());
    ASSERT_NE(a9, nullptr);
    auto* mem9 = dynamic_cast<MemberExpr*>(a9->target.get());
    ASSERT_NE(mem9, nullptr);
    EXPECT_EQ(mem9->member, "z");
}
