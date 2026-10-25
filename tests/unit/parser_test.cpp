#include <gtest/gtest.h>
#include "kernc/Lexer.h"
#include "kernc/Parser.h"
#include "kernc/Diagnostics.h"

using namespace kernc;

static std::unique_ptr<KernelDecl> parseKernel(const std::string& src) {
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
