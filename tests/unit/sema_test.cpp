#include <gtest/gtest.h>
#include "kernc/Lexer.h"
#include "kernc/Parser.h"
#include "kernc/Sema.h"
#include "kernc/Diagnostics.h"

using namespace kernc;

static bool checkSource(const std::string& src, DiagnosticsEngine& diag) {
    Lexer lexer(src, &diag);
    Parser parser(lexer, diag);
    auto program = parser.parseProgram();
    if (!program || diag.hasErrors()) return false;
    Sema sema(diag);
    return sema.checkProgram(*program);
}

TEST(SemaTest, ValidKernel) {
    std::string src = "kernel saxpy(float a, float* x, float* y, int n) {\n"
                      "    int i = block_idx * block_dim + thread_idx;\n"
                      "    if (i < n) {\n"
                      "        y[i] = a * x[i] + y[i];\n"
                      "    }\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_TRUE(ok);
    EXPECT_FALSE(diag.hasErrors());
}

TEST(SemaTest, ImplicitIntToFloatPromotion) {
    std::string src = "kernel test(float* x) {\n"
                      "    x[0] = x[0] + 1; // 1 is int, x[0] is float\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_TRUE(ok);
    EXPECT_FALSE(diag.hasErrors());
}

TEST(SemaTest, UndeclaredVariable) {
    std::string src = "kernel test() {\n"
                      "    int x = y + 1;\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(diag.hasErrors());
    EXPECT_NE(diag.formatAll().find("undeclared variable 'y'"), std::string::npos);
}

TEST(SemaTest, RedeclaredVariableInSameScope) {
    std::string src = "kernel test() {\n"
                      "    int x = 1;\n"
                      "    int x = 2;\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(diag.hasErrors());
    EXPECT_NE(diag.formatAll().find("redefinition of variable 'x'"), std::string::npos);
}

TEST(SemaTest, TypeMismatchInAssignment) {
    std::string src = "kernel test(int* p) {\n"
                      "    int x = 0;\n"
                      "    x = 3.14; // float assigned to int without cast\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(diag.hasErrors());
    EXPECT_NE(diag.formatAll().find("type mismatch"), std::string::npos);
}

TEST(SemaTest, SubscriptNonPointer) {
    std::string src = "kernel test() {\n"
                      "    int x = 5;\n"
                      "    int y = x[0];\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(diag.hasErrors());
    EXPECT_NE(diag.formatAll().find("subscripted value is not a pointer"), std::string::npos);
}

TEST(SemaTest, IntLiteralsAtS32EdgesAccepted) {
    // INT_MAX, INT_MIN written as -2147483648, and hex bit patterns up to 0xFFFFFFFF.
    std::string src = "kernel test(int* o) {\n"
                      "    o[0] = 2147483647;\n"
                      "    o[1] = -2147483648;\n"
                      "    o[2] = 0xFFFFFFFF;\n"
                      "    o[3] = 0x80000000;\n"
                      "}\n";
    DiagnosticsEngine diag;
    EXPECT_TRUE(checkSource(src, diag)) << diag.formatAll();
}

TEST(SemaTest, IntLiteralsBeyondS32Rejected) {
    // 2147483648 is only allowed as the operand of unary minus.
    for (std::string lit : {"2147483648", "3000000000", "0x100000000", "99999999999999999999"}) {
        std::string src = "kernel test(int* o) {\n"
                          "    o[0] = " + lit + ";\n"
                          "}\n";
        DiagnosticsEngine diag;
        EXPECT_FALSE(checkSource(src, diag)) << lit;
        EXPECT_NE(diag.formatAll().find("2:12: error: integer literal '" + lit +
                                        "' is out of range for type 'int'"),
                  std::string::npos) << diag.formatAll();
    }
}

TEST(SemaTest, BareReturnAccepted) {
    std::string src = "kernel test(int* o, int n) {\n"
                      "    if (thread_idx >= n) return;\n"
                      "    o[thread_idx] = 1;\n"
                      "}\n";
    DiagnosticsEngine diag;
    EXPECT_TRUE(checkSource(src, diag)) << diag.formatAll();
}

TEST(SemaTest, ReturnWithValueRejected) {
    std::string src = "kernel test(int n) {\n"
                      "    return n + 1;\n"
                      "}\n";
    DiagnosticsEngine diag;
    EXPECT_FALSE(checkSource(src, diag));
    EXPECT_NE(diag.formatAll().find("2:12: error: kernel 'test' cannot return a value"),
              std::string::npos) << diag.formatAll();
}

TEST(SemaTest, NonIntSubscript) {
    std::string src = "kernel test(float* p) {\n"
                      "    float y = p[1.5];\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(diag.hasErrors());
    EXPECT_NE(diag.formatAll().find("array subscript is not an integer"), std::string::npos);
}
