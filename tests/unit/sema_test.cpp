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

TEST(SemaTest, BoolValuesAccepted) {
    std::string src = "kernel test(int* o, float* f, int n) {\n"
                      "    bool t = true;\n"
                      "    bool b = n < 10;\n"
                      "    bool c = t && !b || n == 3;\n"
                      "    bool d = c == b;\n"
                      "    bool e = cast<bool>(n);\n"
                      "    bool g = cast<bool>(f[0]);\n"
                      "    if (d != e) { o[0] = cast<int>(g); }\n"
                      "    while (b) { b = false; }\n"
                      "    f[1] = cast<float>(t);\n"
                      "}\n";
    DiagnosticsEngine diag;
    EXPECT_TRUE(checkSource(src, diag)) << diag.formatAll();
}

TEST(SemaTest, BoolTypeErrorsNameBool) {
    // bool does not mix with int/float except through cast<>, and errors must say 'bool'
    // (not the internal predicate type).
    const std::pair<const char*, const char*> cases[] = {
        {"int x = true;", "cannot initialize 'int' with 'bool'"},
        {"bool b = 1;", "cannot initialize 'bool' with 'int'"},
        {"int x = true + 1;", "invalid operands to binary operator '+' ('bool' and 'int')"},
        {"bool b = true < false;", "invalid operands to comparison operator '<' ('bool' and 'bool')"},
        {"bool b = true == 1;", "invalid operands to comparison operator '==' ('bool' and 'int')"},
        {"o[0] = n < 3;", "cannot assign 'bool' to 'int'"},
    };
    for (const auto& [stmt, expected] : cases) {
        std::string src = std::string("kernel test(int* o, int n) {\n    ") + stmt + "\n}\n";
        DiagnosticsEngine diag;
        EXPECT_FALSE(checkSource(src, diag)) << stmt;
        EXPECT_NE(diag.formatAll().find(expected), std::string::npos)
            << stmt << "\n" << diag.formatAll();
    }
}

TEST(SemaTest, BoolParameterRejected) {
    // A .pred register cannot be a kernel .param; callers pass an int instead.
    std::string src = "kernel test(bool flag) {\n}\n";
    DiagnosticsEngine diag;
    EXPECT_FALSE(checkSource(src, diag));
    EXPECT_NE(diag.formatAll().find("1:13: error: kernel parameter 'flag' cannot have type 'bool'"),
              std::string::npos) << diag.formatAll();
}

TEST(SemaTest, SharedArraysAccepted) {
    std::string src = "kernel test(float* o, int n) {\n"
                      "    shared float a[256];\n"
                      "    shared int b[64];\n"
                      "    a[thread_idx] = 1.0;\n"
                      "    b[thread_idx % 64] = thread_idx;\n"
                      "    barrier();\n"
                      "    o[0] = a[0] + cast<float>(b[1]);\n"
                      "    { shared float c[8]; c[0] = a[1]; }\n"
                      "}\n";
    DiagnosticsEngine diag;
    EXPECT_TRUE(checkSource(src, diag)) << diag.formatAll();
}

TEST(SemaTest, SharedArrayErrors) {
    const std::pair<const char*, const char*> cases[] = {
        {"shared float a[0];", "2:5: error: shared array 'a' must have a positive size"},
        {"shared bool a[4];", "shared arrays must have element type 'int' or 'float' (got 'bool')"},
        {"shared float* a[4];", "shared arrays must have element type 'int' or 'float' (got 'float*')"},
        {"shared float a[4];\n    float x = a;", "shared array 'a' cannot be used as a value"},
        {"shared float a[4];\n    a = 1.0;", "cannot assign to shared array 'a'"},
        {"shared float a[4];\n    float x = a[1.5];", "array subscript is not an integer"},
        {"int a = 1;\n    shared float a[4];", "redefinition of variable 'a'"},
        // 48 KB of static shared memory per block on sm_86 (ptxas: "0xc000 max").
        {"shared float a[8192];\n    shared float b[4097];",
         "error: kernel 'test' uses 49156 bytes of shared memory (static limit is 49152)"},
    };
    for (const auto& [stmts, expected] : cases) {
        std::string src = std::string("kernel test(float* o) {\n    ") + stmts + "\n}\n";
        DiagnosticsEngine diag;
        EXPECT_FALSE(checkSource(src, diag)) << stmts;
        EXPECT_NE(diag.formatAll().find(expected), std::string::npos)
            << stmts << "\n" << diag.formatAll();
    }
}

TEST(SemaTest, DeviceFunctionsAccepted) {
    std::string src =
        "func float square(float x) { return x * x; }\n"
        "func bool in_range(int i, int n) { return i >= 0 && i < n; }\n"
        "func void store(float* out, int i, float v) { out[i] = v; }\n"
        "func int fib(int k) { if (k < 2) { return k; } return fib(k - 1) + fib(k - 2); }\n"
        // is_even calls is_odd before it is defined: signatures are collected first.
        "func bool is_even(int k) { if (k == 0) { return true; } return is_odd(k - 1); }\n"
        "func bool is_odd(int k) { if (k == 0) { return false; } return is_even(k - 1); }\n"
        "func int sign(int v) { if (v < 0) { return -1; } else { return 1; } }\n"
        "func float lane() { return cast<float>(thread_idx); }\n"
        "kernel k(float* x, int n) {\n"
        "    if (!in_range(thread_idx, n)) return;\n"
        "    store(x, thread_idx, square(3));\n" // int argument promoted to float
        "    float f = cast<float>(fib(5)) + cast<float>(sign(-2)) + lane();\n"
        "    if (is_even(4)) { x[0] = f; }\n"
        "}\n";
    DiagnosticsEngine diag;
    EXPECT_TRUE(checkSource(src, diag)) << diag.formatAll();
}

TEST(SemaTest, DeviceFunctionErrors) {
    const std::pair<const char*, const char*> cases[] = {
        {"kernel k() { foo(1); }", "call to undeclared function 'foo'"},
        {"func int f(int a) { return a; }\nkernel k() { int x = f(1, 2); }",
         "function 'f' expects 1 argument(s), got 2"},
        {"func int f(int a) { return a; }\nkernel k() { int x = f(1.5); }",
         "argument 1 of 'f': cannot pass 'float' as 'int'"},
        {"kernel a() { }\nkernel k() { a(); }",
         "cannot call kernel 'a' (kernels are launched from the host)"},
        {"func void f() { }\nkernel k() { int x = f(); }", "void function 'f' used as a value"},
        {"func int f(int a) {\n    if (a > 0) { return 1; }\n}\nkernel k() { }",
         "3:1: error: control reaches end of non-void function 'f'"},
        {"func int f() { return 1.5; }\nkernel k() { }",
         "cannot return 'float' from function 'f' returning 'int'"},
        {"func int f() { return; }\nkernel k() { }", "non-void function 'f' must return a value"},
        {"func void f() { return 1; }\nkernel k() { }", "void function 'f' cannot return a value"},
        {"func int f() { return 1; }\nfunc int f() { return 2; }\nkernel k() { }",
         "2:1: error: redefinition of function 'f'"},
        {"func int k() { return 1; }\nkernel k() { }", "redefinition of function 'k'"},
        {"func void f() { shared float t[4]; }\nkernel k() { }",
         "shared arrays can only be declared in kernels"},
        {"func float* f(float* p) { return p; }\nkernel k() { }",
         "device functions cannot return pointers"},
        {"func void f(void x) { }\nkernel k() { }",
         "parameter 'x' cannot have type 'void'"},
        {"kernel k(void x) { }",
         "kernel parameter 'x' cannot have type 'void'"},
        {"func void f(int a, int a) { }\nkernel k() { }",
         "duplicate parameter name 'a'"},
        {"func void f(int thread_idx) { }\nkernel k() { }",
         "parameter cannot have reserved builtin name 'thread_idx'"},
    };
    for (const auto& [src, expected] : cases) {
        DiagnosticsEngine diag;
        EXPECT_FALSE(checkSource(src, diag)) << src;
        EXPECT_NE(diag.formatAll().find(expected), std::string::npos)
            << src << "\n" << diag.formatAll();
    }
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

TEST(SemaTest, BitwiseOperatorsOnIntsAccepted) {
    std::string src = "kernel test(int a, int b) {\n"
                      "    int c = (a & b) | (a ^ b);\n"
                      "    int d = (a << 2) >> 1;\n"
                      "    int e = ~a;\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_TRUE(ok) << diag.formatAll();
    EXPECT_FALSE(diag.hasErrors());
}

TEST(SemaTest, BitwiseOperatorsOnFloatsRejected) {
    std::string src = "kernel test(float a, int b) {\n"
                      "    int c = a & b;\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(diag.hasErrors());
    EXPECT_NE(diag.formatAll().find("operator '&' requires integer operands"), std::string::npos);
}

TEST(SemaTest, BitwiseNotOnFloatRejected) {
    std::string src = "kernel test(float a) {\n"
                      "    float c = ~a;\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(diag.hasErrors());
    EXPECT_NE(diag.formatAll().find("unary '~' requires integer operand"), std::string::npos);
}

TEST(SemaTest, MultiDimensionalBuiltinsAccepted) {
    std::string src = "kernel test(int* out) {\n"
                      "    int idx = (thread_idx_z * block_dim_y + thread_idx_y) * block_dim_x + thread_idx_x;\n"
                      "    int bidx = (block_idx_z * grid_dim_y + block_idx_y) * grid_dim_x + block_idx_x;\n"
                      "    out[idx] = bidx;\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_TRUE(ok) << diag.formatAll();
    EXPECT_FALSE(diag.hasErrors());
}

TEST(SemaTest, MultiDimensionalBuiltinShadowingRejected) {
    std::string src1 = "kernel test(int thread_idx_y) {}\n";
    DiagnosticsEngine diag1;
    EXPECT_FALSE(checkSource(src1, diag1));
    EXPECT_NE(diag1.formatAll().find("reserved builtin name"), std::string::npos);

    std::string src2 = "kernel test() { int block_idx_z = 0; }\n";
    DiagnosticsEngine diag2;
    EXPECT_FALSE(checkSource(src2, diag2));
    EXPECT_NE(diag2.formatAll().find("reserved builtin name"), std::string::npos);

    std::string src3 = "kernel test() { int grid_dim_x = 0; }\n";
    DiagnosticsEngine diag3;
    EXPECT_FALSE(checkSource(src3, diag3));
    EXPECT_NE(diag3.formatAll().find("reserved builtin name"), std::string::npos);
}
