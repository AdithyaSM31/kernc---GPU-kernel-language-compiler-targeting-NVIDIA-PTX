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

TEST(SemaTest, WarpIntrinsicsAccepted) {
    std::string src = "kernel test(int* out_i, float* out_f, int* in_i, float* in_f) {\n"
                      "    int mask = __activemask();\n"
                      "    int mask2 = activemask();\n"
                      "    int val_i = in_i[thread_idx];\n"
                      "    float val_f = in_f[thread_idx];\n"
                      "    \n"
                      "    out_i[0] = __shfl_sync(mask, val_i, 0);\n"
                      "    out_i[1] = shfl_sync(mask, val_i, 0);\n"
                      "    out_f[0] = __shfl_sync(mask, val_f, 0);\n"
                      "    out_f[1] = shfl_sync(mask, val_f, 0);\n"
                      "    \n"
                      "    out_i[2] = __shfl_down_sync(mask, val_i, 16);\n"
                      "    out_i[3] = shfl_down_sync(mask, val_i, 16);\n"
                      "    out_f[2] = __shfl_down_sync(mask, val_f, 16);\n"
                      "    out_f[3] = shfl_down_sync(mask, val_f, 16);\n"
                      "    \n"
                      "    out_i[4] = __shfl_up_sync(mask, val_i, 1);\n"
                      "    out_i[5] = shfl_up_sync(mask, val_i, 1);\n"
                      "    out_f[4] = __shfl_up_sync(mask, val_f, 1);\n"
                      "    out_f[5] = shfl_up_sync(mask, val_f, 1);\n"
                      "    \n"
                      "    out_i[6] = __shfl_xor_sync(mask, val_i, 1);\n"
                      "    out_i[7] = shfl_xor_sync(mask, val_i, 1);\n"
                      "    out_f[6] = __shfl_xor_sync(mask, val_f, 1);\n"
                      "    out_f[7] = shfl_xor_sync(mask, val_f, 1);\n"
                      "    \n"
                      "    out_i[8] = __ballot_sync(mask, thread_idx < 16);\n"
                      "    out_i[9] = ballot_sync(mask, val_i);\n"
                      "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_TRUE(ok) << diag.formatAll();
    EXPECT_FALSE(diag.hasErrors());
}

TEST(SemaTest, WarpIntrinsicsInvalidTypesRejected) {
    struct TestCase {
        std::string src;
        std::string expectedErr;
    };
    std::vector<TestCase> cases = {
        {"kernel k(int* p) { int x = __shfl_sync(p, 1, 0); }", "argument 1 ('mask') of '__shfl_sync' must be 'int', got 'int*'"},
        {"kernel k(float m) { int x = __shfl_sync(m, 1, 0); }", "argument 1 ('mask') of '__shfl_sync' must be 'int', got 'float'"},
        {"kernel k(int* p) { int x = __shfl_down_sync(-1, p, 1); }", "argument 2 ('val') of '__shfl_down_sync' must be 'int' or 'float', got 'int*'"},
        {"kernel k() { int x = __shfl_down_sync(-1, true, 1); }", "argument 2 ('val') of '__shfl_down_sync' must be 'int' or 'float', got 'bool'"},
        {"kernel k() { int x = __shfl_up_sync(-1, 1, 1.5); }", "argument 3 of '__shfl_up_sync' must be 'int', got 'float'"},
        {"kernel k(int* p) { int x = __shfl_xor_sync(-1, 1, p); }", "argument 3 of '__shfl_xor_sync' must be 'int', got 'int*'"},
        {"kernel k() { int x = __ballot_sync(-1, 1.5); }", "argument 2 ('predicate') of '__ballot_sync' must be 'bool' or 'int', got 'float'"},
        {"kernel k(int* p) { int x = __ballot_sync(-1, p); }", "argument 2 ('predicate') of '__ballot_sync' must be 'bool' or 'int', got 'int*'"},
        {"kernel k(int* p) { int x = __ballot_sync(p, 1); }", "argument 1 ('mask') of '__ballot_sync' must be 'int', got 'int*'"},
    };

    for (const auto& tc : cases) {
        DiagnosticsEngine diag;
        EXPECT_FALSE(checkSource(tc.src, diag)) << tc.src;
        EXPECT_NE(diag.formatAll().find(tc.expectedErr), std::string::npos)
            << "Expected: " << tc.expectedErr << "\nGot: " << diag.formatAll();
    }
}

TEST(SemaTest, WarpIntrinsicsArityMismatchesRejected) {
    struct TestCase {
        std::string src;
        std::string expectedErr;
    };
    std::vector<TestCase> cases = {
        {"kernel k() { int x = __activemask(1); }", "intrinsic '__activemask' expects 0 argument(s), got 1"},
        {"kernel k() { int x = activemask(1, 2); }", "intrinsic 'activemask' expects 0 argument(s), got 2"},
        {"kernel k() { int x = __ballot_sync(-1); }", "intrinsic '__ballot_sync' expects 2 argument(s), got 1"},
        {"kernel k() { int x = ballot_sync(-1, 1, 2); }", "intrinsic 'ballot_sync' expects 2 argument(s), got 3"},
        {"kernel k() { int x = __shfl_sync(-1, 1); }", "intrinsic '__shfl_sync' expects 3 argument(s), got 2"},
        {"kernel k() { int x = shfl_down_sync(-1, 1, 2, 3); }", "intrinsic 'shfl_down_sync' expects 3 argument(s), got 4"},
    };

    for (const auto& tc : cases) {
        DiagnosticsEngine diag;
        EXPECT_FALSE(checkSource(tc.src, diag)) << tc.src;
        EXPECT_NE(diag.formatAll().find(tc.expectedErr), std::string::npos)
            << "Expected: " << tc.expectedErr << "\nGot: " << diag.formatAll();
    }
}

TEST(SemaTest, WarpIntrinsicsRedefinitionOrShadowingRejected) {
    std::string src1 = "func int __shfl_sync(int a) { return a; }\nkernel k() {}\n";
    DiagnosticsEngine diag1;
    EXPECT_FALSE(checkSource(src1, diag1));
    EXPECT_NE(diag1.formatAll().find("cannot redefine intrinsic '__shfl_sync' as a function"), std::string::npos);

    std::string src2 = "kernel k(int activemask) {}\n";
    DiagnosticsEngine diag2;
    EXPECT_FALSE(checkSource(src2, diag2));
    EXPECT_NE(diag2.formatAll().find("reserved builtin name"), std::string::npos);

    std::string src3 = "kernel k() { int shfl_down_sync = 1; }\n";
    DiagnosticsEngine diag3;
    EXPECT_FALSE(checkSource(src3, diag3));
    EXPECT_NE(diag3.formatAll().find("reserved builtin name"), std::string::npos);

    std::string src4 = "kernel __shfl_sync() {}\n";
    DiagnosticsEngine diag4;
    EXPECT_FALSE(checkSource(src4, diag4));
    EXPECT_NE(diag4.formatAll().find("cannot redefine intrinsic '__shfl_sync' as a kernel"), std::string::npos);
}

TEST(SemaTest, AtomicOperationsValidSignatures) {
    std::string src =
        "kernel test_atomics(int* p_i, float* p_f) {\n"
        "    shared int s_i[64];\n"
        "    shared float s_f[64];\n"
        "    // Statement form (discard return)\n"
        "    atomic_add(p_i, 1);\n"
        "    __atomic_add(p_i, 1);\n"
        "    atomicAdd(p_i, 1);\n"
        "    __atomicAdd(p_i, 1);\n"
        "    atomic_sub(p_i, 1);\n"
        "    atomicSub(p_i, 1);\n"
        "    atomic_min(p_i, 1);\n"
        "    atomicMin(p_i, 1);\n"
        "    atomic_max(p_i, 1);\n"
        "    atomicMax(p_i, 1);\n"
        "    atomic_and(p_i, 1);\n"
        "    atomicAnd(p_i, 1);\n"
        "    atomic_or(p_i, 1);\n"
        "    atomicOr(p_i, 1);\n"
        "    atomic_xor(p_i, 1);\n"
        "    atomicXor(p_i, 1);\n"
        "    // Expression form (capturing return value)\n"
        "    int old_add = atomic_add(p_i, 2, 5);\n"
        "    int old_sub = atomicSub(p_i, 2, 5);\n"
        "    int old_subscript = atomic_add(p_i[3], 7);\n"
        "    int old_shared = atomic_add(s_i, 10);\n"
        "    int old_shared_idx = atomic_add(s_i, 4, 10);\n"
        "    int old_shared_subscript = atomic_add(s_i[4], 10);\n"
        "    // Float atomics (add and exch)\n"
        "    float old_f_add = atomic_add(p_f, 1.5);\n"
        "    float old_f_add_idx = atomic_add(p_f, 2, 2.5);\n"
        "    float old_f_subscript = atomic_add(p_f[3], 3.5);\n"
        "    float old_sf_add = atomic_add(s_f, 4.5);\n"
        "    float old_sf_add_idx = atomic_add(s_f, 1, 5.5);\n"
        "    float old_sf_subscript = atomic_add(s_f[1], 6.5);\n"
        "    int old_exch_i = atomic_exch(p_i, 42);\n"
        "    int old_exchange_i = atomic_exchange(p_i, 43);\n"
        "    int old_Exchange_i = atomicExchange(p_i, 44);\n"
        "    float old_exch_f = atomicExch(p_f[0], 3.14);\n"
        "    float old_sf_exch = atomic_exch(s_f, 2.71);\n"
        "    // CAS\n"
        "    int old_cas1 = atomic_cas(p_i, 0, 1);\n"
        "    int old_cas2 = atomicCAS(p_i, 2, 0, 1);\n"
        "    int old_cas3 = atomic_cas(p_i[3], 0, 1);\n"
        "    int old_cas4 = atomicCAS(s_i, 0, 1);\n"
        "    int old_cas5 = atomic_cas(s_i, 4, 0, 1);\n"
        "    int old_cas6 = atomicCAS(s_i[4], 0, 1);\n"
        "}\n"
        "func int dev_helper(int* ptr, int v) {\n"
        "    return atomic_add(ptr, v);\n"
        "}\n";
    DiagnosticsEngine diag;
    bool ok = checkSource(src, diag);
    EXPECT_TRUE(ok) << diag.formatAll();
    EXPECT_FALSE(diag.hasErrors());
}

TEST(SemaTest, AtomicOperationsInvalidTypesRejected) {
    struct TestCase {
        std::string src;
        std::string expectedErr;
    };
    std::vector<TestCase> cases = {
        {"kernel k(int x) { atomic_add(x, 1); }", "argument 1 ('ptr') of 'atomic_add' must be a pointer or shared array, got 'int'"},
        {"kernel k(int* p) { atomic_sub(p, 1.5); }", "argument 2 ('val') of 'atomic_sub' must be 'int', got 'float'"},
        {"kernel k(float* p) { atomic_sub(p, 1); }", "intrinsic 'atomic_sub' does not support 'float' memory operand"},
        {"kernel k(float* p) { atomic_min(p, 1); }", "intrinsic 'atomic_min' does not support 'float' memory operand"},
        {"kernel k(float* p) { atomic_max(p, 1); }", "intrinsic 'atomic_max' does not support 'float' memory operand"},
        {"kernel k(float* p) { atomic_and(p, 1); }", "intrinsic 'atomic_and' does not support 'float' memory operand"},
        {"kernel k(float* p) { atomic_or(p, 1); }", "intrinsic 'atomic_or' does not support 'float' memory operand"},
        {"kernel k(float* p) { atomic_xor(p, 1); }", "intrinsic 'atomic_xor' does not support 'float' memory operand"},
        {"kernel k(float* p) { atomic_cas(p, 0, 1); }", "intrinsic 'atomic_cas' does not support 'float' memory operand"},
        {"kernel k(int* p) { atomic_add(p, 1.5, 2); }", "argument 2 ('index') of 'atomic_add' must be 'int', got 'float'"},
        {"kernel k(int* p) { atomic_add(p, true); }", "argument 2 ('val') of 'atomic_add' must be 'int', got 'bool'"},
        {"kernel k(int* p) { atomic_cas(p, 0, 1.5); }", "argument 3 ('val') of 'atomic_cas' must be 'int', got 'float'"},
        {"kernel k(int* p) { atomic_cas(p, 1.5, 0); }", "argument 2 ('cmp') of 'atomic_cas' must be 'int', got 'float'"},
    };

    for (const auto& tc : cases) {
        DiagnosticsEngine diag;
        EXPECT_FALSE(checkSource(tc.src, diag)) << tc.src;
        EXPECT_NE(diag.formatAll().find(tc.expectedErr), std::string::npos)
            << "Expected: " << tc.expectedErr << "\nGot: " << diag.formatAll();
    }
}

TEST(SemaTest, AtomicOperationsArityMismatchesRejected) {
    struct TestCase {
        std::string src;
        std::string expectedErr;
    };
    std::vector<TestCase> cases = {
        {"kernel k(int* p) { atomic_add(p); }", "intrinsic 'atomic_add' expects 2 or 3 argument(s), got 1"},
        {"kernel k(int* p) { atomic_add(p, 1, 2, 3); }", "intrinsic 'atomic_add' expects 2 or 3 argument(s), got 4"},
        {"kernel k(int* p) { atomic_add(p[0], 1, 2); }", "intrinsic 'atomic_add' with subscripted pointer expects 2 argument(s), got 3"},
        {"kernel k(int* p) { atomic_cas(p, 1); }", "intrinsic 'atomic_cas' expects 3 or 4 argument(s), got 2"},
        {"kernel k(int* p) { atomic_cas(p, 1, 2, 3, 4); }", "intrinsic 'atomic_cas' expects 3 or 4 argument(s), got 5"},
        {"kernel k(int* p) { atomic_cas(p[0], 1, 2, 3); }", "intrinsic 'atomic_cas' with subscripted pointer expects 3 argument(s), got 4"},
    };

    for (const auto& tc : cases) {
        DiagnosticsEngine diag;
        EXPECT_FALSE(checkSource(tc.src, diag)) << tc.src;
        EXPECT_NE(diag.formatAll().find(tc.expectedErr), std::string::npos)
            << "Expected: " << tc.expectedErr << "\nGot: " << diag.formatAll();
    }
}

TEST(SemaTest, AtomicOperationsRedefinitionOrShadowingRejected) {
    std::string src1 = "func int atomic_add(int a, int b) { return a + b; }\nkernel k() {}\n";
    DiagnosticsEngine diag1;
    EXPECT_FALSE(checkSource(src1, diag1));
    EXPECT_NE(diag1.formatAll().find("cannot redefine intrinsic 'atomic_add' as a function"), std::string::npos);

    std::string src2 = "kernel atomicAdd(int* p) {}\n";
    DiagnosticsEngine diag2;
    EXPECT_FALSE(checkSource(src2, diag2));
    EXPECT_NE(diag2.formatAll().find("cannot redefine intrinsic 'atomicAdd' as a kernel"), std::string::npos);

    std::string src3 = "kernel k(int atomic_cas) {}\n";
    DiagnosticsEngine diag3;
    EXPECT_FALSE(checkSource(src3, diag3));
    EXPECT_NE(diag3.formatAll().find("reserved builtin name"), std::string::npos);

    std::string src4 = "kernel k() { int atomicCAS = 1; }\n";
    DiagnosticsEngine diag4;
    EXPECT_FALSE(checkSource(src4, diag4));
    EXPECT_NE(diag4.formatAll().find("reserved builtin name"), std::string::npos);
}

TEST(SemaTest, VectorOperationsAccepted) {
    std::string src =
        "kernel test(int4* in_i, float4* in_f, int* raw_i, float* raw_f) {\n"
        "    int2 v2 = int2(1, 2);\n"
        "    int4 v4 = make_int4(1, 2, 3, 4);\n"
        "    float2 f2 = float2(1.0, 2.0);\n"
        "    float4 f4 = make_float4(1, 2, 3, 4);\n" // implicit int -> float promotion in float4
        "    int x = v4.x;\n"
        "    int y = v4.y;\n"
        "    int z = v4.z;\n"
        "    int w = v4.w;\n"
        "    float fx = f4.x;\n"
        "    v4.x = 10;\n"
        "    in_i[0].y = 20;\n"
        "    shared int4 s_i[16];\n"
        "    s_i[thread_idx].z = 30;\n"
        "    int4 loaded = load_int4(raw_i, 0);\n"
        "    store_int4(raw_i, 1, loaded);\n"
        "    float2 f_loaded = load_float2(in_f, 2);\n"
        "    store_float2(raw_f, 3, f_loaded);\n"
        "}\n";
    DiagnosticsEngine diag;
    EXPECT_TRUE(checkSource(src, diag)) << diag.formatAll();
}

TEST(SemaTest, VectorOperationsErrors) {
    struct TestCase {
        std::string src;
        std::string expectedErr;
    };

    TestCase cases[] = {
        {"kernel k() { int2 v = int2(1, 2); int z = v.z; }", "vector member 'z' is out of range for 2-element vector 'int2'"},
        {"kernel k() { float2 v = float2(1.0, 2.0); float w = v.w; }", "vector member 'w' is out of range for 2-element vector 'float2'"},
        {"kernel k() { int4 v = int4(1, 2, 3, 4); int f = v.foo; }", "invalid member 'foo' on vector type 'int4'"},
        {"kernel k() { int a = 5; int x = a.x; }", "member access '.' on non-vector type 'int'"},
        {"kernel k() { int2 v = int2(1); }", "vector constructor 'int2' expects 2 argument(s), got 1"},
        {"kernel k() { int4 v = int4(1, 2); }", "vector constructor 'int4' expects 4 argument(s), got 2"},
        {"kernel k() { int2 v = int2(1.5, 2); }", "argument 1 of 'int2' must be 'int', got 'float'"},
        {"kernel k(float* p) { int4 v = load_int4(p, 0); }", "has incompatible buffer element type 'float*'"},
        {"kernel k(int* p) { store_int4(p, 0, 5); }", "argument 3 ('value') of 'store_int4' must be 'int4', got 'int'"},
    };

    for (const auto& tc : cases) {
        DiagnosticsEngine diag;
        EXPECT_FALSE(checkSource(tc.src, diag)) << tc.src;
        EXPECT_NE(diag.formatAll().find(tc.expectedErr), std::string::npos)
            << "Expected: " << tc.expectedErr << "\nGot: " << diag.formatAll();
    }
}


