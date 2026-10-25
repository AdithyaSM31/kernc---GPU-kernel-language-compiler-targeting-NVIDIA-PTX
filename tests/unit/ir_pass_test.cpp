#include <gtest/gtest.h>
#include "kernc/Lexer.h"
#include "kernc/Parser.h"
#include "kernc/Sema.h"
#include "kernc/IRGen.h"
#include "kernc/Passes.h"
#include "kernc/PTXEmitter.h"
#include <cctype>
#include <cstdint>
#include <limits>
#include <sstream>
#include <unordered_map>

using namespace kernc;

static IRFunction compileToIR(const std::string& src) {
    DiagnosticsEngine diag;
    Lexer lexer(src, &diag);
    Parser parser(lexer, diag);
    auto program = parser.parseProgram();
    EXPECT_TRUE(program != nullptr) << diag.formatAll();
    if (!program) return IRFunction{};
    Sema sema(diag);
    bool ok = sema.checkProgram(*program);
    EXPECT_TRUE(ok) << diag.formatAll();
    if (!ok) return IRFunction{}; // IRGen assumes a well-typed AST
    IRGen irgen(diag);
    return irgen.emitKernel(*program->kernels[0]);
}

TEST(IRPassTest, ShortCircuitBranchThreading) {
    std::string src = "kernel test(int a, int b) {\n"
                      "    if (a < 10 && b > 20) {\n"
                      "        int c = 1;\n"
                      "    }\n"
                      "}\n";
    auto fn = compileToIR(src);
    // Should have multiple blocks due to short-circuit branch threading (L_and_rhs)
    EXPECT_GT(fn.blocks.size(), 2);

    bool foundBraCond = false;
    for (const auto& bb : fn.blocks) {
        for (const auto& inst : bb.instructions) {
            if (inst.op == IROp::BraCond) {
                foundBraCond = true;
            }
        }
    }
    EXPECT_TRUE(foundBraCond);
}

TEST(IRPassTest, ConstantFolding) {
    std::string src = "kernel test() {\n"
                      "    int x = 3 + 4 * 2;\n"
                      "}\n";
    auto fn = compileToIR(src);
    Optimizer::runConstantFolding(fn);

    // After constant folding, 4 * 2 = 8, 3 + 8 = 11
    bool foundFoldedConst = false;
    for (const auto& bb : fn.blocks) {
        for (const auto& inst : bb.instructions) {
            if (inst.op == IROp::LoadImm && inst.immInt == 11) {
                foundFoldedConst = true;
            }
        }
    }
    EXPECT_TRUE(foundFoldedConst);
}

TEST(IRPassTest, DeadCodeElimination) {
    std::string src = "kernel test() {\n"
                      "    int unused = 100 + 200;\n"
                      "    int also_unused = 50;\n"
                      "}\n";
    auto fn = compileToIR(src);
    Optimizer::runPasses(fn);

    // After full optimization (CF + DCE), unused computations should be removed
    // Only entry and ret should remain in blocks
    int nonRetInstCount = 0;
    for (const auto& bb : fn.blocks) {
        for (const auto& inst : bb.instructions) {
            if (inst.op != IROp::Ret) {
                nonRetInstCount++;
            }
        }
    }
    EXPECT_EQ(nonRetInstCount, 0);
}

// Returns the comma-separated operands of the first emitted instruction with this opcode.
static std::vector<std::string> operandsOf(const std::string& ptx, const std::string& opcode) {
    std::vector<std::string> ops;
    size_t pos = ptx.find("    " + opcode + " ");
    if (pos == std::string::npos) return ops;
    size_t start = pos + 4 + opcode.size() + 1;
    size_t end = ptx.find(';', start);
    std::stringstream ss(ptx.substr(start, end - start));
    std::string op;
    while (std::getline(ss, op, ',')) {
        op.erase(0, op.find_first_not_of(' '));
        ops.push_back(op);
    }
    return ops;
}

static bool isRegClass(const std::string& reg, const std::string& prefix) {
    // "%r" must not match "%rd", so require a digit right after the prefix.
    return reg.size() > prefix.size() && reg.compare(0, prefix.size(), prefix) == 0 &&
           std::isdigit(static_cast<unsigned char>(reg[prefix.size()]));
}

TEST(IRPassTest, UninitializedIntComparedKeepsS32RegisterClass) {
    // A variable whose first appearance is as a setp source must still be a %r register.
    std::string src = "kernel t(int* out, int n) {\n"
                      "    int j;\n"
                      "    if (j < n) { out[0] = 1; }\n"
                      "}\n";
    auto fn = compileToIR(src);
    std::string ptx = PTXEmitter().emitFunction(fn);

    auto ops = operandsOf(ptx, "setp.lt.s32");
    ASSERT_EQ(ops.size(), 3u) << ptx;
    EXPECT_TRUE(isRegClass(ops[0], "%p")) << ptx;
    EXPECT_TRUE(isRegClass(ops[1], "%r")) << ptx;
    EXPECT_TRUE(isRegClass(ops[2], "%r")) << ptx;
}

TEST(IRPassTest, UninitializedIntIndexKeepsS32RegisterClass) {
    // mul.wide.s32 takes 32-bit sources and produces a 64-bit result.
    std::string src = "kernel t(int* out) {\n"
                      "    int j;\n"
                      "    out[j] = 1;\n"
                      "}\n";
    auto fn = compileToIR(src);
    std::string ptx = PTXEmitter().emitFunction(fn);

    auto ops = operandsOf(ptx, "mul.wide.s32");
    ASSERT_EQ(ops.size(), 3u) << ptx;
    EXPECT_TRUE(isRegClass(ops[0], "%rd")) << ptx;
    EXPECT_TRUE(isRegClass(ops[1], "%r")) << ptx;
    EXPECT_TRUE(isRegClass(ops[2], "%r")) << ptx;
}

// Optimizes the kernel, then returns the constant stored by each st.global, in program order.
// Folding must have reduced every stored value to an immediate for this to be meaningful.
static std::vector<int32_t> storedConstants(const std::string& src) {
    auto fn = compileToIR(src);
    Optimizer::runPasses(fn);
    std::unordered_map<VReg, int32_t> imms;
    std::vector<int32_t> stored;
    for (const auto& bb : fn.blocks) {
        for (const auto& inst : bb.instructions) {
            if (inst.op == IROp::LoadImm) imms[inst.dst] = inst.immInt;
            if (inst.op == IROp::Store) {
                EXPECT_TRUE(imms.count(inst.src2)) << "stored value was not folded:\n" << fn.dump();
                stored.push_back(imms[inst.src2]);
            }
        }
    }
    return stored;
}

static int countOps(const IRFunction& fn, IROp op) {
    int n = 0;
    for (const auto& bb : fn.blocks)
        for (const auto& inst : bb.instructions)
            if (inst.op == op) ++n;
    return n;
}

TEST(IRPassTest, FoldFloatToIntSaturatesLikeCvtRzi) {
    // cvt.rzi.s32.f32 truncates toward zero and clamps out-of-range values.
    auto v = storedConstants("kernel t(int* o) {\n"
                             "    o[0] = cast<int>(1e20);\n"
                             "    o[1] = cast<int>(-1e20);\n"
                             "    o[2] = cast<int>(2.9);\n"
                             "    o[3] = cast<int>(-2.9);\n"
                             "}\n");
    ASSERT_EQ(v.size(), 4u);
    EXPECT_EQ(v[0], INT32_MAX);
    EXPECT_EQ(v[1], INT32_MIN);
    EXPECT_EQ(v[2], 2);
    EXPECT_EQ(v[3], -2);
}

TEST(IRPassTest, FoldFloatToIntNaNIsZero) {
    // No NaN literal exists in the language, so build the IR directly.
    IRFunction fn;
    VReg f = fn.freshReg(IRType::F32);
    VReg i = fn.freshReg(IRType::S32);
    IRInst nan{IROp::LoadImm, f, 0, 0, IRType::F32};
    nan.immFloat = std::numeric_limits<float>::quiet_NaN();
    fn.emit(nan);
    fn.emit(IRInst{IROp::CvtF2I, i, f, 0, IRType::S32});

    Optimizer::runConstantFolding(fn);
    const IRInst& cvt = fn.blocks[0].instructions[1];
    ASSERT_EQ(cvt.op, IROp::LoadImm);
    EXPECT_EQ(cvt.immInt, 0);
}

TEST(IRPassTest, FoldSignedArithmeticWrapsLikeS32) {
    // add/sub/mul.lo/neg .s32 wrap modulo 2^32 on the GPU. Folding with C++ signed
    // arithmetic is undefined behaviour in the compiler (caught by KERNC_SANITIZE).
    auto v = storedConstants("kernel t(int* o) {\n"
                             "    o[0] = 2147483647 + 1;\n"
                             "    o[1] = (-2147483647 - 1) - 1;\n"
                             "    o[2] = 65536 * 65536;\n"
                             "    o[3] = -(-2147483647 - 1);\n"
                             "}\n");
    ASSERT_EQ(v.size(), 4u);
    EXPECT_EQ(v[0], INT32_MIN);
    EXPECT_EQ(v[1], INT32_MAX);
    EXPECT_EQ(v[2], 0);
    EXPECT_EQ(v[3], INT32_MIN);
}

TEST(IRPassTest, FoldLeavesIntMinDivideByMinusOneToHardware) {
    // INT_MIN / -1 overflows; div.s32/rem.s32 leave the result unspecified, so the
    // folder must not invent one.
    auto fn = compileToIR("kernel t(int* o) {\n"
                          "    o[0] = (-2147483647 - 1) / -1;\n"
                          "    o[1] = (-2147483647 - 1) % -1;\n"
                          "}\n");
    Optimizer::runPasses(fn);
    EXPECT_EQ(countOps(fn, IROp::Div), 1) << fn.dump();
    EXPECT_EQ(countOps(fn, IROp::Mod), 1) << fn.dump();
}

TEST(IRPassTest, EdgeIntLiteralsLowerToTwosComplementBits) {
    auto v = storedConstants("kernel t(int* o) {\n"
                             "    o[0] = -2147483648;\n"
                             "    o[1] = 0xFFFFFFFF;\n"
                             "    o[2] = 0x80000000;\n"
                             "}\n");
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], INT32_MIN);
    EXPECT_EQ(v[1], -1);
    EXPECT_EQ(v[2], INT32_MIN);
}

static bool isTerminator(IROp op) {
    return op == IROp::Bra || op == IROp::BraCond || op == IROp::Ret;
}

TEST(IRPassTest, ReturnTerminatesItsBasicBlock) {
    // CFG invariant: a terminator may only appear as the last instruction of a block, even
    // when source code follows the return.
    auto fn = compileToIR("kernel t(int* o, int n) {\n"
                          "    if (thread_idx >= n) return;\n"
                          "    o[0] = 1;\n"
                          "    return;\n"
                          "    o[1] = 2;\n"
                          "}\n");
    int rets = 0;
    for (const auto& bb : fn.blocks) {
        for (size_t k = 0; k < bb.instructions.size(); ++k) {
            if (bb.instructions[k].op == IROp::Ret) ++rets;
            if (isTerminator(bb.instructions[k].op)) {
                EXPECT_EQ(k + 1, bb.instructions.size())
                    << "terminator in the middle of block " << bb.label << ":\n" << fn.dump();
            }
        }
    }
    EXPECT_GE(rets, 3) << fn.dump(); // two explicit returns plus the implicit one at the end
}

TEST(IRPassTest, ComparisonAsValueLowersToSetpIntoPredicate) {
    // Previously a comparison in value context fell through to the default case and
    // emitted an add. It must setp straight into a .pred register, and a bool used as a
    // condition must branch on that register without re-testing it against zero.
    auto fn = compileToIR("kernel t(int* o, int n) {\n"
                          "    bool b = n < 10;\n"
                          "    if (b) { o[0] = 1; }\n"
                          "}\n");
    EXPECT_EQ(countOps(fn, IROp::Add), 0) << fn.dump();
    EXPECT_EQ(countOps(fn, IROp::SetpNe), 0) << fn.dump();
    bool setpIntoPred = false;
    for (const auto& bb : fn.blocks)
        for (const auto& inst : bb.instructions)
            if (inst.op == IROp::SetpLt && fn.regType(inst.dst) == IRType::Pred) setpIntoPred = true;
    EXPECT_TRUE(setpIntoPred) << fn.dump();
}

TEST(IRPassTest, ShortCircuitValueGuardsLoad) {
    // `ok = i < n && a[i] > 0` must not load a[i] unless i < n, so no load may appear before
    // the first branch (i.e. in the entry block).
    auto fn = compileToIR("kernel t(int* a, int n) {\n"
                          "    bool ok = thread_idx < n && a[thread_idx] > 0;\n"
                          "    if (ok) { a[0] = 1; }\n"
                          "}\n");
    ASSERT_FALSE(fn.blocks.empty());
    for (const auto& inst : fn.blocks[0].instructions) {
        EXPECT_NE(inst.op, IROp::Load) << "load before the short-circuit branch:\n" << fn.dump();
    }
    EXPECT_GE(countOps(fn, IROp::Load), 1) << fn.dump();
}

TEST(IRPassTest, PTXEmissionValidStructure) {
    std::string src = "kernel vecadd(int* a, int* b, int* c, int n) {\n"
                      "    int i = block_idx * block_dim + thread_idx;\n"
                      "    if (i < n) {\n"
                      "        c[i] = a[i] + b[i];\n"
                      "    }\n"
                      "}\n";
    auto fn = compileToIR(src);
    Optimizer::runPasses(fn);
    PTXEmitter emitter;
    std::string ptx = emitter.emitFunction(fn);

    EXPECT_NE(ptx.find(".visible .entry vecadd"), std::string::npos);
    EXPECT_NE(ptx.find("ld.param.u64"), std::string::npos);
    EXPECT_NE(ptx.find("cvta.to.global.u64"), std::string::npos);
    EXPECT_NE(ptx.find("mul.wide.s32"), std::string::npos);
    EXPECT_NE(ptx.find("st.global.s32"), std::string::npos);
    EXPECT_NE(ptx.find("ret;"), std::string::npos);
}
