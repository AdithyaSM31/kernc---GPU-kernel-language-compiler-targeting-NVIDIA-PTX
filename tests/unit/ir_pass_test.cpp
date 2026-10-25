#include <gtest/gtest.h>
#include "kernc/Lexer.h"
#include "kernc/Parser.h"
#include "kernc/Sema.h"
#include "kernc/IRGen.h"
#include "kernc/Passes.h"
#include "kernc/PTXEmitter.h"
#include <cctype>
#include <sstream>

using namespace kernc;

static IRFunction compileToIR(const std::string& src) {
    DiagnosticsEngine diag;
    Lexer lexer(src, &diag);
    Parser parser(lexer, diag);
    auto program = parser.parseProgram();
    EXPECT_TRUE(program != nullptr);
    Sema sema(diag);
    EXPECT_TRUE(sema.checkProgram(*program));
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
