#include <gtest/gtest.h>
#include "kernc/Lexer.h"
#include "kernc/Parser.h"
#include "kernc/Sema.h"
#include "kernc/IRGen.h"
#include "kernc/Passes.h"
#include "kernc/PTXEmitter.h"

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
