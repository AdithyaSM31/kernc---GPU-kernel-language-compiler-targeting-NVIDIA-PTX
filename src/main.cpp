#include "kernc/Lexer.h"
#include "kernc/Parser.h"
#include "kernc/Sema.h"
#include "kernc/IRGen.h"
#include "kernc/Passes.h"
#include "kernc/PTXEmitter.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <input.kern> [options]\n"
              << "Options:\n"
              << "  -o <file>        Output PTX file (defaults to stdout)\n"
              << "  -O0              Disable optimizations\n"
              << "  -O1              Enable optimizations (default)\n"
              << "  -lineinfo        Emit .file/.loc source line info (also --generate-line-info)\n"
              << "  --dump-tokens    Print token stream\n"
              << "  --dump-ir        Print unoptimized IR\n"
              << "  --dump-opt-ir    Print optimized IR\n"
              << "  -h, --help       Show this help message\n";
}

int main(int argc, char** argv) {
    std::string inputFile;
    std::string outputFile;
    bool optimize = true;
    bool dumpTokens = false;
    bool dumpIR = false;
    bool dumpOptIR = false;
    bool lineInfo = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "-O0") {
            optimize = false;
        } else if (arg == "-O1") {
            optimize = true;
        } else if (arg == "-lineinfo" || arg == "--generate-line-info") {
            lineInfo = true;
        } else if (arg == "--dump-tokens") {
            dumpTokens = true;
        } else if (arg == "--dump-ir") {
            dumpIR = true;
        } else if (arg == "--dump-opt-ir") {
            dumpOptIR = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            inputFile = arg;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (inputFile.empty()) {
        std::cerr << "Error: no input file specified\n";
        printUsage(argv[0]);
        return 1;
    }

    std::ifstream in(inputFile);
    if (!in.is_open()) {
        std::cerr << "Error: could not open input file '" << inputFile << "'\n";
        return 1;
    }

    std::string source((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    in.close();

    kernc::DiagnosticsEngine diag(inputFile, source);

    if (dumpTokens) {
        kernc::Lexer lexer(source, &diag);
        while (true) {
            auto tok = lexer.next();
            std::cout << tok.toString() << "\n";
            if (tok.kind == kernc::TokenKind::Eof) break;
        }
        if (diag.hasErrors()) {
            diag.printAll(std::cerr);
            return 1;
        }
        return 0;
    }

    // 1. Lexing & Parsing
    kernc::Lexer lexer(source, &diag);
    kernc::Parser parser(lexer, diag);
    auto program = parser.parseProgram();

    if (diag.hasErrors() || !program) {
        diag.printAll(std::cerr);
        return 1;
    }

    // 2. Semantic Analysis
    kernc::Sema sema(diag);
    if (!sema.checkProgram(*program) || diag.hasErrors()) {
        diag.printAll(std::cerr);
        return 1;
    }

    // 3. IR Generation
    kernc::IRGen irgen(diag);
    auto irModule = irgen.emitModule(*program);

    if (dumpIR) {
        std::cout << "// --- Unoptimized IR ---\n" << irModule.dump();
    }

    // 4. Optimizations
    if (optimize) {
        kernc::Optimizer::runPasses(irModule);
        if (dumpOptIR) {
            std::cout << "// --- Optimized IR ---\n" << irModule.dump();
        }
    }

    // 5. PTX Emission
    kernc::PTXOptions ptxOptions;
    ptxOptions.lineInfo = lineInfo;
    ptxOptions.sourceFile = inputFile;
    kernc::PTXEmitter emitter(ptxOptions);
    std::string ptx = emitter.emitModule(irModule);

    if (!outputFile.empty() && outputFile != "-") {
        std::ofstream out(outputFile);
        if (!out.is_open()) {
            std::cerr << "Error: could not open output file '" << outputFile << "'\n";
            return 1;
        }
        out << ptx;
        out.close();
    } else {
        std::cout << ptx;
    }

    return 0;
}
