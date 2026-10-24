# `kernc` — Standalone GPU Kernel Compiler targeting NVIDIA PTX (`sm_86`)

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Target](https://img.shields.io/badge/PTX-sm__86-orange.svg)](https://docs.nvidia.com/cuda/parallel-thread-execution/)

`kernc` is a standalone C++17 compiler for a domain-specific GPU kernel language (`.kern`) targeting NVIDIA PTX (`sm_86`, Ampere architecture). It features a complete compilation pipeline from front-end lexing and recursive-descent parsing with precedence climbing, through semantic analysis, 3-address control-flow graph (CFG) IR lowering, optimization passes (constant folding, algebraic simplification, dead code elimination), to clean PTX code emission.

The compiler also includes an automated test suite featuring **unit tests**, **file-based diagnostic tests**, **`ptxas` assembler validation**, and **hardware end-to-end execution tests** using the CUDA Driver API on an NVIDIA GeForce RTX 3060 Laptop GPU.

---

## Architecture Overview

```
                      ┌────────────────────────┐
                      │    .kern Source File   │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │      Lexer (C++17)     │
                      │  (Zero-copy, Locs)     │
                      └───────────┬────────────┘
                                  │ Token Stream
                                  ▼
                      ┌────────────────────────┐
                      │   Parser (Precedence   │
                      │        Climbing)       │
                      └───────────┬────────────┘
                                  │ AST (unique_ptr)
                                  ▼
                      ┌────────────────────────┐
                      │    Semantic Analysis   │
                      │  (Scoping, Line:Col)   │
                      └───────────┬────────────┘
                                  │ Validated AST
                                  ▼
                      ┌────────────────────────┐
                      │      IR Lowering       │
                      │ (Branch Threading CFG) │
                      └───────────┬────────────┘
                                  │ 3-Address IR
                                  ▼
                      ┌────────────────────────┐
                      │   Optimization Passes  │
                      │  (ConstFold, DCE)      │
                      └───────────┬────────────┘
                                  │ Optimized IR
                                  ▼
                      ┌────────────────────────┐
                      │      PTX Emitter       │
                      │   (sm_86, Hex Floats)  │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │   .ptx Output File     │
                      └───────────┬────────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                               │
          ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│  ptxas sm_86     │                            │ CUDA Driver API  │
│  Validation      │                            │ E2E GPU Runner   │
└──────────────────┘                            └──────────────────┘
```

---

## Directory Structure

```
kernc/
├── CMakeLists.txt                 # Top-level CMake configuration
├── README.md                      # Architecture and documentation
├── docs/                          # Detailed architecture specifications
│   └── specs/
│       ├── 01-scope.md
│       ├── 02-architecture.md
│       ├── 03-ir-and-codegen.md
│       └── 04-testing.md
├── include/kernc/                 # Public C++ headers
│   ├── SourceLocation.h           # Source location & range tracking
│   ├── Diagnostics.h              # Diagnostic engine with caret reporting
│   ├── Token.h                    # Token kinds & representations
│   ├── Lexer.h                    # Zero-copy string_view lexer
│   ├── AST.h                      # AST node hierarchy (unique_ptr)
│   ├── Parser.h                   # Precedence climbing parser
│   ├── Sema.h                     # Semantic analyzer & symbol table
│   ├── IR.h                       # 3-address CFG IR instructions & types
│   ├── IRGen.h                    # AST to IR lowering engine
│   ├── Passes.h                   # Constant folding & DCE passes
│   └── PTXEmitter.h               # PTX code generation targeting sm_86
├── src/                           # Compiler implementation
│   ├── Diagnostics.cpp
│   ├── Lexer.cpp
│   ├── AST.cpp
│   ├── Parser.cpp
│   ├── Sema.cpp
│   ├── IR.cpp
│   ├── IRGen.cpp
│   ├── Passes.cpp
│   ├── PTXEmitter.cpp
│   └── main.cpp                   # kernc CLI compiler driver
├── examples/                      # Example GPU kernels
│   ├── vecadd.kern                # Vector addition (int)
│   ├── saxpy.kern                 # SAXPY (float)
│   ├── clamp.kern                 # Range clamping with branching (float)
│   └── relu.kern                  # ReLU with branching (float)
├── tests/                         # Comprehensive test suite
│   ├── CMakeLists.txt
│   ├── unit/                      # GoogleTest unit tests
│   │   ├── lexer_test.cpp
│   │   ├── parser_test.cpp
│   │   ├── sema_test.cpp
│   │   └── ir_pass_test.cpp
│   ├── diagnostic/                # Negative expected-error tests
│   │   ├── err_undeclared.kern
│   │   ├── err_redeclared.kern
│   │   ├── err_type_mismatch.kern
│   │   ├── err_invalid_index.kern
│   │   ├── err_bad_cast.kern
│   │   ├── err_assign_builtin.kern
│   │   └── run_diagnostic_tests.py
│   └── e2e/                       # Hardware GPU execution tests
│       └── test_kernels.cpp       # CUDA Driver API runner
└── tools/
    └── runner/                    # Standalone CLI PTX execution tool
        └── runner.cpp
```

---

## Key Technical Decisions & Interview Talking Points

### 1. Non-SSA 3-Address IR
- **Rationale**: PTX itself operates on virtual registers (`%r`, `%f`, `%rd`, `%p`) with an infinite register assumption. `ptxas` (NVIDIA's PTX optimizing assembler) performs its own internal SSA construction, dominance frontier analysis, instruction scheduling, and physical register allocation.
- **Benefit**: Retaining a non-SSA 3-address IR in `kernc` eliminates unnecessary dominance-frontier computation and subsequent out-of-SSA phi-elimination passes without sacrificing any downstream performance.

### 2. Branch Threading for Short-Circuit `&&` / `||`
- Rather than materializing boolean integer temporaries (`0` or `1`) with `selp`, conditions with `&&` and `||` use direct **branch threading** (`emitCondition`).
- `a && b` evaluates `a`; if false, it branches directly to `L_false`; if true, it falls through to evaluate `b`. This mirrors Clang/LLVM and GCC lowering, producing fewer register allocations and clean branch structures.

### 3. Address Arithmetic with `mul.wide.s32`
- In PTX 64-bit address mode (`.address_size 64`), memory indexing requires 64-bit effective addresses.
- Array indexing `ptr[i]` multiplies the 32-bit index `%r_i` with the 32-bit element size (4 bytes) using `mul.wide.s32 %rd_off, %r_i, %r_sz;` directly into a 64-bit register `%rd_off`, followed by `add.s64` to the base pointer. This matches the exact instruction sequence generated by `nvcc`.

### 4. Parameter Loading & Global Address Conversion
- Kernel parameters in PTX reside in `.param` space. Pointer arguments cannot be dereferenced directly from parameter space.
- `kernc` loads pointer parameters with `ld.param.u64 %rd_raw, [param_name];` and converts them to generic/global addresses via `cvta.to.global.u64 %rd_base, %rd_raw;` in the entry block prologue.

### 5. IEEE-754 Hexadecimal Float Immediate Encoding
- NVIDIA PTX mandates that 32-bit floating-point literals be encoded as IEEE-754 binary32 hexadecimal words: `0fXXXXXXXX`.
- For example, `0.0f` is emitted as `0f00000000`, `3.14f` is emitted as `0f4048F5C3`. `PTXEmitter` serializes float immediates using bitwise byte transfers (`std::memcpy`) formatted to 8 uppercase hex digits.

### 6. Non-Power-of-Two and Non-Block-Multiple Boundary Verification
- GPU kernels in practice frequently process input arrays whose lengths are not multiples of the thread block size (`blockDim = 256`).
- The test harness validates inputs with $N=1000$, $N=117$, $N=33$, and boundary $N=1$, rigorously exercising thread boundary checks (`if (i < n)`).

---

## Example Kernel & Generated PTX

### Source: `examples/saxpy.kern`
```c
kernel saxpy(float a, float* x, float* y, int n) {
    int i = block_idx * block_dim + thread_idx;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

### Generated PTX (`sm_86`, `.version 7.8`)
```ptx
// Generated by kernc targeting sm_86
.version 7.8
.target sm_86
.address_size 64

.visible .entry saxpy(
    .param .f32 saxpy_param_0,
    .param .u64 saxpy_param_1,
    .param .u64 saxpy_param_2,
    .param .s32 saxpy_param_3
)
{
    .reg .s32   %r<10>;
    .reg .f32   %f<5>;
    .reg .s64   %rd<10>;
    .reg .pred  %p<1>;

entry:
    ld.param.f32 %f0, [saxpy_param_0];
    ld.param.u64 %rd0, [saxpy_param_1];
    cvta.to.global.u64 %rd1, %rd0;
    ld.param.u64 %rd2, [saxpy_param_2];
    cvta.to.global.u64 %rd3, %rd2;
    ld.param.s32 %r0, [saxpy_param_3];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.s32 %r3, %r1, %r2;
    mov.u32 %r4, %tid.x;
    add.s32 %r5, %r3, %r4;
    mov.s32 %r6, %r5;
    setp.lt.s32 %p0, %r6, %r0;
    @%p0 bra L_then_0;
    bra L_if_end_1;
L_then_0:
    mov.s32 %r7, 4;
    mul.wide.s32 %rd4, %r6, %r7;
    add.s64 %rd5, %rd3, %rd4;
    mov.s32 %r8, 4;
    mul.wide.s32 %rd6, %r6, %r8;
    add.s64 %rd7, %rd1, %rd6;
    ld.global.f32 %f1, [%rd7];
    mul.f32 %f2, %f0, %f1;
    mov.s32 %r9, 4;
    mul.wide.s32 %rd8, %r6, %r9;
    add.s64 %rd9, %rd3, %rd8;
    ld.global.f32 %f3, [%rd9];
    add.f32 %f4, %f2, %f3;
    st.global.f32 [%rd5], %f4;
    bra L_if_end_1;
L_if_end_1:
    ret;
}
```

---

## Build & Test Instructions

### Prerequisites (WSL2 Ubuntu 22.04)
- CMake 3.16+
- C++17 compiler (GCC 11+)
- NVIDIA CUDA Toolkit 13.0 (`/usr/local/cuda-13.0/bin/ptxas`)
- NVIDIA Driver Library (`/usr/lib/wsl/lib/libcuda.so`)
- GoogleTest (`libgtest-dev`) & Python 3

### Building
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

### Running All Tests via CTest
```bash
ctest --output-on-failure
```

### Test Suite Summary
```
100% tests passed, 0 tests failed out of 37

  - 23 Unit Tests (Lexer, Parser, Sema, IRPass)       [PASSED]
  -  9 Hardware GPU Execution Tests (CUDA Driver API) [PASSED]
  -  1 Negative Diagnostic Test Suite (6 files)       [PASSED]
  -  4 ptxas sm_86 Acceptance Tests                   [PASSED]
```

### Running the CLI Compiler
```bash
./build/kernc examples/saxpy.kern -o saxpy.ptx
```

### Running the Standalone Runner
```bash
./build/kernc-runner saxpy.ptx saxpy 1024
```
