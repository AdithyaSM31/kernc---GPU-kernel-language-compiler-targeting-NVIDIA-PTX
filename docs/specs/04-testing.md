# `kernc` Specification 04: Testing & Verification Strategy

## Test Levels

`kernc` employs four comprehensive test suites:

### 1. Unit Tests (GoogleTest)
- `tests/unit/lexer_test.cpp`: Keywords, numeric formats (int/hex/float), source locations, comments.
- `tests/unit/parser_test.cpp`: Precedence climbing, associativity, casts, unary, control flow AST.
- `tests/unit/sema_test.cpp`: Type checker, scope stack, promotion, diagnostic error triggers.
- `tests/unit/ir_pass_test.cpp`: Branch threading, constant folding, dead code elimination, PTX emission.

### 2. Negative Diagnostic Tests
- Located in `tests/diagnostic/err_*.kern`.
- Verified automatically by `tests/diagnostic/run_diagnostic_tests.py`.
- Checks that erroneous programs fail compilation with precise line, column, and diagnostic messages.

### 3. PTX Acceptance Tests (`ptxas -arch=sm_86`)
- `tests/CMakeLists.txt` registers acceptance tests for example kernels:
  - `vecadd.kern`
  - `saxpy.kern`
  - `clamp.kern`
  - `relu.kern`
- Each kernel is compiled to PTX and assembled by `/usr/local/cuda-13.0/bin/ptxas -arch=sm_86`.
- Ensures zero PTX syntax or semantic errors.

### 4. End-to-End GPU Tests (CUDA Driver API)
- `tests/e2e/test_kernels.cpp` tests real GPU execution on NVIDIA GeForce RTX 3060 Laptop GPU:
  - Sizes:
    - Power of two (e.g., $N=1024$, $N=2048$).
    - Non-block multiples (e.g., $N=1000$, $N=117$, $N=33$).
    - Boundary ($N=1$).
  - Kernels:
    - `vecadd`: Vector addition.
    - `saxpy`: Scaled vector addition with float constant.
    - `clamp`: Branching clamping within $[min, max]$.
    - `relu`: Branching ReLU $\max(0, x)$.
  - Verifies exact or epsilon-bound match against CPU reference implementation.
