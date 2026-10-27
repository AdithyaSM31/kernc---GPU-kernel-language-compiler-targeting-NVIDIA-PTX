# `kernc` Specification 04: Testing & Verification Strategy

## Test Levels

`kernc` employs six comprehensive test suites:

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
  - `reduce.kern`
  - `functions.kern`
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
    - `reduce`: Tree reduction using shared memory and barriers.
    - `functions`: Device functions with recursion, mutual recursion, float/bool/pointer args and returns.
  - Verifies exact or epsilon-bound match against CPU reference implementation.

### 5. Line-Table Tests (`kernc -lineinfo` → ptxas → `nvdisasm --print-line-info`)
- `tests/lineinfo/*.kern` are annotated with trailing `//@ REGEX` comments; the SASS that
  ptxas attributes to that line must include an instruction matching REGEX (e.g. `//@ STG`
  on a global store, `//@ BAR` on `barrier()`).
- `tests/lineinfo/check_lineinfo.py` also requires that no SASS is attributed to a blank or
  comment-only line, and that the line table names the `.kern` file.
- Run at `-O0` and `-O1`. Unit tests cover the per-instruction locations, the `.file`/`.loc`
  text and file-name escaping; a GPU test JIT-loads line-info PTX with
  `CU_JIT_GENERATE_LINE_INFO` and checks the results.

### 6. Standalone Runner CLI Tests (`kernc-runner`)
- `tests/runner/test_runner.py` validates the standalone CUDA Driver API execution utility `kernc-runner` (`tools/runner/runner.cpp`):
  - `--help` and usage formatting detailing `--arg`, `--grid`, `--block`, and `--dump`.
  - Kernel execution on NVIDIA GeForce RTX 3060 Laptop GPU through CUDA Driver API (`cuLaunchKernel`, `cuCtxSynchronize`).
  - Typed argument parsing for 32-bit scalars (`i32:<val>`, `f32:<val>`) and device memory buffers (`buf:<type>:<N>:<init>` with `zeros`, `rand`, `seq`).
  - Output inspection via `--dump <arg_index>[:<count>]` verifying element-by-element correctness against mathematical expectations.
  - Driver JIT compilation logging (`CU_JIT_INFO_LOG_BUFFER`, `CU_JIT_ERROR_LOG_BUFFER`) under `--verbose`.
  - Negative test suite verifying robust error handling and resource cleanup on missing files, unknown kernel symbols, argument count mismatches, and invalid dump specifications.
  - Argument type checking: each `--arg` must match its PTX `.param` type (`.f32` → `f32:`, `.s32`/`.u32`/`.b32` → `i32:`, `.u64`/`.b64` pointer → `buf:`). Mismatches must be rejected before launch with the argument number and expected type (stage 21).

