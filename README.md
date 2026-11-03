# kernc: a GPU kernel compiler targeting NVIDIA PTX (sm_86)

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Target](https://img.shields.io/badge/PTX-sm__86-orange.svg)](https://docs.nvidia.com/cuda/parallel-thread-execution/)
[![Tests](https://img.shields.io/badge/tests-205%20passing-brightgreen.svg)](#testing)

kernc compiles a small C-like GPU kernel language (`.kern` files) to PTX, the virtual ISA that
NVIDIA's `ptxas` assembles into machine code (SASS). It is written in C++17 from scratch: a
hand-written lexer and recursive-descent parser, a type checker, a three-address IR with its own
optimizer, and a PTX emitter. There is no LLVM.

Every kernel it emits has to pass `ptxas -arch=sm_86`, and the test suite then runs that PTX on a
real GPU (an RTX 3060 Laptop GPU) through the CUDA Driver API and checks every result against a
CPU reference.

```c
kernel saxpy(float a, float* x, float* y, int n) {
    int i = block_idx * block_dim + thread_idx;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

```bash
kernc saxpy.kern -o saxpy.ptx        # .kern -> PTX
kernc-runner saxpy.ptx saxpy ...     # PTX -> GPU, through the CUDA Driver API
```

## Highlights

- **Matches nvcc on the SASS it produces** for the shipped kernels, measured from the cubin:
  SAXPY 16 instructions in both; the shared-memory reduction 37 instructions with 8 registers
  (nvcc: 37 and 10); device-function inlining 145 instructions vs nvcc's 143, with the same
  six calls left out of line. [Details](#performance-compared-with-nvcc).
- **An optimizer that does what ptxas won't.** ptxas does not inline `.func` calls and does
  not strength-reduce `div.s32` by a constant, so kernc does both: inlining, magic-number
  division, CSE, constant folding with PTX semantics, CFG cleanup, loop unrolling and list
  scheduling.
- **Works with NVIDIA's tools.** `-lineinfo` emits `.file`/`.loc`, so `nvdisasm` maps SASS
  back to `.kern` lines and `compute-sanitizer` reports an out-of-bounds write as
  `oob.kern:5`. racecheck finds the reduction race-free, and catches the race when the
  barriers are removed.
- **Front-end diagnostics** with `file:line:col` and a caret: type errors, literal range,
  call arity and types, plus warnings for possibly-uninitialized variables and `barrier()`
  inside divergent control flow.
- **205 tests** (unit, GPU end-to-end, ptxas acceptance, line tables, diagnostics,
  compute-sanitizer), all passing in a normal build and under UndefinedBehaviorSanitizer.
  Writing them found 16 real bugs, listed [below](#bugs-the-tests-found).

## Contents

- [Quick start](#quick-start)
- [A walkthrough: SAXPY from source to SASS](#a-walkthrough-saxpy-from-source-to-sass)
- [The language](#the-language)
- [Command-line tools](#command-line-tools)
- [Architecture](#architecture)
- [Performance compared with nvcc](#performance-compared-with-nvcc)
- [Debugging with NVIDIA's tools](#debugging-with-nvidias-tools)
- [Design decisions](#design-decisions)
- [Testing](#testing)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)

---

## Quick start

### Requirements

kernc is developed on WSL2 (Ubuntu 22.04) with:

| Tool | Version used | Needed for |
|---|---|---|
| CMake | 3.16 or newer | building |
| GCC | 11 (any C++17 compiler) | building |
| CUDA toolkit | 13.0, at `/usr/local/cuda-13.0` | `ptxas`, `nvdisasm`, `cuobjdump`, `compute-sanitizer`, and `cuda.h` for the GPU tests and runner |
| GoogleTest | 1.11 (`libgtest-dev`) | unit and GPU tests |
| Python 3 | 3.10 | the script-driven tests |
| NVIDIA GPU | sm_86 (Ampere) | running kernels |

The compiler itself needs only a C++17 compiler and CMake. Without CUDA, the GPU tests and
`kernc-runner` are left out of the build, and the ptxas tests are skipped.

On WSL2 the GPU driver comes from Windows (`/usr/lib/wsl/lib/libcuda.so`). Do not install a
Linux NVIDIA driver inside WSL.

### Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To build with UndefinedBehaviorSanitizer, which turns undefined behaviour inside the compiler
(such as signed overflow while constant folding) into a test failure:

```bash
cmake -S . -B build-ubsan -DKERNC_SANITIZE=ON
cmake --build build-ubsan -j
ctest --test-dir build-ubsan
```

### Compile a kernel and run it on the GPU

```bash
./build/kernc examples/saxpy.kern -o saxpy.ptx
/usr/local/cuda-13.0/bin/ptxas -arch=sm_86 saxpy.ptx -o saxpy.cubin   # optional: check it assembles

./build/kernc-runner saxpy.ptx saxpy --grid 4 --block 256 \
    --arg f32:2.0 --arg buf:f32:1000:seq --arg buf:f32:1000:seq --arg i32:1000 \
    --dump 2:8
```

```
[Runner] Loaded GPU: NVIDIA GeForce RTX 3060 Laptop GPU
[Runner] Found kernel: saxpy
[Runner] Launching kernel: grid=4 block=256 args=4
[Runner] Kernel execution completed successfully.
[Runner] Dump arg 2 (f32, count=8):
  [0] = 0
  [1] = 3
  [2] = 6
  [3] = 9
  [4] = 12
  [5] = 15
  [6] = 18
  [7] = 21
[Runner] Success!
```

`x` and `y` both start as `0, 1, 2, ...`, so `y[i] = 2·i + i = 3i`.

---

## A walkthrough: SAXPY from source to SASS

This follows [`examples/saxpy.kern`](examples/saxpy.kern) through each stage.

**1. IR before optimization** (`kernc examples/saxpy.kern -O0 --dump-ir`). IRGen translates each
statement on its own, so every `x[i]` / `y[i]` recomputes its own address, three times in all:

```
$L_then_0:
    ldimm %v14:s32 = 4
    mul.wide %v15:s64 = %v7, %v14
    add.s64 %v16:s64 = %v5, %v15          // &y[i], for the store
    ldimm %v17:s32 = 4
    mul.wide %v18:s64 = %v7, %v17
    add.s64 %v19:s64 = %v3, %v18          // &x[i]
    load %v20:f32 = global [%v19]
    fmul %v21:f32 = %v1, %v20
    ldimm %v22:s32 = 4
    mul.wide %v23:s64 = %v7, %v22
    add.s64 %v24:s64 = %v5, %v23          // &y[i] again, for the load
    load %v25:f32 = global [%v24]
    fadd %v26:f32 = %v21, %v25
    store global [%v16], %v26
    bra $L_if_end_1
```

**2. IR after optimization** (`--dump-opt-ir`). CSE keeps one `mul.wide` and one `&y[i]`, CFG
cleanup drops the `bra` to the next block, and the scheduler starts both loads before the
arithmetic that needs them:

```
$L_then_0:
    ldimm %v14:s32 = 4
    mul.wide %v15:s64 = %v7, %v14
    add.s64 %v19:s64 = %v3, %v15
    add.s64 %v16:s64 = %v5, %v15
    load %v20:f32 = global [%v19]
    load %v25:f32 = global [%v16]
    fmul %v21:f32 = %v1, %v20
    fadd %v26:f32 = %v21, %v25
    store global [%v16], %v26
```

**3. PTX** (`kernc examples/saxpy.kern`; the `.version` / `.target` / `.address_size` header is
omitted). Registers are declared per type class. Kernel pointers arrive as generic addresses and
are converted once with `cvta.to.global`. Names that kernc generates contain `$`, which a
`.kern` identifier cannot, so they can never collide with user names:

```ptx
.visible .entry saxpy(
    .param .f32 saxpy$param_0,
    .param .u64 saxpy$param_1,
    .param .u64 saxpy$param_2,
    .param .s32 saxpy$param_3
)
{
    .reg .s32   %r<8>;
    .reg .f32   %f<5>;
    .reg .s64   %rd<7>;
    .reg .pred  %p<1>;

$entry:
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    ld.param.u64 %rd0, [saxpy$param_1];
    ld.param.u64 %rd1, [saxpy$param_2];
    mov.u32 %r2, %tid.x;
    ld.param.s32 %r3, [saxpy$param_3];
    mul.lo.s32 %r4, %r0, %r1;
    ld.param.f32 %f0, [saxpy$param_0];
    cvta.to.global.u64 %rd2, %rd0;
    cvta.to.global.u64 %rd3, %rd1;
    add.s32 %r5, %r4, %r2;
    mov.s32 %r6, %r5;
    setp.lt.s32 %p0, %r5, %r3;
    @%p0 bra $L_then_0;
    bra $L_if_end_1;
$L_then_0:
    mov.s32 %r7, 4;
    mul.wide.s32 %rd4, %r6, %r7;
    add.s64 %rd5, %rd2, %rd4;
    add.s64 %rd6, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    mul.f32 %f3, %f0, %f1;
    add.f32 %f4, %f3, %f2;
    st.global.f32 [%rd6], %f4;
$L_if_end_1:
    ret;
}
```

**4. SASS** (`kernc -lineinfo`, then `ptxas -arch=sm_86`, then `nvdisasm --print-line-info`,
with NOPs and encodings trimmed). ptxas fuses `mul.f32` + `add.f32` into `FFMA`, turns the
bounds check into a predicated `EXIT`, and keeps each instruction attributed to its `.kern` line:

```
	//## File "examples/saxpy.kern", line 2
        MOV R1, c[0x0][0x28]
        S2R R4, SR_CTAID.X
        S2R R3, SR_TID.X
        IMAD R4, R4, c[0x0][0x0], R3
	//## File "examples/saxpy.kern", line 3
        ISETP.GE.AND P0, PT, R4, c[0x0][0x178], PT
    @P0 EXIT
	//## File "examples/saxpy.kern", line 4
        MOV R5, 0x4
        ULDC.64 UR4, c[0x0][0x118]
        IMAD.WIDE R2, R4, R5, c[0x0][0x168]
        IMAD.WIDE R4, R4, R5, c[0x0][0x170]
        LDG.E R2, [R2.64]
        LDG.E R7, [R4.64]
        FFMA R7, R2, c[0x0][0x160], R7
        STG.E [R4.64], R7
	//## File "examples/saxpy.kern", line 6
        EXIT
```

---

## The language

The full reference is [docs/specs/01-scope.md](docs/specs/01-scope.md). This section covers
what you need to write kernels.

### Types

| Type | PTX | Notes |
|---|---|---|
| `int` | `.s32` | 32-bit signed, wraps on overflow like `.s32` arithmetic |
| `float` | `.f32` | IEEE-754 single precision |
| `bool` | `.pred` | `true` / `false`; no implicit conversion to or from numbers |
| `int2`, `int4`, `float2`, `float4` | `.v2` / `.v4` | components `.x .y .z .w`; 8- or 16-byte aligned |
| `int*`, `float*`, `int2*`, `int4*`, `float2*`, `float4*` | `.u64` | global-memory pointers; indexing scales by the element size |
| `shared T name[N];` | `.shared` | per-block array, `T` scalar or vector, `N` a literal |

### Execution geometry

| kernc | CUDA equivalent | PTX |
|---|---|---|
| `thread_idx`, `thread_idx_x`, `thread_idx.x` | `threadIdx.x` | `%tid.x` |
| `block_idx`, `block_idx_y`, `block_idx.z` | `blockIdx.*` | `%ctaid.*` |
| `block_dim`, `block_dim_x`, ... | `blockDim.*` | `%ntid.*` |
| `grid_dim`, `grid_dim_x`, ... | `gridDim.*` | `%nctaid.*` |

The CUDA spellings (`threadIdx.x`, `blockIdx.y`, ...) are accepted too. All of these are
`int` and are reserved names.

### Statements and operators

- Declarations (`int x = 0;`, `float y;`), assignment, `if`/`else`, `while`, `for`,
  `return;`, `barrier();` (`bar.sync 0`), and function calls.
- Arithmetic `+ - * / %`, bitwise `& | ^ ~ << >>`, comparisons `< <= > >= == !=`,
  short-circuit `&& ||` and `!`, with ISO C precedence.
- `cast<int>(x)`, `cast<float>(x)`, `cast<bool>(x)` are the only explicit conversions; the only
  implicit one is `int` to `float`.
- Literals: decimal and hex integers, and floats (`3.14`, `1.0f`, `.5`, `1e-3`).

### Device functions

```c
func float square(float x) { return x * x; }

// bool crosses the call as a 32-bit 0/1, because .param has no predicate type.
func bool in_range(int i, int n) { return i >= 0 && i < n; }

// Recursion and mutual recursion go through the real ABI (ptxas gives fib a stack frame).
func int fib(int k) {
    if (k < 2) { return k; }
    return fib(k - 1) + fib(k - 2);
}

kernel use_functions(float* x, float* y, int* z, int n) {
    int i = block_idx * block_dim + thread_idx;
    if (!in_range(i, n)) return;
    y[i] = square(x[i]);
    z[i] = fib(i % 15);
}
```

`func` lowers to a PTX `.func`. Functions can be called before they are defined; kernc emits a
prototype for every `.func` at the top of the module. Parameters are `int`, `float`, `bool` or
pointers, and the return type is `void`, `int`, `float` or `bool`. See
[`examples/functions.kern`](examples/functions.kern).

### Shared memory and barriers

```c
kernel reduce_sum(float* in, float* partial, int n) {
    shared float sdata[256];
    int tid = thread_idx;
    int i = block_idx * block_dim + thread_idx;
    float v = 0.0;
    if (i < n) { v = in[i]; }          // no early return: every thread must reach barrier()
    sdata[tid] = v;
    barrier();

    for (int s = block_dim / 2; s > 0; s = s / 2) {   // uniform across the block
        if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
        barrier();
    }
    if (tid == 0) { partial[block_idx] = sdata[0]; }
}
```

Shared arrays can only be declared in kernels, at most 48 KB per kernel (the sm_86 static
limit). They are emitted at module scope as `<kernel>$<name>` with `.align 16`.

### Warp intrinsics

`shfl_sync`, `shfl_down_sync`, `shfl_up_sync`, `shfl_xor_sync` (each with an optional sub-warp
`width`, a power of two from 2 to 32), `ballot_sync` and `activemask`. The `__` spellings
(`__shfl_down_sync`) work too.

```c
float v = in[i];
for (int off = 16; off > 0; off = off / 2) {
    v = v + shfl_down_sync(0xFFFFFFFF, v, off);
}
```

### Atomics

`atomic_add`, `atomic_sub`, `atomic_min`, `atomic_max`, `atomic_and`, `atomic_or`,
`atomic_xor`, `atomic_exch` and `atomic_cas` (also as `atomicAdd` etc.) on global pointers and
shared arrays. Each returns the old value. `add` and `exch` also work on `float`; the rest are
`int`-only. They accept `atomic_add(p, v)`, `atomic_add(p, i, v)` or `atomic_add(p[i], v)`.

### Vector types

```c
kernel scale4(float4* data, float s) {
    float4 v = data[thread_idx];                         // one ld.global.v4.f32
    data[thread_idx] = make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}
```

`load_float4(p, k)` / `store_float4(p, k, v)` (and the `int2`/`int4`/`float2` versions) read
or write a vector through a scalar pointer or shared array. The offset `k` counts whole vectors,
like `reinterpret_cast<float4*>(p)[k]` in CUDA, so every access is naturally aligned.

### Rules that matter for correctness

- **`bool` mixes with nothing.** `cast<bool>(x)` means `x != 0`, and NaN counts as true.
  Conditions accept `bool` or `int`.
- **Float comparisons follow C.** Every comparison with a NaN is false except `!=`, which is
  true. kernc lowers `!=` to the unordered `setp.neu.f32`.
- **Shifts follow PTX, not C.** The count is clamped to 32, so `x << 32` is 0 and `x >> 40`
  is all sign bits. Constant folding uses the same rule.
- **Integer literals must fit in 32 bits.** Decimal up to `2147483647`, with `-2147483648`
  allowed (the rule from Java); hex up to `0xFFFFFFFF`, read as a bit pattern.
- **`return;` ends only the calling thread.** Kernels return `void`; a non-`void` device
  function must return a value on every path.
- **Every thread in a block must reach every `barrier()`.** kernc warns about a barrier inside
  control flow that depends on the thread.
- **Kernel parameters cannot be `bool`**, because a `.param` cannot have predicate type. Pass an
  `int`.

---

## Command-line tools

### `kernc`

```
kernc <input.kern> [options]
```

| Option | Meaning |
|---|---|
| `-o <file>` | Write PTX to `<file>` (default: stdout) |
| `-O0` | Disable the optimizer |
| `-O1` | Enable the optimizer (default) |
| `-lineinfo`, `--generate-line-info` | Emit `.file` / `.loc` source line info |
| `--dump-tokens` | Print the token stream |
| `--dump-ir` | Print the IR before optimization |
| `--dump-opt-ir` | Print the IR after optimization |
| `-h`, `--help` | Show usage |

kernc exits with status 1 if there are any errors. Diagnostics show the location, the source
line and a caret:

```
bad.kern:3:15: error: type mismatch in variable initialization: cannot initialize 'float' with 'float*'
      float x = out;
                ^
bad.kern:4:14: error: integer literal '3000000000' is out of range for type 'int'
      out[i] = 3000000000;
               ^
```

Warnings don't stop compilation:

```
warn.kern:5:9: warning: 'barrier()' called inside divergent branch; all threads in a block must reach the barrier to avoid deadlock
          barrier();
          ^
warn.kern:6:18: warning: variable 'acc' may be uninitialized
          out[i] = acc;
                   ^
```

### `kernc-runner`

`kernc-runner` loads PTX with `cuModuleLoadDataEx` (the driver JIT-compiles it), allocates and
initializes device buffers, launches with `cuLaunchKernel`, and prints results. It is how the
command-line tests and the compute-sanitizer tests run kernels.

```
kernc-runner [options] <ptx_file> <kernel_name>
```

| Option | Meaning |
|---|---|
| `-p`, `--ptx <path>` / `-k`, `--kernel <name>` | PTX file and kernel (or give them positionally) |
| `-g`, `--grid <n>` | Number of blocks (default 1); `2,3,2` or `4x4` for 2D/3D |
| `-b`, `--block <n>` | Threads per block (default 256); 2D/3D as for `--grid` |
| `--arg <spec>` | One kernel argument, in order: `i32:<v>`, `f32:<v>`, or `buf:<i32\|f32>:<N>:<zeros\|seq\|rand>` |
| `--dump <idx>[:<count>]` | After the launch, print the first `<count>` elements of buffer argument `<idx>` (default `min(N, 16)`) |
| `-v`, `--verbose` | Print the driver JIT info and error logs |
| `-h`, `--help` | Show usage |

Before launching, the runner checks every `--arg` against the kernel's PTX `.param` list, both
the count and the type, so a wrong argument is an error and never a fault or a silently wrong
result:

```
Error: argument 0 ('i32:2') does not match kernel 'saxpy': its PTX parameter is .f32, which needs a float (f32:<val>).
```

---

## Architecture

```
            ┌──────────────────────────── front end ───────────────────────────┐
 .kern ──►  Lexer ──► Parser ──► Sema ──► IRGen ──►
            tokens    AST        types,   3-address IR,
            line:col             scopes,  basic blocks,
                                 warnings loops ≤16 unrolled

            ┌───────────────── optimizer (-O1) ─────────────────┐
       ──►  Inlining ──► ┌ constant folding ┐ ──► Instruction ──►  PTXEmitter ──► .ptx
            (module)     │ CSE + copy prop. │     scheduling       registers,
                         │ dead-code elim.  │     (per block)      state spaces,
                         └ CFG cleanup ─────┘                      .file/.loc
                          repeated until no change (≤10 rounds)
```

| Stage | Files | What it does |
|---|---|---|
| Lexer | `Lexer.cpp`, `Token.h` | `string_view` tokens carrying line and column; decimal, hex and float literals |
| Parser | `Parser.cpp`, `AST.h` | Recursive descent with precedence climbing; keeps parentheses (`ParenExpr`) so Sema can apply the literal-range rule |
| Sema | `Sema.cpp` | Nested scopes, type rules, literal range, shared-memory limit, call checking, definite-assignment and barrier-divergence warnings |
| IRGen | `IRGen.cpp`, `IR.h` | Non-SSA three-address IR with typed virtual registers; booleans as jumping code; unrolls `for` loops with a constant trip count of at most 16 |
| Optimizer | `Passes.cpp` | The passes below |
| PTXEmitter | `PTXEmitter.cpp` | Register declarations per type class, `.param` / `.global` / `.shared` addressing, prototypes, `.loc` |
| Driver | `main.cpp`, `Diagnostics.cpp` | Command line, pass pipeline, diagnostic printing |

### Optimizer passes

| Pass | What it does |
|---|---|
| Inlining | Builds the call graph and inlines non-recursive device functions of at most 64 IR instructions, with fresh registers and labels per call site. Recursive and mutually recursive functions stay as calls. |
| Constant folding | Folds and simplifies with the exact semantics of the PTX instruction (see [design decision 4](#4-constant-folding-is-an-interpreter-for-the-target)). Lowers signed `/` and `%` by a constant to shifts (powers of two) or a magic-number multiply (other constants). |
| CSE and copy propagation | Within a basic block. A redefinition kills every expression that uses the register; stores, barriers, atomics, warp intrinsics and calls kill loads (see [design decision 6](#6-memory-ordering-who-may-move-a-load)). |
| Dead-code elimination | Removes instructions whose results are unused. Stores, atomics, barriers and calls are always kept. |
| CFG cleanup | Removes unreachable blocks, folds branches to the next block, redirects jumps through blocks that only branch, and merges straight-line blocks, which gives the local passes bigger blocks to work on. |
| Instruction scheduling | List scheduling inside each block, prioritized by latency-weighted critical path (global loads 100, shared loads 80, ALU 20, stores 1), so loads start early. |

The IR and code-generation rules are in [docs/specs/03-ir-and-codegen.md](docs/specs/03-ir-and-codegen.md).

---

## Performance compared with nvcc

ptxas is itself an optimizing compiler, so the useful question is what reaches the hardware. For
each kernel I wrote the same algorithm in CUDA, compiled it with `nvcc -arch=sm_86 -cubin`,
assembled kernc's PTX with `ptxas -arch=sm_86` (CUDA 13.0), and compared the cubins:

- **Instructions:** non-NOP lines of `cuobjdump -sass` (this includes the final self-branch `BRA`).
- **Registers:** `cuobjdump -res-usage`.

| Kernel | kernc `-O0` | kernc `-O1` | nvcc |
|---|---|---|---|
| SAXPY | 16 instr, 10 regs | 16 instr, 10 regs | 16 instr, 10 regs |
| `reduce_sum` (shared-memory reduction) | 70 instr, 12 regs, 2 × `MUFU.RCP` | **37 instr, 8 regs** | 37 instr, 10 regs |
| `x / 3` and `x % 3` | 32 instr, 10 regs, `MUFU.RCP` | **13 instr, 10 regs** | 12 instr, 10 regs |
| `functions.kern` (whole module) | 228 instr, 11 `CALL` | **145 instr, 6 `CALL`** | 143 instr, 6 `CALL` |

### SAXPY: ptxas cleans up the rest

For a kernel this small, ptxas removes everything kernc's `-O0` output leaves behind. The SASS
from kernc and from nvcc is the same apart from register numbering.

### Division by a constant: ptxas will not do it

At `-O0`, `s = s / 2` in the reduction loop is a `div.s32`, and ptxas expands every `div.s32`
into a reciprocal approximation on the special-function unit (`MUFU.RCP`) plus a correction
sequence. That is why the `-O0` reduction is 70 instructions. kernc's `-O1` lowers signed
division by a constant itself:

- **By 2^k:** add a bias of `(x >> 31) >>> (32 - k)` and shift right arithmetically. The bias
  makes the result round toward zero, as C requires (`-7 / 2 == -3`, where a plain shift gives
  `-4`). For `k = 1`, ptxas fuses the shift and add into one `LEA.HI`.
- **By any other constant** (`d ∉ {0, ±1, INT_MIN}`): the Granlund–Montgomery magic-number
  method from *Hacker's Delight* (`computeSignedMagic`). `x / 3` becomes a high multiply by
  `0x55555556`, plus the sign bit of `x`:

  ```ptx
  mul.hi.s32 %r4, %r0, %r1;      // %r1 = 1431655766 = 0x55555556
  shr.u32    %r5, %r0, %r2;      // %r2 = 31: the sign bit of x
  add.s32    %r6, %r4, %r5;      // q = x / 3
  mul.lo.s32 %r7, %r3, %r6;      // %r3 = 3
  sub.s32    %r8, %r0, %r7;      // r = x - 3q = x % 3
  ```

  ptxas turns this into one `IMAD.HI` (with the sign bit folded into its addend), one `SHF`
  and one `IMAD` for the remainder:

  ```
  SHF.R.U32.HI R3, RZ, 0x1f, R7
  IMAD.HI R5, R7, 0x55555556, R2
  IMAD R7, R5, -0x3, R7
  STG.E [R2.64], R5
  STG.E [R2.64+0x4], R7
  ```

The GPU tests check both lowerings bit-for-bit against hardware `div.s32` / `rem.s32` (the
`-O0` build) and against the CPU, including negative values, zero and `INT_MIN`.

### Inlining: ptxas will not do that either

ptxas never inlines a PTX `.func`: at `-O0` all 11 calls in `functions.kern` stay `CALL.REL`
in the SASS. nvcc's front end inlines before it emits PTX, so kernc has to as well. After
inlining, kernc and nvcc leave the same six calls: the self-recursive `fib`, and the mutually
recursive `is_even` / `is_odd`. The kernel `use_functions` uses 24 registers and a 0-byte stack
frame in all three builds. The CUDA version is a line-by-line translation.

---

## Debugging with NVIDIA's tools

kernc's `-lineinfo` works like nvcc's `-lineinfo`, not `-G`: optimization stays on, and each IR
instruction carries the location of the statement that produced it. The emitter writes a `.loc`
whenever the location changes, and ptxas carries the table through its own optimizations into
the cubin. The [walkthrough](#a-walkthrough-saxpy-from-source-to-sass) shows the result in
`nvdisasm --print-line-info`.

NVIDIA's runtime tools read the same table. Run through `kernc-runner` under
`compute-sanitizer --tool memcheck`, an out-of-bounds write
([`tests/sanitizer/oob.kern`](tests/sanitizer/oob.kern)) is reported against its source line.
Without `-lineinfo` the report only gives `oob+0xa0`.

```
========= Invalid __global__ write of size 4 bytes
=========     at oob+0xa0 in oob.kern:5
=========     by thread (0,0,0) in block (0,0,0)
=========     Access to 0x507a04000 is out of bounds
```

`compute-sanitizer --tool racecheck` finds **0 hazards** in `examples/reduce.kern`. With its
`barrier()` calls removed, it reports the shared-memory race at the right lines: the read of
`sdata[tid + s]` against the writes on lines 16 and 20.

```
========= Error: Race reported between Read access at reduce_sum+0x170 in reduce_nobar.kern:20
=========     and Write access at reduce_sum+0x110 in reduce_nobar.kern:16 [2432 hazards]
=========     and Write access at reduce_sum+0x1a0 in reduce_nobar.kern:20 [2032 hazards]
```

Both checks run as the CTest `compute_sanitizer_memcheck_racecheck`.

> **WSL2:** compute-sanitizer needs the GPU debugger interface. In an Administrator PowerShell
> on Windows, run
> `reg add "HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger" /v EnableInterface /t REG_DWORD /d 1 /f`,
> then `wsl --shutdown`. Until then the sanitizer test is reported as skipped, not failed.

---

## Design decisions

These are the choices I'd expect to be asked about, and why I made them.

### 1. A non-SSA three-address IR

PTX already has unlimited virtual registers and no phi nodes, and ptxas does the real register
allocation. An SSA IR would need an out-of-SSA pass just to get back to PTX. The cost shows up in
the optimizer: a variable's register can be assigned more than once, so CSE and copy
propagation have to treat every assignment as a kill (an available expression or copy dies as
soon as one of its registers is redefined), and they work within a basic block, not across the
whole function.

### 2. A register's type is fixed when the register is created

`IRFunction::freshReg(IRType)` records the type and has no default argument, so every place
that creates a register has to state one. The original emitter guessed a register's class from
the first instruction that used it, so an uninitialized `int` compared in a `setp` became a
`.pred` register, and ptxas rejected the PTX.

### 3. Booleans: jumping code and values

In a condition, `a && b` never produces a value; it becomes branches (the "jumping code" scheme
in the Dragon Book, §6.6). As a value (`bool ok = i < n && x[i] > 0;`) it still has to
short-circuit: `and.pred` would evaluate both sides and read `x[i]` out of bounds. So kernc
reuses the branches and writes true or false into the result on each path. Comparisons `setp`
straight into a predicate, `!` is `not.pred`, `==` / `!=` on bools are `xor.pred`, and
`cast<int>(b)` is `selp`.

### 4. Constant folding is an interpreter for the target

Folding must produce exactly what the PTX instruction would produce at run time, not what C++
would. The `ptx::` helpers in `Passes.cpp` model one instruction each:

- `cvt.rzi.s32.f32` saturates and maps NaN to 0, so `cast<int>(1e20)` folds to `INT_MAX`.
- `.s32` arithmetic wraps, so the folder computes in `uint32_t` (signed overflow in the compiler
  would be undefined behaviour, which the UBSan build checks for).
- Shift counts are clamped to 32, not masked to 5 bits as on x86.
- Division by zero and `INT_MIN / -1` are never folded; they are left to the hardware.
  (Folding `INT_MIN / -1` on the host crashed kernc with SIGFPE.)

The differential GPU test (the same kernel at `-O0` and `-O1` must give bit-identical results)
is what keeps this honest.

### 5. Address arithmetic depends on the state space

Global pointers are 64-bit: the index is widened and scaled in one `mul.wide.s32`, then added
with `add.s64` to a base converted once with `cvta.to.global`. Shared arrays are module-scope
`.shared` symbols (nvcc also puts them at module scope). Their addresses are 32-bit offsets
into the block's shared window (`mov.u32` of the symbol, `mul.lo.s32`, `add.s32`), which saves
registers. All shared arrays get `.align 16`, so a `v4` access to any of them is legal.

### 6. Memory ordering: who may move a load

Two passes move or remove memory operations, and both need an aliasing rule:

- **CSE** can reuse an earlier load only if nothing in between could have changed the memory.
  A global store kills every global load (two pointer parameters may point to the same
  buffer); a shared store kills loads from the same array. Barriers, atomics, warp intrinsics
  and calls kill every load.
- **The scheduler** may swap a load and a store only when they access *two different shared
  arrays*, the one case where kernc can prove the addresses differ. An earlier version also
  treated different base pointers as disjoint; with `a == b`, `a[i] = 42; out[i] = b[i];`
  read the stale value at `-O1`. The GPU tests now pass the same buffer for both pointers.

`bar.sync` is a fence for both passes. Moving shared-memory accesses across a barrier is
exactly the race that racecheck catches.

### 7. Inlining happens in kernc, because ptxas won't

ptxas treats a `.func` as a real call, so a call costs an ABI sequence and blocks optimization
across the boundary. kernc inlines non-recursive functions of up to 64 IR instructions. It
finds recursion, including mutual recursion, from the call graph, and leaves those functions as
calls. An inlined `bool` parameter or return value also loses its `.b32` round trip
(`selp` to 0/1, then `setp.ne` back), which exists only for the ABI.

### 8. Device functions use PTX's abstract calling convention

A call stores its arguments into scoped `.param` variables and runs `call ($ret), f, (...)`;
ptxas maps that onto the real ABI (registers, and a stack frame when needed). Every `.func` gets
a prototype at the top of the module, because ptxas rejects a call to a later function without
one ("requires call prototype"), and this is also what makes mutual recursion work. kernc emits
plain `call` rather than `call.uni`, because `call.uni` promises the call is never divergent.

### 9. Generated names live where user names can't

Every PTX name kernc makes up contains `$` (`$entry`, `$L_then_0`, `f$param_0`, `f$retval`,
`k$buf`). PTX identifiers allow `$` and `.kern` identifiers don't, so a collision is
impossible, not just unlikely. Before this change, a function named `entry` or `f_retval`
produced PTX that ptxas rejected. nvcc keeps its labels apart the same way (`$L__BB0_2`).

### 10. Integer literal range follows Java

`-2147483648` lexes as minus applied to `2147483648`, which doesn't fit in an `int`. C gets
around this with a wider literal type; kernc has none, so it uses Java's rule (JLS §3.10.1): the
literal `2147483648` is allowed only as the operand of unary minus. The check is in Sema, not
the lexer, because only Sema can see the minus. The parser keeps parentheses, so
`-(2147483648)` is still an error, as in Java.

### 11. Loop unrolling is in IRGen

A `for` loop with a constant trip count of at most 16 is unrolled while the IR is generated, with
the induction variable bound to a constant in each copy. Constant folding then folds the
addresses, and the scheduler can overlap the loads from different iterations. Doing it in IRGen
is simple, but it means unrolling also happens at `-O0`.

### 12. Diagnostics guard kernc's own kernels

The barrier-divergence warning is checked against the project itself: a test compiles every
shipped kernel and fails on any warning that the file doesn't declare with
`// Expected warning:`. The first run found a `barrier()` inside `if (i < n)` in one of
kernc's own regression kernels, which could deadlock in a partially filled block.

---

## Testing

```bash
ctest --test-dir build --output-on-failure             # everything (205 tests)
ctest --test-dir build -R KernelE2ETest                # only the GPU tests
ctest --test-dir build -R "ptx_|lineinfo"              # only the ptxas / nvdisasm checks
./build/tests/kernc_unit_tests --gtest_filter='*CSE*'  # one group of unit tests
```

| Suite | Tests | What it checks |
|---|---|---|
| Unit tests (GoogleTest) | 138 | Lexer 9, parser 16, Sema 41, IR and passes 72 |
| GPU end-to-end (Driver API) | 31 | Kernels JIT-loaded with `cuModuleLoadDataEx`, launched, and compared with a CPU reference |
| ptxas acceptance | 6 | Every file in `examples/` assembles with `ptxas -arch=sm_86` |
| ptxas regression | 24 | The 12 kernels in `tests/regress/` assemble at both `-O0` and `-O1` |
| Line tables | 2 | `kernc -lineinfo` → ptxas → `nvdisasm --print-line-info`: each `//@ REGEX` line must own matching SASS, at `-O0` and `-O1` |
| Diagnostics | 1 (11 files) | Each `tests/diagnostic/err_*.kern` must fail with the expected message and location |
| No unexpected warnings | 1 | Every shipped kernel compiles without warnings unless it declares them |
| kernc-runner | 1 (21 stages) | Launching, argument parsing, `--dump`, and the error paths, including `--arg` type checking |
| compute-sanitizer | 1 | memcheck reports the out-of-bounds write at its `.kern` line; racecheck finds the reduction clean, and the barrier-free version racy |

The GPU tests cover, among other things:

- vector add, SAXPY, clamp and ReLU at sizes that are not block multiples (1000, 117, 33, 1);
- the reduction for 1 to 2²⁰ + 3 elements, and a block reverse through shared memory;
- 13 kinds of `bool` expression, with NaN and `-0.0` inputs;
- warp shuffles (down, up, xor, idx, with sub-warp widths), ballot and activemask;
- every atomic under contention from 4096 threads;
- device functions with float, bool, pointer and void calls, recursion and mutual recursion;
- signed division and modulo by powers of two and other constants, bit-for-bit against
  hardware `div.s32` and the CPU, including `INT_MIN`;
- CSE with two pointer parameters that point to the same buffer, so a store must kill a load;
- the scheduler keeping a store before a load when memory may alias;
- unrolled loops (a stencil, a dot product), vector loads and stores, and 2D/3D launches.

Three techniques do most of the work:

- **Differential testing.** The same kernel at `-O0` and `-O1` must give bit-identical results on
  the GPU. This needs no hand-written expected values, because the hardware decides what's
  correct.
- **Testing on the real target.** "ptxas accepts it" is not enough: the misaligned vector loads
  and the scheduler aliasing bug both assembled fine and only failed on the GPU.
- **Mutation checks.** After each fix, I broke it again on purpose to confirm a test fails.
  For example, `setp.ne` in place of `setp.neu` fails the NaN test; removing the sign bias in
  power-of-two division fails on negative numbers; turning off load invalidation in CSE gives 19
  instead of 36 in the aliasing test; removing `bar.sync` fails the block reverse at thread 64,
  the first thread of the third warp; and removing `.loc` emission fails the line-table and
  memcheck tests.

### Bugs the tests found

The first version passed its own 37 tests. Testing against the GPU, ptxas and UBSan then found
these. Each fix is a separate commit, with a test that failed before the fix.

| Bug | Effect |
|---|---|
| Register class inferred from the first use | ptxas rejected kernels that read an uninitialized variable in `setp`, `mul.wide` or `st` |
| `cast<int>(1e20)` folded with C++ semantics | `INT_MIN` at `-O1` but `INT_MAX` on the GPU at `-O0` |
| `INT_MIN / -1` folded on the host | **kernc itself crashed** with SIGFPE |
| Signed overflow while folding | Undefined behaviour inside the compiler (it happened to wrap) |
| `if (v % 2)` treated as a comparison | **Compiled as `v == 2`**; the `%` was never computed |
| Float `!=` lowered to `setp.ne.f32` | `x != x` was false for NaN |
| Comparisons used as values | Lowered to `add.s32` (hidden until `bool` existed) |
| Out-of-range literals | `3000000000` was silently truncated |
| Generated PTX names shared the user's namespace | A function named `entry` or `f_retval`, or two kernels' shared arrays, made ptxas reject a valid program |
| Scheduler treated different pointers as non-aliasing | **Wrong results at `-O1`** when two pointers point to the same buffer |
| Shift folding masked the count to 5 bits (x86) | `1 << 32` folded to 1; the GPU gives 0 |
| Vector intrinsic offsets counted scalars | `load_int4(p, 1)` was a 16-byte load at a 4-byte boundary: `CUDA_ERROR_MISALIGNED_ADDRESS` |
| Atomics accepted vector targets | `atomic_add` on a `float4*` emitted an integer `atom.add.s32` on float data |
| kernc-runner checked only the argument count | An `i32` passed for a pointer faulted; an `i32` for a float printed **`Success!` with wrong results** |
| `barrier()` inside `if (i < n)` in a regression kernel | Possible deadlock in partially filled blocks; found by the barrier-divergence warning |
| Vector values in function signatures were accepted | **kernc segfaulted** on a `float4` kernel parameter or on any call passing or returning a vector; an uncalled `func int2` was emitted with a `.b32` return value, dropping `y` |

---

## Limitations

- **Not supported yet:** structs, 64-bit integers other than pointers, `bool` kernel
  parameters, and vector values as kernel or device-function parameters or return values
  (Sema rejects them and suggests a vector pointer, which works).
- **The optimizer is local.** CSE, copy propagation and scheduling work within basic blocks;
  there is no global value numbering, loop-invariant code motion or partial unrolling.
- **Unrolling is not controlled by `-O0`**, because it happens in IRGen.
- **Line info is per statement**, not per expression.
- **One target.** kernc emits PTX for sm_86 only.

---

## Repository layout

```
include/kernc/       headers: Token, Lexer, AST, Parser, Sema, IR, IRGen, Passes, PTXEmitter, Diagnostics
src/                 the compiler; main.cpp is the kernc command-line driver
tools/runner/        kernc-runner: launch PTX on the GPU through the CUDA Driver API
examples/            vecadd, saxpy, clamp, relu, reduce (shared memory), functions (device functions)
tests/unit/          GoogleTest: lexer, parser, Sema, IR and passes
tests/e2e/           GPU tests through the CUDA Driver API
tests/regress/       12 kernels assembled by ptxas at -O0 and -O1
tests/diagnostic/    err_*.kern files, their checker, and the no-unexpected-warnings check
tests/lineinfo/      line-table kernel and check_lineinfo.py (ptxas + nvdisasm)
tests/runner/        kernc-runner command-line tests
tests/sanitizer/     compute-sanitizer memcheck / racecheck test
docs/specs/          language scope, architecture, IR and code generation, testing strategy
```

Further reading in `docs/specs/`:
[language scope](docs/specs/01-scope.md),
[architecture](docs/specs/02-architecture.md),
[IR and code generation](docs/specs/03-ir-and-codegen.md),
[testing strategy](docs/specs/04-testing.md).

---

Adithya · [adithyasankarmenon@gmail.com](mailto:adithyasankarmenon@gmail.com)
