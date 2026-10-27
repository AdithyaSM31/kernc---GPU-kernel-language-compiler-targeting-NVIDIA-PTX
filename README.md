# kernc: a GPU kernel compiler targeting NVIDIA PTX (sm_86)

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Target](https://img.shields.io/badge/PTX-sm__86-orange.svg)](https://docs.nvidia.com/cuda/parallel-thread-execution/)

`kernc` is a compiler, written in C++17, for a small C-like GPU kernel language (`.kern`
files). It emits PTX that `ptxas -arch=sm_86` accepts, and the test suite runs that PTX on an
NVIDIA GeForce RTX 3060 Laptop GPU through the CUDA Driver API, checking every result against a
CPU reference.

```c
kernel saxpy(float a, float* x, float* y, int n) {
    int i = block_idx * block_dim + thread_idx;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

What it supports:

- `int`, `float`, `bool`, and vector types `int2`, `int4`, `float2`, `float4`; pointers `int*`, `float*`, `int2*`, `int4*`, `float2*`, `float4*`; `shared` arrays with 8/16-byte alignment
- Vector component access (`.x`, `.y`, `.z`, `.w`), constructors (`int2(...)`, `make_float4(...)`, etc.) with implicit int-to-float promotion, and 64/128-bit vector memory operations (`ld.global.v2/v4`, `st.global.v2/v4`, `ld.shared.v2/v4`, `st.shared.v2/v4`) lowering to hardware SASS `LDG.E.64/128`, `STG.E.64/128`, `LDS.64/128`, `STS.64/128`
- Vector intrinsics (`load_int2/4`, `load_float2/4`, `store_int2/4`, `store_float2/4`) on scalar and vector buffers
- 1D, 2D, and 3D execution geometry (`thread_idx[_x|_y|_z]`, `block_idx_*`, `block_dim_*`, `grid_dim_*`, or `.x/.y/.z`)
- warp-level intrinsics (`shfl_sync`, `shfl_down_sync`, `shfl_up_sync`, `shfl_xor_sync`, `ballot_sync`, `activemask`, with and without `__`)
- atomic memory operations on global and shared memory (`atom.global`, `atom.shared` for `add`, `sub`, `min`, `max`, `and`, `or`, `xor`, `exch`, `cas`)
- device functions (`func`), including recursion and mutual recursion
- `if` / `else`, `while`, `for`, `return`, `barrier()`
- short-circuit `&&` / `||`, in conditions and as values
- explicit `cast<T>()` between `int`, `float` and `bool`
- bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`) on `int` with ISO C precedence
- constant folding, algebraic simplification, and dead-code elimination (with dead vector load pruning)
- `.file` / `.loc` source line info with `-lineinfo`
- diagnostics with `file:line:column` and a caret

The suite has 164 tests, and all of them pass, both in a normal build and under
UndefinedBehaviorSanitizer.

---

## Quick start (WSL2 Ubuntu 22.04)

Requirements: CMake 3.16+, GCC 11+, Python 3, GoogleTest (`libgtest-dev`), and the CUDA 13.0
toolkit (`/usr/local/cuda-13.0`). On WSL2 the GPU driver comes from Windows
(`/usr/lib/wsl/lib/libcuda.so`); do not install a Linux driver inside WSL.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build && ctest --output-on-failure
```

Compile a kernel:

```bash
./build/kernc examples/saxpy.kern -o saxpy.ptx            # optimized (-O1 is the default)
./build/kernc examples/saxpy.kern -O0 --dump-ir           # print the unoptimized IR
./build/kernc examples/saxpy.kern -lineinfo -o saxpy.ptx  # add .file/.loc line info
/usr/local/cuda-13.0/bin/ptxas -arch=sm_86 saxpy.ptx -o saxpy.cubin
```

| Option | Meaning |
|---|---|
| `-o <file>` | Output PTX file (default: stdout) |
| `-O0` / `-O1` | Disable / enable optimization (default `-O1`) |
| `-lineinfo`, `--generate-line-info` | Emit `.file` / `.loc` source line info |
| `--dump-tokens`, `--dump-ir`, `--dump-opt-ir` | Print tokens, IR before / after optimization |

Execute a kernel on the GPU (`kernc-runner`):

```bash
./build/kernc-runner vecadd.ptx vecadd \
    --grid 4 --block 256 \
    --arg buf:i32:1024:seq \
    --arg buf:i32:1024:seq \
    --arg buf:i32:1024:zeros \
    --arg i32:1024 \
    --dump 2:8
```

`kernc-runner` JIT-loads the PTX through the CUDA Driver API and launches it with typed arguments,
1D/2D/3D grid and block geometry, initialized device buffers (`zeros`, `seq`, `rand`) and
`--dump` to print results. Before launching, it checks every `--arg` against the kernel's PTX
`.param` list: the count, and the type (`.f32` needs `f32:`, `.s32` needs `i32:`, a `.u64` pointer
needs `buf:`). A mismatch is reported by argument number instead of faulting or silently producing
wrong results.

| Runner Option | Meaning |
|---|---|
| `-p`, `--ptx <path>` | Path to PTX file (or positional argument) |
| `-k`, `--kernel <name>` | Kernel function name (or positional argument) |
| `-g`, `--grid <n>` | Grid dimension / block count (default: `1`) |
| `-b`, `--block <n>` | Block dimension / threads per block (default: `256`) |
| `--arg <spec>` | Kernel argument in call order: `i32:<val>`, `f32:<val>`, or `buf:<i32\|f32>:<N>:<init>` (`zeros`, `seq`, `rand`) |
| `--dump <idx>[:<cnt>]` | Copy back and display first `<cnt>` elements (default `min(N, 16)`) of buffer `<idx>` |
| `-v`, `--verbose` | Display Driver JIT compilation logs |
| `-h`, `--help` | Show command-line usage |

To build with UndefinedBehaviorSanitizer, which turns undefined behaviour inside the compiler
into a hard failure:

```bash
cmake -S . -B build-ubsan -DKERNC_SANITIZE=ON
cmake --build build-ubsan -j
cd build-ubsan && ctest
```

---

## The language

The full reference is in [docs/specs/01-scope.md](docs/specs/01-scope.md).

| | |
|---|---|
| Types | `int` (`.s32`), `float` (`.f32`), `bool` (`.pred`); vectors `int2`, `int4`, `float2`, `float4`; pointers `int*`, `float*`, `int2*`, `int4*`, `float2*`, `float4*` |
| Built-ins | `thread_idx[_x\|_y\|_z]`, `block_idx_*`, `block_dim_*`, `grid_dim_*` (or `.x`, `.y`, `.z` / `threadIdx`, etc.) |
| Intrinsics | Warp: `shfl_sync`, `shfl_down_sync`, `shfl_up_sync`, `shfl_xor_sync`, `ballot_sync`, `activemask`<br>Atomic: `atomic_add`, `atomic_sub`, `atomic_min`, `atomic_max`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_exch`, `atomic_cas`<br>Vector: `load_int2/4`, `load_float2/4`, `store_int2/4`, `store_float2/4` |
| Statements | declarations, assignment, `if`/`else`, `while`, `for`, `return`, `barrier();`, `shared T name[N];`, function calls |
| Operators | `+ - * / %`, unary `- ! ~`, `& \| ^ << >>`, `< <= > >= == !=`, `&&` `\|\|` (short-circuit), `p[i]`, vector member `.x .y .z .w` |
| Constructors | `int2(...)`, `int4(...)`, `float2(...)`, `float4(...)` (and `make_*`) |
| Casts | `cast<int>(x)`, `cast<float>(x)`, `cast<bool>(x)` |

Rules that matter for correctness:

- **No implicit conversions except `int` → `float`.** `bool` mixes with nothing; use `cast<>`.
  `cast<bool>(x)` means `x != 0`, and NaN counts as true.
- **Float comparisons follow C.** Every comparison is false when an operand is NaN, except
  `!=`, which is true. It lowers to the unordered `setp.neu.f32`.
- **Integer literals must fit in 32 bits.** kernc uses the same rule as Java: decimal values up
  to `2147483647`, except that `-2147483648` is allowed; hex up to `0xFFFFFFFF`, read as a bit
  pattern.
- **`return;` ends the calling thread only.** Kernels return void, so `return x;` is an error.
  Device functions returning non-void must return a value on all control paths.
- **Shared memory.** `shared` arrays have `int`/`float` elements and a literal size, with at
  most 48 KB per kernel (the sm_86 static limit). Every thread in a block must reach every
  `barrier()`. Shared arrays can only be declared in kernels.
- **Device functions (`func`).** Declared with `func <type> name(params) { ... }`. May return `void`,
  `int`, `float` or `bool`. Unlike kernels, device functions can accept `bool` parameters (marshaled
  across the ABI as 32-bit values) and can be recursive. Pointer parameters receive already-converted
  global addresses (no redundant `cvta`). Forward calls and mutual recursion are supported via
  module-level prototypes.
- **`bool` cannot be a kernel parameter.** A `.param` variable cannot have predicate type, so
  pass an `int` to a kernel instead.

---

## Architecture

```
 .kern ──► Lexer ──► Parser ──► Sema ──► IRGen ──► Optimizer ──► PTXEmitter ──► .ptx
           tokens    AST        typed     3-address   constant      registers,
           with      (unique_   AST,      IR, basic   folding,      state spaces,
           line:col  ptr)       scopes    blocks      DCE           .file/.loc
```

| Stage | Files | Notes |
|---|---|---|
| Lexer | `Lexer.cpp`, `Token.h` | `string_view` tokens with line:column; decimal, hex and float literals |
| Parser | `Parser.cpp`, `AST.h` | Recursive descent with precedence climbing for binary operators |
| Sema | `Sema.cpp`, `Sema.h` | Nested scopes, type rules, literal range, shared-memory limit |
| IRGen | `IRGen.cpp`, `IR.h` | Non-SSA three-address IR with typed virtual registers and basic blocks |
| Optimizer | `Passes.cpp` | Constant folding (with PTX semantics) and dead-code elimination, repeated until nothing changes (at most 10 rounds) |
| PTXEmitter | `PTXEmitter.cpp` | Register declarations, `.param`/`.global`/`.shared` access, line info |
| Driver | `main.cpp` | CLI; diagnostics print as `file:line:col: error: ...` with the source line and a caret |

The IR and code-generation rules are in [docs/specs/03-ir-and-codegen.md](docs/specs/03-ir-and-codegen.md).

---

## Example: SAXPY

`./build/kernc examples/saxpy.kern` (the `.version` / `.target` / `.address_size` header is
omitted):

```ptx
.visible .entry saxpy(
    .param .f32 saxpy$param_0,
    .param .u64 saxpy$param_1,
    .param .u64 saxpy$param_2,
    .param .s32 saxpy$param_3
)
{
    .reg .s32   %r<10>;
    .reg .f32   %f<5>;
    .reg .s64   %rd<10>;
    .reg .pred  %p<1>;

$entry:
    ld.param.f32 %f0, [saxpy$param_0];
    ld.param.u64 %rd0, [saxpy$param_1];
    cvta.to.global.u64 %rd1, %rd0;
    ld.param.u64 %rd2, [saxpy$param_2];
    cvta.to.global.u64 %rd3, %rd2;
    ld.param.s32 %r0, [saxpy$param_3];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.s32 %r3, %r1, %r2;
    mov.u32 %r4, %tid.x;
    add.s32 %r5, %r3, %r4;
    mov.s32 %r6, %r5;
    setp.lt.s32 %p0, %r6, %r0;
    @%p0 bra $L_then_0;
    bra $L_if_end_1;
$L_then_0:
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
    bra $L_if_end_1;
$L_if_end_1:
    ret;
}
```

### Compared with nvcc

I compiled the same kernel as CUDA with `nvcc -arch=sm_86` and assembled both PTX files with
ptxas 13.0:

| | kernc | nvcc |
|---|---|---|
| PTX | `mul.lo` + `add`, `mul.f32` + `add.f32`, the `y[i]` address computed twice, index scale held in a register | `mad.lo.s32`, `fma.rn.f32`, one shared `mul.wide.s32 %rd5, %r1, 4`, `cvta` moved after the bounds check |
| SASS after ptxas | 16 instructions, 10 registers | 16 instructions, 10 registers |

(Instruction counts here exclude NOP padding.)

The SASS is identical apart from register numbering. ptxas removed the duplicate address
computation, fused `mul.f32` + `add.f32` into `FFMA`, and turned the bounds check into
`@P0 EXIT`. For a kernel this simple, ptxas cleans up kernc's PTX completely.

The reduction is where front-end optimization matters. Originally, kernc emitted `div.s32` for `s = s / 2`,
which ptxas expanded into a 20-instruction Newton-Raphson approximation sequence using `MUFU.RCP`, resulting
in **70 SASS instructions and 12 registers** (compared to CUDA's **37 and 10**).

With strength reduction implemented in `kernc`'s optimizer (`-O1`), signed division and modulo by constant
powers of two ($2^k$) are lowered to branchless shift/add/mask sequences that truncate toward zero:

- For $k = 1$ ($s / 2$): `shr.u32 %r_sign, %r_x, 31` $\to$ `add.s32 %r_biased, %r_x, %r_sign` $\to$ `shr.s32 %r_out, %r_biased, 1`.
  On sm_86, ptxas fuses the shift and add into a single `LEA.HI` instruction!
- Because loop bounds and strides are uniform across the CTA, ptxas scalarizes the division onto the
  uniform datapath (`UR4`, `ULEA.HI`, `USHF.R.S32.HI`), freeing vector ALUs and vector registers.

For `examples/reduce.kern`, kernc now achieves **36 executable SASS instructions and 8 registers**
(48 instructions including exit NOP bundle alignment), matching `nvcc` (37 instructions, 10 registers)
with zero `MUFU.RCP` instructions.

---

## Design decisions

These are the decisions I'd expect to defend in a review, along with the reasons behind them.

**1. Non-SSA three-address IR.** PTX already has unlimited virtual registers and no phi
nodes, and ptxas does the real register allocation. An SSA IR would therefore need an
out-of-SSA pass just to get back to PTX. The cost of skipping SSA shows up in the optimizer:
it only folds registers that are defined once (expression temporaries), and it never folds
through variables. There's no copy propagation.

**2. A register's type is fixed when the register is created.** `IRFunction::freshReg(IRType)`
records the type, and it has no default argument, so every creation site has to state one.
Originally the emitter guessed a register's class from the first instruction that used it.
An uninitialized `int` compared in a `setp` therefore became a `.pred` register, and ptxas
rejected the PTX. This is also what made shared memory easy to add: its 32-bit address
registers are just `S32`.

**3. Booleans: jumping code vs values.** In a condition, `a && b` never produces a value; it
becomes branches (the "jumping code" scheme in the Dragon Book, §6.6). As a value
(`bool ok = i < n && x[i] > 0`), it still has to short-circuit. `and.pred` would evaluate both
sides and read `x[i]` out of bounds, so kernc reuses the branches and writes true or false into
the result on each path. Plain comparisons `setp` straight into a predicate, `!` lowers to
`not.pred`, `==`/`!=` on bools lower to `xor.pred`, and `cast<int>(b)` lowers to `selp`.

**4. Constant folding is an interpreter for the target.** Folding must produce exactly what the
PTX instruction would produce at run time, not what C++ would. `cvt.rzi.s32.f32` saturates and
maps NaN to 0, so `cast<int>(1e20)` folds to INT_MAX. `.s32` arithmetic wraps, so the folder
computes in `uint32_t`. Division by zero and `INT_MIN / -1` are left for the hardware. The
`ptx::` helpers in `Passes.cpp` model one instruction each.

**5. Address arithmetic depends on the state space.** Global pointers are 64-bit: the index is
widened and scaled in one `mul.wide.s32`, then added with `add.s64`, after a one-time
`cvta.to.global`. Shared arrays are module-scope `.shared` symbols named `<kernel>$<name>`
(nvcc also puts them at module scope, under mangled names). Their addresses are 32-bit offsets into the block's shared window
(`mov.u32` of the symbol, `mul.lo.s32`, `add.s32`), which saves registers.

**6. Barriers and divergence.** `bar.sync 0` requires every thread in the block to arrive. The
reduction therefore loads with `if (i < n) v = in[i];` instead of returning early, and its loop
condition depends only on `block_dim`, so it's the same for every thread. kernc has no pass
that moves loads or stores, but any future one (CSE, hoisting) would have to treat `bar.sync`
as a fence for shared memory.

**7. Line info works like nvcc's `-lineinfo`, not `-G`.** Each IR instruction carries the
location of the statement that produced it, and a `.loc` is written only when the location
changes. ptxas carries this through optimization into the cubin's `.nv_debug_line_sass` table.
For SAXPY, `nvdisasm --print-line-info` shows (trimmed):

```
	//## File "examples/saxpy.kern", line 2
 S2R R2, SR_CTAID.X
 S2R R3, SR_TID.X
 IMAD R2, R2, c[0x0][0x0], R3
	//## File "examples/saxpy.kern", line 3
 ISETP.GE.AND P0, PT, R2, c[0x0][0x178], PT
 @P0 EXIT
	//## File "examples/saxpy.kern", line 4
 ...
 FFMA R7, R4, c[0x0][0x160], R7
 STG.E [R2.64], R7
	//## File "examples/saxpy.kern", line 6
 EXIT
```

Optimization stays on: ptxas turned the bounds check into a predicated `EXIT` and still kept
the right lines.

**8. Integer literal range follows Java.** `-2147483648` lexes as minus applied to
`2147483648`, which doesn't fit in an `int` on its own. C handles this by giving the literal a
wider type, but kernc has no wider type, so it uses Java's rule from JLS §3.10.1. The check
runs in Sema, not the lexer, because only Sema can see the surrounding unary minus.

**9. Device functions use PTX's abstract calling convention.** A call stores its arguments
into scoped `.param` variables and runs `call ($ret), f, ($arg0, ...)`; ptxas lowers this onto
the real ABI (registers, plus a stack frame when needed). Every `.func` gets a prototype at
the top of the module, because ptxas rejects a call to a later function without one ("requires
call prototype"); this also makes mutual recursion work. A `.param` can't have predicate
type, so `bool` crosses a call as a 32-bit 0/1. Pointer arguments are already global addresses,
so the callee skips `cvta`; that's only valid because every kernc pointer points to global
memory. Calls are side effects for DCE, and their arguments count as uses. Plain `call`
(not `call.uni`) is emitted, because `call.uni` would promise the call is never divergent.

**10. Generated names live where user names can't.** Every PTX name kernc invents contains
`$` (`$entry`, `$L_then_0`, `f$param_0`, `f$retval`, `k$buf`, `$arg0`). PTX identifiers allow
`$` but `.kern` identifiers don't, so a collision is impossible rather than merely unlikely.
Before this, a function named `entry` or `f_retval`, or two kernels' shared arrays, could
produce PTX that ptxas rejected. nvcc keeps its own labels apart the same way (`$L__BB0_2`).

**11. Atomic Memory Operations (`atom.global`, `atom.shared`).** Hardware atomic memory operations
enable safe concurrent read-modify-write cycles across parallel threads. `kernc` supports `add`,
`sub`, `min`, `max`, `and`, `or`, `xor`, `exch`, and `cas` on global pointers (`int*`, `float*`)
and shared memory arrays (`shared int`, `shared float`). Because the NVIDIA PTX ISA has no native
`atom.sub` instruction, `atomic_sub(p, val)` is lowered to `neg.s32` into a dedicated virtual register
followed by `atom.add.s32`. `float` memory operands are supported for `add` and `exch` (matching
hardware capabilities). Dead-code elimination (DCE) explicitly classifies all atomic operations as
side-effecting so they are never pruned even if their return value is discarded.

**12. Loop Unrolling & Basic Block Instruction Scheduling.**
Constant trip-count `for` loops ($0 < T \le 16$, e.g., `for (int i = 0; i < 4; i = i + 1)`) are automatically
detected and unrolled directly into straight-line instructions within the basic block, eliminating `bra`,
`bra.cond`, and `setp` control-flow branch overhead. The unrolled induction variables are bound to constant
registers, allowing downstream constant folding to fold element addresses (`base + i * 4`) and propagate
constants. Following DCE, a local dependency DAG-based list scheduler (`Optimizer::runInstructionScheduling`)
schedules instructions within basic blocks using latency-aware priority heuristics: global loads (base
latency 100) are hoisted early to hide the 200–400 cycle memory latency on Ampere (`sm_86`), independent ALU
operations (latency 20) are interleaved to maximize Instruction-Level Parallelism (ILP), and stores (latency 1)
are sunk to the end. The scheduler enforces strict fencing across `barrier()`, atomic operations, warp
intrinsics, calls, and branches, while respecting RAW, WAR, and WAW memory dependencies. SASS disassembly via
`nvdisasm` verifies that unrolled loops emit pipelined `LDG.E` / `LDG.E.128` instructions with zero loop branch instructions.

---

## Testing

| Suite | Count | What it checks |
|---|---|---|
| Unit tests (GoogleTest) | 103 | Lexer 9, parser 15, Sema 34 (accept/reject, type/arity rules, diagnostic locations), IR, passes, scheduling & PTX 45 |
| GPU end-to-end (Driver API) | 27 | Each kernel is JIT-loaded with `cuModuleLoadDataEx`, launched, and compared with a CPU reference (details below) |
| Diagnostics | 1 suite, 8 files | `tests/diagnostic/err_*.kern` must fail with the expected message (and location, where given) |
| ptxas acceptance | 6 | Every file in `examples/` assembles with `ptxas -arch=sm_86` |
| ptxas regression | 24 | `tests/regress/*.kern` (12 kernels) assemble at both `-O0` and `-O1` |
| Line tables | 2 | `kernc -lineinfo` → ptxas → `nvdisasm --print-line-info`; each `//@ REGEX` line must own matching SASS |
| Runner CLI e2e | 1 suite, 21 stages | `kernc-runner` driver invocation, argument marshaling, device execution, error validation, and `--arg` type checking against the PTX `.param` types |

The GPU tests cover:

- vector add, SAXPY, clamp and ReLU at sizes that aren't block multiples (1000, 117, 33, 1);
- an early-return guard where the extra threads must not write anything;
- a `return` inside a loop;
- 13 kinds of `bool` expression, with NaN and `-0.0` inputs;
- a block reverse through shared memory;
- the shipped reduction example, for 1 to 2²⁰+3 elements;
- warp-level tree reduction (`__shfl_down_sync` / `shfl_down_sync` with offsets 16, 8, 4, 2, 1), butterfly exchange (`__shfl_xor_sync` / `shfl_xor_sync`), broadcast (`__shfl_sync` / `shfl_sync`), warp ballot (`__ballot_sync` bool predicate, `ballot_sync` int predicate), active mask (`__activemask` / `activemask`), and shift up (`__shfl_up_sync` / `shfl_up_sync`) for `int` and `float` at `-O0` and `-O1` vs CPU reference;
- atomic memory operations under heavy contention across 4096 parallel threads (16 blocks of 256 threads) verifying exact global and shared memory results (`atomic_add` int/float, `atomic_sub`, `atomic_min`, `atomic_max`, `atomic_or`, `atomic_and`, `atomic_xor`, `atomic_exch` int/float, and single-winner `atomic_cas`) at `-O0` and `-O1` vs CPU reference;
- device functions (`examples/functions.kern`), with recursion, mutual recursion, float, bool, pointer and void calls at `-O0` and `-O1`;
- functions and arrays named like kernc's old generated PTX names (`entry`, `L_then_0`,
  `f_retval`, `param0`, `k_buf`, …), which must compile and call the right function;
- PTX with line info under `CU_JIT_GENERATE_LINE_INFO`;
- a differential test where `-O0` and `-O1` must give bit-identical results on edge-case
  constants;
- a differential test where `-O0` (GPU `div.s32`/`rem.s32`) and `-O1` (strength-reduced shifts/masks)
  must give bit-identical results across positive, negative, zero, and `INT_MIN` values for `/ 2^k` and `% 2^k`;
- loop unrolling and latency-hiding instruction scheduling across a 5-point stencil filter, 8-element dot-product reduction, and pipelined multi-stream load/ALU/store sequences at `-O0` and `-O1` vs CPU reference;
- the scheduler keeping a store before a load when memory may alias: two kernel pointer
  parameters (or two device-function parameters) given the same buffer, and one thread
  storing and then loading the same shared element;
- vector intrinsic offsets counted in whole vectors (`load_int4(p, 1)` reads `p[4..7]`) on
  global and shared scalar buffers, without misaligned-address faults.

Two techniques do most of the work:

- **Differential testing.** `-O0` vs `-O1` on the GPU needs no hand-written expected values,
  because the GPU itself decides what's correct.
- **Mutation checks.** After each fix, I reverted it on purpose to confirm a test fails:
  - the old float-to-int fold fails the differential test;
  - `setp.ne` instead of `setp.neu` fails the NaN case;
  - `shfl.sync.up` instead of `shfl.sync.down` fails the warp tree reduction test;
  - inverting list scheduling priority (stores high, loads low) fails the load-hoisting latency check (`lastLoadIdx < firstStoreIdx`, on two different shared arrays);
  - removing `bar.sync` fails the block-reverse test, first at thread 64, the first thread of
    the third warp, which read data before its partner warp had written it;
  - removing statement locations fails all six line-table checks;
  - omitting the sign bias in power-of-two division fails the differential test on negative integers.

### Bugs the tests found

The first version passed its own 37 tests, but checks against the GPU, ptxas and UBSan found
these bugs:

| Bug | Effect |
|---|---|
| Register class inferred from the first use | ptxas rejected kernels that read an uninitialized variable in `setp`, `mul.wide` or `st` |
| `cast<int>(1e20)` folded with C++ semantics | Gave INT_MIN at `-O1` but INT_MAX at `-O0` on the GPU |
| `INT_MIN / -1` folded on the host | **kernc itself crashed** with SIGFPE |
| Signed overflow while folding | Undefined behaviour inside the compiler; it happened to wrap |
| `if (v % 2)` treated as a comparison | **Compiled as `v == 2`**, and the `%` was never computed |
| Float `!=` lowered to `setp.ne.f32` | `x != x` was false for NaN |
| Comparisons used as values | Lowered to `add.s32` (hidden until `bool` existed) |
| Out-of-range literals | `3000000000` was silently truncated |
| Generated PTX names shared the user namespace | A function named `entry`, `f_retval` or `k_buf` (or two kernels' arrays, `k`+`b_uf` vs `k_b`+`uf`) made ptxas reject a valid program |
| Scheduler treated different base pointers as non-aliasing | **Wrong results at `-O1`**: with `a == b`, `a[i] = 42; out[i] = b[i];` read the stale value, because the load moved above the store (two accesses to one shared array too) |
| Shift folding masked the count to 5 bits (x86) | `1 << 32` folded to 1 and `1000 >> 40` to 3, but PTX clamps the count, so the GPU gives 0 at `-O0` |
| Vector intrinsic offsets counted scalars | `load_int4(p, 1)` issued a 16-byte load at a 4-byte boundary: **`CUDA_ERROR_MISALIGNED_ADDRESS`** |
| Atomics accepted vector targets as int | `atomic_add` on a `float4*` emitted an integer `atom.add.s32` on float data |
| `kernc-runner` checked only the argument count | An `i32` for a pointer faulted with `CUDA_ERROR_ILLEGAL_ADDRESS`; an `i32` for a float or a buffer for an int printed **`Success!` with wrong results** |

Every fix is in the commit history, together with a test that failed before the fix.

---

## Limitations and next steps

- **Strength reduction for non-power-of-two constants.** Signed `/` and `%` by $2^k$ are
  already strength-reduced to branchless shift/add/mask sequences (see above). Dividing by
  arbitrary invariant constants via fixed-point multiplication (Granlund-Montgomery / magic
  numbers) is a natural next step.
- **No CSE or copy propagation in kernc.** For SAXPY ptxas makes up for it, as shown above,
  but kernc's own PTX still recomputes addresses and reloads constants.
- **No inlining.** ptxas keeps every device-function call as a real `CALL` with the ABI's
  overhead: in `examples/functions.kern` it inlined none of the 7 functions, not even
  `square`. nvcc's front end inlines before emitting PTX, so this is front-end work, like
  strength reduction.
- **No unreachable-block removal.** Code after `return` leaves an unreachable `$L_after_ret`
  block in the PTX; ptxas removes it.
- **No definite-assignment analysis.** Reading an uninitialized variable produces valid PTX
  but an undefined value, as in C, and kernc doesn't warn about it.
- **No barrier-divergence warning.** A `barrier()` behind a thread-dependent condition (or
  after an early `return` taken by some threads) isn't diagnosed.
- **Not yet supported:** structs, 64-bit types other than pointers, and
  `bool` kernel parameters.
- **Line info is per statement**, not per expression. Also, `-(2147483648)` is accepted
  because parentheses aren't kept in the AST; Java rejects it.
- **compute-sanitizer can't run under WSL here** until the Windows debugger interface is
  enabled (run `EnableDebuggerInterface.bat` as administrator). With it, memcheck could report
  faults against `.kern` lines using `-lineinfo`.

---

## Repository layout

```
include/kernc/       public headers: Token, Lexer, AST, Parser, Sema, IR, IRGen, Passes, PTXEmitter
src/                 implementation, plus main.cpp (the kernc CLI)
examples/            vecadd, saxpy, clamp, relu, reduce (shared-memory reduction)
tests/unit/          GoogleTest: lexer, parser, Sema, IR and PTX emission
tests/e2e/           GPU tests through the CUDA Driver API
tests/diagnostic/    err_*.kern files and the runner that checks their messages
tests/regress/       kernels assembled by ptxas at -O0 and -O1
tests/lineinfo/      line-table kernels and check_lineinfo.py (ptxas + nvdisasm)
tools/runner/        kernc-runner (CLI kernel execution via CUDA Driver API)
docs/specs/          scope, architecture, IR and codegen, testing strategy
```

