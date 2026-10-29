# `kernc` Specification 03: Intermediate Representation & PTX Emission

## Intermediate Representation (IR)

The IR is a non-SSA 3-address virtual-register code organized into Basic Blocks forming a Control Flow Graph (CFG).

### Key Decisions
1. **Non-SSA Representation**:
   - PTX is itself a virtual register architecture with infinite register capability.
   - `ptxas` executes SSA conversion, dominance frontier calculation, and register allocation internally.
   - Avoiding SSA in the front-end simplifies code generation while letting `ptxas` do backend optimization.

2. **Short-Circuit Branch Threading**:
   - Logical `&&` and `||` expressions are lowered directly into control-flow branches rather than materializing boolean temporaries.
   - `emitCondition(a && b, L_true, L_false)` lowers `a` branching to `L_rhs` on true, and `L_false` on false. `L_rhs` evaluates `b` and branches to `L_true` or `L_false`.

3. **64-bit Address Computation**:
   - GPU global pointers in `.address_size 64` require 64-bit effective addresses.
   - Array indexing `ptr[i]` generates:
     ```ptx
     mov.s32       %r_sz, 4;
     mul.wide.s32  %rd_off, %r_i, %r_sz;
     add.s64       %rd_addr, %rd_base, %rd_off;
     ```
   - `mul.wide.s32` multiplies two 32-bit registers directly into a 64-bit register, matching `nvcc`'s emitted address arithmetic.

4. **Parameter Loading in Prologue**:
   - Pointers passed in `.param .u64` are converted from generic/parameter space to global space via `cvta.to.global.u64`.
   - Scalars are loaded directly with `ld.param.s32` or `ld.param.f32`.

5. **Float Literals in PTX**:
   - Emits IEEE-754 bit representations as hex literals `0fXXXXXXXX` (e.g., `0f00000000` for `0.0f`).

6. **Source Line Info (`-lineinfo`)**:
   - Every IR instruction carries the source location of the statement that produced it.
     IRGen sets `IRFunction::currentLoc` per statement and restores it afterwards, so the
     branches closing an `if`/loop map to that statement. Loop conditions (emitted after the
     body) map to the condition expression. The parameter prologue maps to the `kernel` line
     and the final `ret` to the closing brace, as with nvcc.
   - With `-lineinfo` (or `--generate-line-info`), the emitter writes `.file 1 "<input>"` once
     and a `.loc 1 <line> <column>` whenever the location changes. A `.loc` applies to the
     instructions that follow it in the text, so labels do not reset it.
   - ptxas turns these into the cubin's `.nv_debug_line_sass` table, which nvdisasm,
     cuda-gdb, compute-sanitizer and Nsight use to map SASS back to `.kern` lines. Like nvcc's
     `-lineinfo` (and unlike `-G`), this does not disable optimization. Off by default; without
     it the PTX is unchanged.

7. **Device Functions and Calling Convention**:
   - Forward Prototypes: Every `.func` has a forward prototype emitted at module scope
     before any function definitions (`.func (.param .<t> f$retval) f(.param .<t> f$param_0, ...);`).
     This allows calls to functions defined later in the file without requiring forward declarations.
   - Call Sites: Scoped brace blocks declare `.param` registers, store arguments with `st.param`,
     execute a non-uniform `call` (allowing divergent branches), and load the return value with `ld.param`:
     ```ptx
     {
         .param .f32 $arg0;
         st.param.f32 [$arg0], %f1;
         .param .f32 $ret;
         call ($ret), use, ($arg0);
         ld.param.f32 %f0, [$ret];
     }
     ```
   - ABI Types & Parameter Loading:
     - Pointer parameters to `.func` are loaded with `ld.param.b64` and receive already-converted
       64-bit global addresses (no `cvta.to.global`).
     - `bool` crosses the ABI as a 32-bit integer (`.b32` 0/1); converted via `selp` before passing/returning
       and `setp.ne` against 0 after receiving.
     - Return statements in non-void device functions emit `IROp::StoreRetval` (`st.param.<t> [f$retval], %r`)
       followed by `IROp::Ret`.
   - Optimization Integration:
     - Dead Code Elimination (DCE) treats `IROp::Call` and `IROp::StoreRetval` as side-effecting operations.
     - Register use counting in DCE includes registers in `IRInst::args`.
     - Register pre-allocation in `PTXEmitter` covers all registers in `inst.args`.

8. **Generated PTX Names**:
   - Every name kernc invents contains `$`, which PTX identifiers allow and `.kern` identifiers
     do not, so it can never collide with a kernel, function or array name from the source.
     The helpers live in `ptxname::` (`include/kernc/IR.h`):

     | Generated name | Form |
     |---|---|
     | Block labels | `$entry`, `$L_then_0`, `$L_for_cond_2`, ... |
     | Parameters | `<fn>$param_<i>` |
     | Return value | `<fn>$retval` |
     | Shared arrays | `<kernel>$<array>` (a second same-named array in the kernel gets `$<n>`) |
     | Call-site temporaries | `$arg<i>`, `$ret` |

   - `tests/regress/name_collisions.kern` uses source names that match the old, underscore-based
     scheme (`entry`, `L_then_0`, `f_retval`, `param0`, `k_buf`, ...); ptxas and a GPU test check it.

9. **Strength Reduction for Integer Division and Modulo**:
   - `ptxas` treats `div.s32` and `rem.s32` literally: when emitted in PTX, `ptxas` lowers them to an iterative ~26-instruction Newton-Raphson approximation sequence using `MUFU.RCP`, even when the divisor is a constant.
   - `kernc`'s optimizer (`Optimizer::runConstantFolding`) lowers signed `/` and `%` by any constant $d \notin \{0, \pm 1, \mathrm{INT\_MIN}\}$ into branchless sequences that truncate toward zero:
     - **Constant power-of-two divisors ($2^k$, $k \in [1, 30]$)**:
       - For $k = 1$ ($2^1 = 2$):
         ```ptx
         shr.u32 %r_sign, %r_x, 31;
         add.s32 %r_biased, %r_x, %r_sign;
         shr.s32 %r_out, %r_biased, 1;
         ```
         On sm_86, `ptxas` fuses `shr.u32` and `add.s32` into a single `LEA.HI` instruction, taking only 2 SASS instructions total.
       - For $k \in [2, 30]$ ($2^k$):
         ```ptx
         shr.s32 %r_mask, %r_x, 31;
         shr.u32 %r_bias, %r_mask, (32 - k);
         add.s32 %r_biased, %r_x, %r_bias;
         shr.s32 %r_out, %r_biased, k;
         ```
       - For signed modulo $x \pmod{2^k}$, the biased dividend is computed identically, masked with `-(1 << k)` to clear the lowest $k$ bits, and subtracted from $x$:
         ```ptx
         and.b32 %r_masked, %r_biased, -(1 << k);
         sub.s32 %r_out, %r_x, %r_masked;
         ```
     - **Non-power-of-two (and negative) constant divisors ($d \notin \{0, \pm 1, \mathrm{INT\_MIN}\}$, $d \ne 2^k$)**:
       - Lowered using the Granlund–Montgomery / Hacker's Delight magic-number algorithm (`computeSignedMagic(|d|)`):
         1. Compute magic multiplier $M$ and post-shift $s$ for $|d|$.
         2. Emit high 32 bits of 64-bit product: `mul.hi.s32 %q0, %x, M;` (`IMAD.HI` in SASS).
         3. If $M < 0$: `add.s32 %q1, %q0, %x;` (compensates for negative sign bit in multiplier).
         4. If $s > 0$: `shr.s32 %q2, %q1, s;`
         5. Extract dividend sign bit: `shr.u32 %r_sign, %x, 31;`
         6. Add sign bit for truncation toward zero: `add.s32 %q_pos, %q2, %r_sign;`
         7. If $d < 0$: `neg.s32 %q, %q_pos;`
         8. For modulo $x \pmod d$: compute $q$, then `mul.lo.s32 %prod, %q, d;` and `sub.s32 %r, %x, %prod;`.
       - Eliminates all 26 instructions of the iterative `MUFU.RCP` reciprocal sequence, reducing kernel SASS to 9 instructions (over 8x reduction in division core, 0 `MUFU.RCP`).
     - Identity cases: $x / 1 \to x$ (`mov`), $x \% 1 \to 0$ (`ldimm 0`).
   - Differential testing (`StrengthReducedDivModMatchesUnoptimizedGPU` and `NonPowerOfTwoDivModMatchesUnoptimizedGPUAndCpu`) runs across positive, negative, zero, and `INT_MIN` inputs against unoptimized `-O0` GPU execution and CPU references, verifying bit-identical output.

10. **Bitwise Operators and Algebraic Identities**:
    - The bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`) operate on 32-bit `int` registers.
    - IR lowering and PTX emission:
      - `BitAnd` -> `IROp::And` -> `and.b32`
      - `BitOr` -> `IROp::Or` -> `or.b32`
      - `BitXor` -> `IROp::Xor` -> `xor.b32`
      - `BitNot` -> `IROp::Not` -> `not.b32`
      - `Shl` -> `IROp::Shl` -> `shl.b32`
      - `Shr` -> `IROp::ShrS32` -> `shr.s32` (signed arithmetic shift)
    - Target mapping: on Ampere sm_86, `ptxas` maps `and.b32`, `or.b32`, `xor.b32`, and `not.b32` to single-cycle `LOP3.LUT` instructions, and `shl.b32`/`shr.s32` to `SHF` instructions.
    - Optimizer algebraic simplifications:
      - Annihilators: `x & 0 = 0`, `x | -1 = -1`.
      - Neutral elements: `x & -1 = x`, `x | 0 = x`, `x ^ 0 = x`, `x << 0 = x`, `x >> 0 = x`.
      - Self-cancellation: `x ^ x = 0`, `x - x = 0`.
      - Constant evaluation: `Passes.cpp` folds bitwise operations with full 32-bit unsigned wrapping semantics (`uint32_t`), ensuring identical behavior across host compilation and GPU target execution.

11. **Multi-Dimensional Execution Geometry (2D & 3D Built-ins)**:
    - Special registers in PTX are 32-bit unsigned registers read via `mov.u32 %r, %special;`:
      - Thread indices: `%tid.x`, `%tid.y`, `%tid.z`
      - Block indices: `%ctaid.x`, `%ctaid.y`, `%ctaid.z`
      - Block dimensions: `%ntid.x`, `%ntid.y`, `%ntid.z`
      - Grid dimensions: `%nctaid.x`, `%nctaid.y`, `%nctaid.z`
    - Target mapping in SASS (sm_86):
      - `%tid` and `%ctaid` map to hardware special registers read by the `S2R` (Special Register to Register) instruction (e.g. `S2R R0, SR_TID.X`, `SR_CTAID.Y`).
      - `%ntid` and `%nctaid` represent kernel launch metadata passed via Driver API launch parameters; `ptxas` resolves them from Constant Memory Bank 0 (`c[0x0]`) or uniform registers without consuming special register file capacity.
    - `kernc-runner` accepts multi-dimensional geometry syntax (`--grid 4x4`, `--grid 2,3,4`, `--block 16x16`, `--block 4,2,2`), launching via `cuLaunchKernel` across all 6 grid/block dimensions.

12. **Warp-Level Intrinsics (Data Exchange & Synchronization)**:
    - Intra-warp communication without shared memory via PTX `shfl.sync` and `vote.sync` instructions.
    - IR Opcode mapping:
      - `__shfl_sync` / `shfl_sync` -> `IROp::ShflIdx` -> `shfl.sync.idx.b32 dst, val, src_lane, 31, mask;`
      - `__shfl_down_sync` / `shfl_down_sync` -> `IROp::ShflDown` -> `shfl.sync.down.b32 dst, val, delta, 31, mask;`
      - `__shfl_up_sync` / `shfl_up_sync` -> `IROp::ShflUp` -> `shfl.sync.up.b32 dst, val, delta, 0, mask;`
      - `__shfl_xor_sync` / `shfl_xor_sync` -> `IROp::ShflBfly` -> `shfl.sync.bfly.b32 dst, val, lane_mask, 31, mask;`
      - `__ballot_sync` / `ballot_sync` -> `IROp::VoteBallot` -> `vote.sync.ballot.b32 dst, pred, mask;`
      - `__activemask` / `activemask` -> `IROp::Activemask` -> `activemask.b32 dst;`
    - Operand types and clamp values:
      - `ptxas` on `sm_86` accepts both 32-bit integer (`.s32`) and 32-bit float (`.f32`) registers as operands to `shfl.sync.*.b32`.
      - Clamp values match CUDA hardware semantics: clamp `31` for index, down, and butterfly shuffle (clamping upper bound to warp size - 1); clamp `0` for up shuffle (threads with source lane < 0 receive their own unmodified register value).
      - Predicates passed to `vote.sync.ballot` are converted to `.pred` registers via `emitAsPred`.
    - Optimization and Synchronization Semantics:
      - `shfl.sync` and `vote.sync` are intra-warp synchronization instructions requiring all non-exited threads in `mask` to arrive before executing. Dead Code Elimination (DCE) explicitly classifies them as side-effecting operations to guarantee that unused shuffle results do not compromise warp synchronization convergence.
      - In contrast, `activemask` is a pure query of the active thread mask with no cross-thread synchronization barrier; dead definitions of `activemask` are cleanly eliminated by DCE when its result register is unused.
      - Register scanner in `PTXEmitter` and register use counter in `Passes` account for 3-input operands via `IRInst::src3`.
      - Default clamps (31 for idx/down/bfly, 0 for up) are stored on `IRInst::immInt` upon instruction construction and emitted directly into PTX.

13. **Atomic Memory Operations (`atom.global`, `atom.shared`)**:
    - Hardware atomic read-modify-write memory transactions across global and shared address spaces.
    - IR Opcode mapping:
      - `atomic_add` -> `IROp::AtomicAdd` -> `atom.{global|shared}.add.{s32|f32} dst, [addr], val;`
      - `atomic_sub` -> `IROp::AtomicSub` -> `neg.s32 %scratch, val; atom.{global|shared}.add.s32 dst, [addr], %scratch;`
      - `atomic_min` -> `IROp::AtomicMin` -> `atom.{global|shared}.min.s32 dst, [addr], val;`
      - `atomic_max` -> `IROp::AtomicMax` -> `atom.{global|shared}.max.s32 dst, [addr], val;`
      - `atomic_and` -> `IROp::AtomicAnd` -> `atom.{global|shared}.and.b32 dst, [addr], val;`
      - `atomic_or`  -> `IROp::AtomicOr`  -> `atom.{global|shared}.or.b32 dst, [addr], val;`
      - `atomic_xor` -> `IROp::AtomicXor` -> `atom.{global|shared}.xor.b32 dst, [addr], val;`
      - `atomic_exch`-> `IROp::AtomicExch`-> `atom.{global|shared}.exch.b32 dst, [addr], val;`
      - `atomic_cas` -> `IROp::AtomicCas` -> `atom.{global|shared}.cas.b32 dst, [addr], cmp, val;`
    - Lowering Architecture and Addressing:
      - Address generation is unified in `emitMemoryAddress`:
        - Global memory pointers use 64-bit addresses: widened index multiplication `mul.wide.s32` followed by 64-bit addition `add.s64` onto the base pointer `%rd`.
        - Shared memory arrays use 32-bit window offsets: base symbol resolution `mov.u32` followed by 32-bit arithmetic `mul.lo.s32` and `add.s32`.
      - Return Value: Every PTX `atom` instruction writes the old value at the target memory location into `dst`. `kernc` assigns this destination register to the calling expression's result.
      - PTX ISA `atom.sub` handling: PTX ISA lacks a native `atom.sub` instruction. `PTXEmitter` detects `IROp::AtomicSub` and allocates a dedicated 32-bit scratch register `%r` to perform `neg.s32` before issuing `atom.add.s32`.
      - Optimization semantics: In `Passes.cpp`, all atomic operations are explicitly classified as side-effecting memory operations. Dead Code Elimination (DCE) never eliminates atomic instructions even when their destination register has a use count of zero.

14. **Vector Memory Operations and Types**:
    - Supported Types: `int2`, `int4`, `float2`, `float4`, pointers (`int2*`, etc.), and shared arrays (`shared int2`, etc.).
    - Local Vector Variable Lowering:
      - Vector local variables are represented as a contiguous array of scalar virtual registers in `VarInfo::regs`.
      - Component extraction (`.x`, `.y`, `.z`, `.w`) and component mutation are zero-cost register selections / moves, eliminating vector packing and unpacking instructions.
    - IR Opcodes and Structure:
      - `LoadV2`, `LoadV4`: Load 2 or 4 elements from global or shared memory into multiple destination virtual registers stored in `inst.dsts`.
      - `StoreV2`, `StoreV4`: Store 2 or 4 elements from arguments stored in `inst.args` into global or shared memory.
    - Addressing and Alignment:
      - Vector global addresses scale the index by the vector size (8 bytes for v2, 16 bytes for v4) using 64-bit address arithmetic (`mul.wide.s32` and `add.s64`). The `load_*`/`store_*` intrinsics scale their offset the same way, by the size of the vector they access, whatever the buffer's element type, so every v2/v4 access is naturally aligned.
      - Every shared array is emitted with `.shared .align 16`, so the v4 intrinsics also work on scalar shared arrays.
    - PTX Lowering:
      - Lowers directly to PTX vector load/store instructions:
        - `ld.{global|shared}.v2.{s32|f32} {%d0, %d1}, [addr];`
        - `ld.{global|shared}.v4.{s32|f32} {%d0, %d1, %d2, %d3}, [addr];`
        - `st.{global|shared}.v2.{s32|f32} [addr], {%s0, %s1};`
        - `st.{global|shared}.v4.{s32|f32} [addr], {%s0, %s1, %s2, %s3};`
    - SASS Hardware Lowering on `sm_86`:
      - `ptxas` lowers `LoadV4`/`StoreV4` to 128-bit memory instructions: `LDG.E.128` / `STG.E.128` (global) and `LDS.128` / `STS.128` (shared).
      - `ptxas` lowers `LoadV2`/`StoreV2` to 64-bit memory instructions: `LDG.E.64` / `STG.E.64` (global) and `LDS.64` / `STS.64` (shared).
    - Optimizer (DCE) Integration:
      - `StoreV2` and `StoreV4` are classified as side-effecting memory operations and are never removed by DCE.
      - `LoadV2` and `LoadV4` track use counts across all destination registers in `inst.dsts`. A vector load is pruned by DCE if and only if every destination register is unused (`useCount == 0`). If at least one component is live, the vector load is preserved.

15. **Loop Unrolling and Basic Block Instruction Scheduling**:
    - **Constant Trip-Count Detection & Loop Unrolling**:
      - `IRGen::emitFor` detects canonical loops with constant bounds and step (`for (int i = C_start; i < C_end; i = i + C_step)`).
      - Computes constant trip count $T = \lfloor (C_{end} - C_{start} + C_{step} - 1) / C_{step} \rfloor$.
      - When $0 < T \le 16$, `IRGen` completely unrolls the loop $T$ times inside the current basic block by binding the induction variable to constant values ($C_{start} + k \cdot C_{step}$) loaded via `LoadImm`.
      - Completely eliminates loop branch overhead (`bra`, `bra.cond`, `setp`) and extra basic blocks (`L_for_cond`, `L_for_body`, `L_for_end`).
      - Downstream constant folding immediately propagates the induction constant into array indexing (`base + (C_start + k*C_step)*4`), folding `MulWide` and `AddS64` into immediate offsets and `Mov`.
    - **Local Dependency DAG Construction**:
      - Instructions within basic blocks are scheduled between fences/barriers (`barrier()`, atomics, shuffles, calls, and terminators).
      - Tracks register RAW (dataflow), WAR (anti-dependency), and WAW (output dependency).
      - Memory aliasing: a load and a store in the same state space are ordered unless they provably touch different memory. The only provable case is two *different shared arrays* (identified by symbol, since every access computes its own base register). Global pointers, whether kernel or device-function parameters, may alias each other as in C/CUDA without `__restrict__`, so their loads never move above their stores. (An earlier version treated distinct base registers as disjoint, which miscompiled `a[i] = 42; out[i] = b[i];` with `a == b`, and even two accesses to one shared array.)
      - Sequential store order is strictly preserved (Store -> Store edges).
    - **Latency-Aware List Scheduling**:
      - Base instruction latencies on sm_86: Global Loads = 100, Shared Loads = 80, Parameter/SReg/Immediate reads = 50, Address computations = 40, ALU/conversions = 20, Stores = 1.
      - Calculates critical path depth via DAG height.
      - Hoists memory loads early to hide memory latency (200-400 cycles on Ampere), interleaves ALU operations to saturate compute pipelines, and sinks stores to the end of the block.
      - Verified on SASS disassembly via `nvdisasm`: eliminates scoreboard stalls by scheduling all `LDG.E` / `LDG.E.128` instructions concurrently before dependent ALU and stores.

16. **Inter-Procedural Inlining of Small, Non-Recursive Device Functions**:
    - **Call Graph & Recursion Analysis**:
      - `CallGraph::build` scans every function block and maps caller functions to called callee symbols.
      - `CallGraph::isRecursive` identifies functions participating in directed cycles (self-recursion like `fib` and mutual recursion like `is_even` $\leftrightarrow$ `is_odd`) via BFS reachability from direct callees.
      - Recursive functions are never inlined, preventing infinite inlining loops.
    - **Inlining Eligibility**:
      - `!fn.isKernel`: kernels cannot be inlined into callers.
      - Non-recursive: `!cg.isRecursive(fn.name)`.
      - Size threshold: total IR instruction count $\le 64$ instructions.
      - Shared memory safety: functions declaring local `.shared` arrays (`!fn.sharedVars.empty()`) are preserved out-of-line to avoid multi-call aliasing.
    - **IR-Level Transformation Mechanics**:
      - Fresh register allocation (`vregMap`) and basic block label remapping (`labelMap`) ensure zero collision with caller virtual registers and block names.
      - Basic block splitting: caller block containing `IROp::Call` is split into `headBB` and continuation `tailBB` (`contLabel`).
      - Inlined body insertion: callee blocks are inserted between `headBB` and `tailBB`.
      - Parameter moves: callee `IROp::LoadParam` instructions become register `Mov` operations.
      - ABI conversion elimination:
        - Bool parameters: caller's `CvtPredToS32` and callee's `SetpNe` against 0 are eliminated; argument predicates move directly into callee predicate registers.
        - Bool returns: callee's `CvtPredToS32` and caller's `SetpNe` against 0 are eliminated; callee return predicates move directly into caller predicate destinations.
        - Pointer parameters: 64-bit generic/global addresses are passed directly via `mov.u64`, bypassing redundant `cvta.to.global`.
      - Continuation threading: callee `StoreRetval` becomes a `Mov` into the caller result register, and callee `Ret` becomes `Bra contLabel`.
      - Line info preservation: every inlined instruction maintains its original `srcLoc` referencing source lines in the callee definition.

17. **Local Common Subexpression Elimination (CSE) & Copy Propagation**:
    - **Local Value Numbering (LVN) Architecture**:
      - `Optimizer::runCSE` runs per basic block within the iterative optimization loop alongside constant folding and dead code elimination until convergence.
      - Maintains an available expressions table (`availExprs`) and a copy-propagation map (`copyMap`).
      - Identifies pure arithmetic operations, constant loading (`LoadImm`), special register reads (`ReadSReg`), address computations (`MulWide`, `AddS64`, `CvtaGlobal`, `SharedAddr`), parameter loads (`LoadParam`), and memory loads (`Load`).
      - Commutative operation canonicalization: for commutative operations (`Add`, `Mul`, `And`, `Or`, `Xor`, `FAdd`, `FMul`, `AddS64`, `MulWide`, `SetpEq`, `SetpNe`, `SetpFEq`, `SetpFNe`, `XorPred`), operands are normalized such that $src_1 \le src_2$.
    - **Non-SSA Definition Invalidation (Kill Semantics)**:
      - The IR is non-SSA three-address code where virtual registers can be reassigned.
      - When an instruction defines register $r_{dst}$, any available expression containing $r_{dst}$ as an operand is killed (since its operand value changed).
      - Any expression whose result was stored in $r_{dst}$ is killed (since $r_{dst}$ no longer holds that value).
      - All copy mappings with $r_{dst}$ as key or value are invalidated.
      - Self-referencing instructions ($r_{dst} = op(r_{dst}, \dots)$) are never inserted into available expressions.
    - **Memory Aliasing and Fence Invalidation (Correctness Rule 2)**:
      - Global memory stores (`Store`, `StoreV2`, `StoreV4` with `AddrSpace::Global`) invalidate all global load expressions.
      - Shared memory stores invalidate loads from the same shared array symbol or unknown shared arrays, while provably disjoint shared arrays remain available.
      - Memory synchronization barriers (`bar.sync`), atomics (`Atomic*`), warp intrinsics (`Shfl*`, `VoteBallot`, `Activemask`), and function calls (`Call`) act as complete memory fences, invalidating all available memory loads.
    - **Copy Propagation & DCE Synergy**:
      - Available subexpressions are rewritten as `Mov dst, availReg` and recorded in `copyMap`.
      - Source registers in subsequent instructions are rewritten to their root registers.
      - Redundant calculations into the same destination register (`dst == availReg`) and self-moves (`Mov dst, dst`) are eliminated immediately.
      - Dead Code Elimination removes all intermediate `Mov` instructions whose definitions have zero remaining uses.

18. **Control Flow Graph (CFG) Cleanup & Branch Optimization**:
    - **Reachability Analysis (Unreachable Block Elimination)**:
      - Traverses the function's CFG via worklist graph traversal starting from entry block `fn.blocks[0]`.
      - Identifies and prunes dead basic blocks created after unconditional returns (e.g., `$L_after_ret_N`) and disconnected subgraphs (such as dead loops or unreachable mutual branch cycles).
    - **Physical Fallthrough Branch Elimination**:
      - Inspects basic blocks ending with an unconditional branch `IROp::Bra`.
      - If the branch target is identical to the label of the immediate physically subsequent basic block in module layout (`trueTarget == nextBb.label`), the branch instruction is eliminated, enabling natural zero-overhead physical fallthrough in emitted PTX.
    - **Jump Trampoline Redirection**:
      - Identifies forwarding blocks consisting solely of an unconditional jump (`bra target`).
      - Rewrites branch targets across all predecessor branch instructions (`Bra`, `BraCond`) to jump directly to the ultimate target, eliminating unnecessary control-flow hops.
    - **Single-Predecessor Block Merging**:
      - Identifies adjacent basic blocks $A$ and $B$ where block $A$ unconditionally flows into $B$ and $B$ has an in-degree of exactly 1 ($\mathrm{preds}(B) = \{A\}$).
      - Concatenates the instructions of $B$ into $A$ and deletes block $B$.
      - By coalescing fragmented blocks into larger straight-line basic blocks, CFG cleanup directly expands the optimization scope for subsequent iterations of Constant Folding and Local Common Subexpression Elimination (CSE).
