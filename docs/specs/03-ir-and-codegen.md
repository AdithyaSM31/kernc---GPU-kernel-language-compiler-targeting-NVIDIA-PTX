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

9. **Strength Reduction for Integer Division and Modulo by $2^k$**:
   - `ptxas` treats `div.s32` and `rem.s32` literally: when emitted in PTX, `ptxas` lowers them to an iterative ~20-instruction Newton-Raphson approximation sequence using `MUFU.RCP`, even when the divisor is a constant power of two.
   - `kernc`'s optimizer (`Optimizer::runConstantFolding`) lowers signed `/` and `%` by a constant power of two ($2^k$, $k \in [1, 30]$) into branchless shift/add/mask arithmetic that truncates toward zero:
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
     - Identity cases: $x / 1 \to x$ (`mov`), $x \% 1 \to 0$ (`ldimm 0`).
   - Differential testing (`StrengthReducedDivModMatchesUnoptimizedGPU`) runs across positive, negative, zero, and `INT_MIN` inputs against unoptimized `-O0` GPU execution, verifying bit-identical output.

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

