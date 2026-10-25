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
