# `kernc` Specification 01: Language Scope & Syntax

## Overview
`kernc` is a standalone C++17 compiler that lowers a specialized GPU kernel language (`.kern`) to NVIDIA PTX (`sm_86`, Ampere architecture). The generated PTX code is assembled via `ptxas` or loaded at runtime via the CUDA Driver API (`cuModuleLoadDataEx`).

## Language Features

### Data Types
- `int`: 32-bit signed integer (`.s32` in PTX).
- `float`: 32-bit IEEE-754 single-precision floating point (`.f32` in PTX).
- Vector types:
  - `int2`: 2-element vector of `int` (8 bytes, 8-byte aligned).
  - `int4`: 4-element vector of `int` (16 bytes, 16-byte aligned).
  - `float2`: 2-element vector of `float` (8 bytes, 8-byte aligned).
  - `float4`: 4-element vector of `float` (16 bytes, 16-byte aligned).
- Pointers: `int*`, `float*`, `int2*`, `int4*`, `float2*`, `float4*` (valid as kernel and device function parameters). Pointer indexing scales by the pointee size (4, 8, or 16 bytes).
- Shared arrays: `shared int`, `shared float`, `shared int2`, `shared int4`, `shared float2`, `shared float4`. Declared with explicit static size `[N]`. Every shared array is aligned to 16 bytes (`.align 16`).
- `bool`: `true` / `false`, held in `.pred` registers. Comparisons and `!`, `&&`, `||` produce
  `bool`. It does not convert implicitly to or from `int`/`float` (use `cast<>`), cannot be a
  kernel parameter (`.param` has no predicate type; pass an `int`), and has no pointer type.
  Conditions (`if`, `while`, `for`, `!`, `&&`, `||`) accept `bool` or `int` (non-zero is true).

### Vector Component Access & Constructors
- Component access:
  - 2-element vectors support `.x` and `.y`.
  - 4-element vectors support `.x`, `.y`, `.z`, and `.w`.
  - Valid as rvalues (`v.x`, `p[i].y`) and lvalues (`v.x = expr;`, `p[i].z = expr;`).
  - Accessing invalid components (e.g., `v2.z` or `v.a`) is a compile-time error.
- Constructors:
  - `int2(x, y)` / `make_int2(x, y)`
  - `int4(x, y, z, w)` / `make_int4(x, y, z, w)`
  - `float2(x, y)` / `make_float2(x, y)`
  - `float4(x, y, z, w)` / `make_float4(x, y, z, w)`
  - Arguments accept matching element types with implicit promotion from `int` to `float` for float vectors. Argument count mismatch or incompatible types produce compile-time errors.

### Literals
- Integers: Decimal (`0`, `42`, `1000`) and hexadecimal (`0x10`, `0xFF`).
  - Range (same rule as Java): a decimal literal must be at most `2147483647`, except that
    `2147483648` is allowed as the operand of unary minus, so `-2147483648` is INT_MIN.
    A hex literal may be up to `0xFFFFFFFF` and is read as a 32-bit two's-complement bit
    pattern (`0xFFFFFFFF` is `-1`, `0x80000000` is INT_MIN). Larger literals are errors.
- Floats: Standard decimal and scientific notation (`3.14`, `1.0f`, `.5`, `1e-3`, `2.5e+2`).

### Built-in Identifiers
`kernc` provides complete 1D, 2D, and 3D execution geometry coordinates, accessible either with underscore suffixes (`_x`, `_y`, `_z`), dot notation (`.x`, `.y`, `.z`), or standard CUDA camelCase (`threadIdx.x`, etc.):
- Thread index in block:
  - `thread_idx_x`, `thread_idx.x`, `threadIdx.x` (or `thread_idx`) -> `%tid.x`
  - `thread_idx_y`, `thread_idx.y`, `threadIdx.y` -> `%tid.y`
  - `thread_idx_z`, `thread_idx.z`, `threadIdx.z` -> `%tid.z`
- Block / CTA index in grid:
  - `block_idx_x`, `block_idx.x`, `blockIdx.x` (or `block_idx`) -> `%ctaid.x`
  - `block_idx_y`, `block_idx.y`, `blockIdx.y` -> `%ctaid.y`
  - `block_idx_z`, `block_idx.z`, `blockIdx.z` -> `%ctaid.z`
- Block dimensions (threads per block):
  - `block_dim_x`, `block_dim.x`, `blockDim.x` (or `block_dim`) -> `%ntid.x`
  - `block_dim_y`, `block_dim.y`, `blockDim.y` -> `%ntid.y`
  - `block_dim_z`, `block_dim.z`, `blockDim.z` -> `%ntid.z`
- Grid dimensions (blocks per grid):
  - `grid_dim_x`, `grid_dim.x`, `gridDim.x` (or `grid_dim`) -> `%nctaid.x`
  - `grid_dim_y`, `grid_dim.y`, `gridDim.y` -> `%nctaid.y`
  - `grid_dim_z`, `grid_dim.z`, `gridDim.z` -> `%nctaid.z`

All built-in identifiers evaluate to 32-bit signed integer values (`int`). Their names are reserved keywords in `Sema` and cannot be shadowed by parameters or local variables.

### Operators
- Arithmetic: `+`, `-`, `*`, `/`, `%` (`%` is int-only, lowering to `rem.s32`).
- Bitwise: `&`, `|`, `^`, `<<`, `>>` on 32-bit `int` operands, lowering to `and.b32`, `or.b32`, `xor.b32`, `shl.b32`, `shr.s32`.
  Shifts follow PTX, not C: the count is read as unsigned and clamped to 32, so `x << 32`,
  `x << -1` and `x >> 40` are defined. A count of 32 or more gives 0 for `<<`, and all sign
  bits (0 or -1) for `>>`. Constant folding uses the same rule.
- Unary: `-` (arithmetic negation), `!` (logical NOT), `~` (bitwise NOT, `not.b32` on `int`).
- Comparison: `<`, `<=`, `>`, `>=`, `==`, `!=` on `int`/`float`; `==`, `!=` also on two `bool`s.
  Float comparisons follow C: all are false when an operand is NaN, except `!=`, which is true
  (lowered to the unordered `setp.neu.f32`).
- Logical: `&&`, `||` with short-circuit evaluation, both in conditions and as values.
- Precedence: Precedence climbing strictly follows ISO C:
  `||` < `&&` < `|` < `^` < `&` < `==`/`!=` < `<`/`<=`/`>`/`>=` < `<<`/`>>` < `+`/`-` < `*`/`/`/`%`
  Unary operators `!`, `-`, `~` have higher precedence than binary operators.
- Subscript / Indexing: `ptr[expr]` for pointer dereferencing and offset calculation.
- Explicit Cast: `cast<type>(expr)` between `int`, `float` and `bool`. `cast<bool>(x)` is
  `x != 0` (NaN is true); `cast<int>(b)` / `cast<float>(b)` give 1 or 0.

### Statements
- Variable Declarations: `int x = expr;`, `float y;`.
- Assignments: `x = expr;`, `ptr[idx] = expr;`.
- Conditional: `if (cond) { ... } else { ... }`.
- Loops: `while (cond) { ... }`, `for (init; cond; step) { ... }`. Loops with constant bounds and step ($0 < T \le 16$) are automatically fully unrolled into straight-line instructions, eliminating `bra`, `bra.cond`, and `setp` branch overhead. Downstream list scheduling pipelines the unrolled memory accesses.
- Return: `return;` -> `ret;`, which ends the calling thread (other threads continue).
  Kernels return void, so `return expr;` is an error.
- Shared memory: `shared float buf[256];` (or `__shared__`) declares a per-block array in the
  `.shared` state space. The element type is `int` or `float`, the size is an integer literal,
  and a kernel may declare at most 48 KB (49152 bytes) of shared memory in total (the sm_86
  static limit). Only elements can be read or written (`buf[i]`); the array itself is not a
  value. Arrays are emitted at module scope as `<kernel>$<name>` and addressed with 32-bit
  shared-window offsets (`mov.u32`, `mul.lo.s32`, `ld/st.shared`).
- Synchronization: `barrier();` -> `bar.sync 0;`. Every thread of the block must reach each
  barrier, so a barrier must not sit behind a condition that differs between threads (or
  after an early `return` taken by some of them). See `examples/reduce.kern`.

### Functions
- Kernels: `kernel name(params) { ... }` defines a top-level GPU entry point (`.visible .entry` in PTX). Kernels return void and cannot accept `bool` parameters.
- Device Functions: `func <type> name(params) { ... }` defines a device function (`.func` in PTX) that can be called from kernels and other device functions.
  - Return types: `void`, `int`, `float`, or `bool`. Device functions cannot return pointers.
  - Non-void functions must return a value on all control paths; reaching the end of the body without a return is a compile error. Void functions cannot return a value.
  - Parameters: `int`, `float`, `bool`, `int*`, `float*`. Unlike kernels, `bool` parameters are permitted (passed across the ABI as 32-bit values). Pointer parameters receive already-converted global addresses.
  - Shared memory: `shared` arrays cannot be declared in device functions (they are block-scope and only allowed in kernels).
  - Calling: Calls can appear as expressions or expression statements (for void or discarded returns). Forward calls and recursion / mutual recursion are supported; signatures are collected in an initial pass and prototypes are emitted ahead of definitions.

### Warp-Level Intrinsics
`kernc` provides warp-level communication and synchronization primitives allowing threads within a 32-lane warp to exchange data and vote without accessing shared or global memory. All intrinsics are available both with and without the `__` prefix:
- `__shfl_sync(mask, val, src_lane)` / `shfl_sync(...)`: broadcast from lane `src_lane` -> `shfl.sync.idx.b32 d, a, b, 31, member_mask;`
- `__shfl_down_sync(mask, val, delta)` / `shfl_down_sync(...)`: shift down by `delta` lanes -> `shfl.sync.down.b32 d, a, b, 31, member_mask;`
- `__shfl_up_sync(mask, val, delta)` / `shfl_up_sync(...)`: shift up by `delta` lanes -> `shfl.sync.up.b32 d, a, b, 0, member_mask;`
- `__shfl_xor_sync(mask, val, lane_mask)` / `shfl_xor_sync(...)`: butterfly exchange -> `shfl.sync.bfly.b32 d, a, b, 31, member_mask;`
- `__ballot_sync(mask, predicate)` / `ballot_sync(...)`: 32-bit warp ballot -> `vote.sync.ballot.b32 d, pred, member_mask;`
- `__activemask()` / `activemask()`: returns 32-bit active lane mask -> `activemask.b32 d;`

Type rules in `Sema`:
- `mask`: `int` (32-bit member mask).
- `val`: `int` or `float` (return type matches `val`).
- `delta` / `src_lane` / `lane_mask`: `int`.
- `predicate`: `bool` or `int` (evaluated/converted to `.pred`).
- `activemask`: 0 arguments, returns `int`.
- Pointer types and `bool` values as shuffle payloads are rejected with compile-time diagnostics.
- Intrinsic names are reserved identifiers and cannot be redefined as functions or shadowed by variables.

### Atomic Memory Operations
`kernc` provides hardware-accelerated atomic memory transactions operating on global memory pointers (`int*`, `float*`) and shared memory arrays (`shared int`, `shared float`). Atomics guarantee serialized, race-free read-modify-write execution across concurrent threads and thread blocks.

Supported operations (available in both `snake_case` and CUDA `camelCase`, with and without `__` prefix):
- `atomic_add(ptr, val)` / `__atomic_add`, `atomicAdd`, `__atomicAdd`: atomic addition (`atom.add.s32`, `atom.add.f32`).
- `atomic_sub(ptr, val)` / `__atomic_sub`, `atomicSub`, `__atomicSub`: atomic subtraction (`neg.s32` + `atom.add.s32`).
- `atomic_min(ptr, val)` / `__atomic_min`, `atomicMin`, `__atomicMin`: atomic signed minimum (`atom.min.s32`).
- `atomic_max(ptr, val)` / `__atomic_max`, `atomicMax`, `__atomicMax`: atomic signed maximum (`atom.max.s32`).
- `atomic_and(ptr, val)` / `__atomic_and`, `atomicAnd`, `__atomicAnd`: atomic bitwise AND (`atom.and.b32`).
- `atomic_or(ptr, val)` / `__atomic_or`, `atomicOr`, `__atomicOr`: atomic bitwise OR (`atom.or.b32`).
- `atomic_xor(ptr, val)` / `__atomic_xor`, `atomicXor`, `__atomicXor`: atomic bitwise XOR (`atom.xor.b32`).
- `atomic_exch(ptr, val)` / `__atomic_exch`, `atomicExch`, `__atomicExch`, `atomic_exchange`, `atomicExchange`: atomic swap / exchange (`atom.exch.b32`).
- `atomic_cas(ptr, cmp, val)` / `__atomic_cas`, `atomicCAS`, `__atomicCAS`, `atomicCas`: atomic compare-and-swap (`atom.cas.b32`).

Syntax forms:
- Direct address: `atomic_add(ptr, val)` or `atomic_cas(ptr, cmp, val)`.
- Explicit index offset: `atomic_add(ptr, index, val)` or `atomic_cas(ptr, index, cmp, val)`.
- Array subscript notation: `atomic_add(ptr[index], val)` or `atomic_cas(ptr[index], cmp, val)`.

Type rules in `Sema`:
- First operand (`ptr`) must be a pointer (`int*`, `float*`) or a shared array (`shared int`, `shared float`).
- `add` and `exch` accept both `int` and `float` memory targets (return type matches pointee type).
- `sub`, `min`, `max`, `and`, `or`, `xor`, and `cas` accept only `int` memory targets (float targets produce compile-time diagnostics).
- `index`, `cmp`, and bitwise `val` operands must be `int`.
- Returns the old (prior) value at the specified memory location; can be consumed as an expression or used as an expression statement.
- Intrinsic names are reserved identifiers and cannot be shadowed or redefined.

### Vector Memory Intrinsics
`kernc` provides explicit helper intrinsics for loading and storing vectorized data:
- `load_int2(ptr, offset)` -> `int2`
- `load_int4(ptr, offset)` -> `int4`
- `load_float2(ptr, offset)` -> `float2`
- `load_float4(ptr, offset)` -> `float4`
- `store_int2(ptr, offset, val)`
- `store_int4(ptr, offset, val)`
- `store_float2(ptr, offset, val)`
- `store_float4(ptr, offset, val)`

Semantics and Type Rules:
- `ptr`: accepts scalar pointers (`int*`, `float*`), vector pointers (`int2*`, `int4*`, `float2*`, `float4*`), and shared memory arrays (`shared int`, `shared float`, `shared int2`, etc.).
- `offset`: integer offset counted in whole vectors of the accessed width, like `reinterpret_cast<int4*>(p)[offset]` in CUDA: `load_int4(p, i)` reads `p[4i .. 4i+3]` and `load_int2(p, i)` reads `p[2i .. 2i+1]`, for scalar and vector buffers alike. Every access is therefore naturally aligned (16 bytes for v4, 8 for v2), provided the buffer itself is (`cuMemAlloc` returns 256-byte-aligned memory, and shared arrays are 16-byte aligned). An earlier version counted scalars, so `load_int4(p, 1)` faulted with `CUDA_ERROR_MISALIGNED_ADDRESS`.
- `val`: matching vector type (`int2`/`int4`/`float2`/`float4`).
- Intrinsic names are reserved identifiers and cannot be shadowed.


