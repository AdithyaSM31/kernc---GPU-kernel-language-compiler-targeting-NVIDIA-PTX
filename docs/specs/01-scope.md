# `kernc` Specification 01: Language Scope & Syntax

## Overview
`kernc` is a standalone C++17 compiler that lowers a specialized GPU kernel language (`.kern`) to NVIDIA PTX (`sm_86`, Ampere architecture). The generated PTX code is assembled via `ptxas` or loaded at runtime via the CUDA Driver API (`cuModuleLoadDataEx`).

## Language Features

### Data Types
- `int`: 32-bit signed integer (`.s32` in PTX).
- `float`: 32-bit IEEE-754 single-precision floating point (`.f32` in PTX).
- Pointers: `int*` and `float*` (valid as kernel and device function parameters).
- `bool`: `true` / `false`, held in `.pred` registers. Comparisons and `!`, `&&`, `||` produce
  `bool`. It does not convert implicitly to or from `int`/`float` (use `cast<>`), cannot be a
  kernel parameter (`.param` has no predicate type; pass an `int`), and has no pointer type.
  Conditions (`if`, `while`, `for`, `!`, `&&`, `||`) accept `bool` or `int` (non-zero is true).

### Literals
- Integers: Decimal (`0`, `42`, `1000`) and hexadecimal (`0x10`, `0xFF`).
  - Range (same rule as Java): a decimal literal must be at most `2147483647`, except that
    `2147483648` is allowed as the operand of unary minus, so `-2147483648` is INT_MIN.
    A hex literal may be up to `0xFFFFFFFF` and is read as a 32-bit two's-complement bit
    pattern (`0xFFFFFFFF` is `-1`, `0x80000000` is INT_MIN). Larger literals are errors.
- Floats: Standard decimal and scientific notation (`3.14`, `1.0f`, `.5`, `1e-3`, `2.5e+2`).

### Built-in Identifiers
- `thread_idx` -> `%tid.x` (Thread index in block, x-dimension).
- `block_idx`  -> `%ctaid.x` (Block/CTA index in grid, x-dimension).
- `block_dim`  -> `%ntid.x` (Block dimension, x-dimension).

### Operators
- Arithmetic: `+`, `-`, `*`, `/`, `%` (`%` is int-only, lowering to `rem.s32`).
- Unary: `-` (arithmetic negation), `!` (logical NOT).
- Comparison: `<`, `<=`, `>`, `>=`, `==`, `!=` on `int`/`float`; `==`, `!=` also on two `bool`s.
  Float comparisons follow C: all are false when an operand is NaN, except `!=`, which is true
  (lowered to the unordered `setp.neu.f32`).
- Logical: `&&`, `||` with short-circuit evaluation, both in conditions and as values.
- Subscript / Indexing: `ptr[expr]` for pointer dereferencing and offset calculation.
- Explicit Cast: `cast<type>(expr)` between `int`, `float` and `bool`. `cast<bool>(x)` is
  `x != 0` (NaN is true); `cast<int>(b)` / `cast<float>(b)` give 1 or 0.

### Statements
- Variable Declarations: `int x = expr;`, `float y;`.
- Assignments: `x = expr;`, `ptr[idx] = expr;`.
- Conditional: `if (cond) { ... } else { ... }`.
- Loops: `while (cond) { ... }`, `for (init; cond; step) { ... }`.
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
