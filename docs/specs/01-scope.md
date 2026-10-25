# `kernc` Specification 01: Language Scope & Syntax

## Overview
`kernc` is a standalone C++17 compiler that lowers a specialized GPU kernel language (`.kern`) to NVIDIA PTX (`sm_86`, Ampere architecture). The generated PTX code is assembled via `ptxas` or loaded at runtime via the CUDA Driver API (`cuModuleLoadDataEx`).

## Language Features

### Data Types
- `int`: 32-bit signed integer (`.s32` in PTX).
- `float`: 32-bit IEEE-754 single-precision floating point (`.f32` in PTX).
- Pointers: `int*` and `float*` (valid exclusively as kernel parameters).
- Predicates: Implicit comparison values (`.pred` in PTX), used directly in branch conditions.

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
- Comparison: `<`, `<=`, `>`, `>=`, `==`, `!=`.
- Logical: `&&`, `||` with true short-circuit evaluation via branch threading.
- Subscript / Indexing: `ptr[expr]` for pointer dereferencing and offset calculation.
- Explicit Cast: `cast<type>(expr)` supporting `cast<int>` and `cast<float>`.

### Statements
- Variable Declarations: `int x = expr;`, `float y;`.
- Assignments: `x = expr;`, `ptr[idx] = expr;`.
- Conditional: `if (cond) { ... } else { ... }`.
- Loops: `while (cond) { ... }`, `for (init; cond; step) { ... }`.
- Synchronization: `barrier();` -> `bar.sync 0;`.
