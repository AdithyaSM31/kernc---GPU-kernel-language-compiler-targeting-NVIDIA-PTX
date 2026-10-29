# `kernc` Specification 02: Pipeline Architecture & Front-end

## Front-end Pipeline

The compilation pipeline is organized into distinct phases with strict error boundaries:

```
[ .kern Source ]
       │
       ▼
  [ Lexer ] ──> Token Stream (string_view, zero-copy, source locations)
       │
       ▼
 [ Parser ] ──> Abstract Syntax Tree (Precedence climbing, unique_ptr ownership)
       │
       ▼
  [ Sema ]  ──> Type checked & Annotated AST (Scope stack, line:col diagnostics)
       │
       ▼
 [ IRGen ]  ──> 3-Address CFG IR (Branch threading, wide indexing)
       │
       ▼
 [ Passes ] ──> Optimized IR (Constant folding, dead code elimination)
       │
       ▼
[ PTXEmitter ] ──> PTX Assembly (sm_86, IEEE 754 hex floats, register counts)
```

## Front-end Components

### 1. Lexer (`include/kernc/Lexer.h`, `src/Lexer.cpp`)
- Tracks 1-based `line` and `column` numbers.
- Zero-copy token representation via `std::string_view` referencing the source buffer.
- Strips single-line (`//`) and multi-line (`/* */`) comments.
- Accurately tokenizes integer (decimal/hex) and floating-point literals with exponent and suffix handling.

### 2. Parser (`include/kernc/Parser.h`, `src/Parser.cpp`)
- Recursive-descent parser with precedence climbing for binary expressions.
- Top-level constructs: `kernel` declarations and `func` device functions.
- Function calls `f(args...)` parsed as primary expressions or expression statements.
- Strict left-associativity enforcement for arithmetic, relational, and logical operators.
- Strict single-ownership AST model using `std::unique_ptr<ASTNode>`.

### 3. Semantic Analysis (`include/kernc/Sema.h`, `src/Sema.cpp`)
- Two-pass analysis:
  - Pass 1 collects all function signatures (name, return type, parameter types) to support forward references and mutual recursion without header files. Reports redefinitions.
  - Pass 2 type-checks bodies, statements, and expressions.
- Lexical scoping using nested symbol tables (`std::vector<std::unordered_map<std::string, Type>>`).
- Diagnostic verification for:
  - Redefinitions of variables and functions.
  - Undeclared identifier references and undeclared function calls.
  - Calling kernels from device code (kernels are host-launched only).
  - Function argument count and argument type checking (implicit `int -> float` promotion allowed; type checks all arguments even when arity is mismatched).
  - Definite-assignment dataflow analysis: warns on variable reads that may be uninitialized across branches and loops (`DiagnosticLevel::Warning`).
  - Barrier-divergence analysis: warns on `barrier()` inside divergent control flow (`thread_idx`-dependent branches) or following a divergent early return.
  - Prohibiting void function calls as values.
  - Return validation: type matching (`int -> float` allowed), void functions returning no value, non-void functions returning a value.
  - Conservative all-paths-return analysis diagnosing missing returns in non-void functions.
  - Device function restrictions: cannot return pointers, cannot declare shared memory arrays.
  - Pointer assignment constraints.
  - Type checking and implicit `int -> float` promotion rules.
  - Subscript legality (base must be a pointer or shared array, index must be an integer).
  - Builtin reassignment rejection.
