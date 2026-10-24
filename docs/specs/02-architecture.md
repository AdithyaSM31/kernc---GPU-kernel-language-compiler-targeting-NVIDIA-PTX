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
- Strict left-associativity enforcement for arithmetic, relational, and logical operators.
- Strict single-ownership AST model using `std::unique_ptr<ASTNode>`.

### 3. Semantic Analysis (`include/kernc/Sema.h`, `src/Sema.cpp`)
- Lexical scoping using nested symbol tables (`std::vector<std::unordered_map<std::string, Type>>`).
- Diagnostic verification for:
  - Redefinitions in current scope.
  - Undeclared identifier references.
  - Pointer assignment constraints.
  - Type checking and implicit `int -> float` promotion rules.
  - Subscript legality (base must be a pointer, index must be an integer).
  - Builtin reassignment rejection.
