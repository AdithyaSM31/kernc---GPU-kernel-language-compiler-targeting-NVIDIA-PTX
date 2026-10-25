#include "kernc/Sema.h"

namespace kernc {

Sema::Sema(DiagnosticsEngine& diag) : diag_(diag) {}

bool Sema::checkProgram(Program& program) {
    bool ok = true;
    for (auto& kernel : program.kernels) {
        if (!checkKernel(*kernel)) {
            ok = false;
        }
    }
    return ok && !diag_.hasErrors();
}

bool Sema::checkKernel(KernelDecl& kernel) {
    currentKernel_ = kernel.name;
    symTable_.enterScope();

    for (const auto& param : kernel.params) {
        if (param.name == "thread_idx" || param.name == "block_idx" || param.name == "block_dim") {
            diag_.error(param.loc, "kernel parameter cannot have reserved builtin name '" + param.name + "'");
            continue;
        }
        if (param.type.isBool()) {
            // bool lives in .pred registers, and the .param space has no predicate type.
            diag_.error(param.loc, "kernel parameter '" + param.name +
                        "' cannot have type 'bool' (pass an int instead)");
        }
        if (!symTable_.declare(param.name, param.type)) {
            diag_.error(param.loc, "duplicate kernel parameter name '" + param.name + "'");
        }
    }

    for (auto& stmt : kernel.body) {
        checkStmt(*stmt);
    }

    symTable_.exitScope();
    return !diag_.hasErrors();
}

void Sema::checkStmt(Stmt& stmt) {
    if (auto* varDecl = dynamic_cast<VarDeclStmt*>(&stmt)) {
        checkVarDecl(*varDecl);
    } else if (auto* assign = dynamic_cast<AssignStmt*>(&stmt)) {
        checkAssign(*assign);
    } else if (auto* ifStmt = dynamic_cast<IfStmt*>(&stmt)) {
        checkIf(*ifStmt);
    } else if (auto* whileStmt = dynamic_cast<WhileStmt*>(&stmt)) {
        checkWhile(*whileStmt);
    } else if (auto* forStmt = dynamic_cast<ForStmt*>(&stmt)) {
        checkFor(*forStmt);
    } else if (auto* block = dynamic_cast<BlockStmt*>(&stmt)) {
        checkBlock(*block);
    } else if (auto* exprStmt = dynamic_cast<ExprStmt*>(&stmt)) {
        checkExpr(*exprStmt->expr);
    } else if (dynamic_cast<BarrierStmt*>(&stmt)) {
        // barrier() takes no args, returns void
    } else if (auto* ret = dynamic_cast<ReturnStmt*>(&stmt)) {
        if (ret->value) {
            checkExpr(*ret->value);
            diag_.error(ret->value->range.begin,
                        "kernel '" + currentKernel_ + "' cannot return a value (kernels return void)");
        }
    }
}

void Sema::checkVarDecl(VarDeclStmt& stmt) {
    if (stmt.name == "thread_idx" || stmt.name == "block_idx" || stmt.name == "block_dim") {
        diag_.error(stmt.range.begin, "cannot declare variable with reserved builtin name '" + stmt.name + "'");
        return;
    }
    if (stmt.type.isPtr()) {
        diag_.error(stmt.range.begin, "pointer variable declarations are not permitted (pointers only allowed as kernel parameters)");
    }
    if (!symTable_.declare(stmt.name, stmt.type)) {
        diag_.error(stmt.range.begin, "redefinition of variable '" + stmt.name + "'");
        return;
    }

    if (stmt.init) {
        Type initType = checkExpr(*stmt.init);
        if (initType.isVoid()) return;

        if (stmt.type == initType) {
            // Perfect match
        } else if (stmt.type.isFloat() && initType.isInt()) {
            // Implicit int -> float promotion allowed
        } else {
            diag_.error(stmt.init->range.begin,
                "type mismatch in variable initialization: cannot initialize '" +
                stmt.type.toString() + "' with '" + initType.toString() + "'");
        }
    }
}

void Sema::checkAssign(AssignStmt& stmt) {
    Type targetType = checkExpr(*stmt.target);
    Type valueType = checkExpr(*stmt.value);

    if (targetType.isVoid() || valueType.isVoid()) return;

    // Check lvalue requirement
    if (auto* builtin = dynamic_cast<BuiltinExpr*>(stmt.target.get())) {
        diag_.error(stmt.target->range.begin, "cannot assign to built-in '" + builtin->name() + "'");
        return;
    }
    if (auto* ident = dynamic_cast<IdentExpr*>(stmt.target.get())) {
        if (targetType.isPtr()) {
            diag_.error(stmt.target->range.begin, "cannot assign to pointer variable '" + ident->name + "'");
            return;
        }
    } else if (dynamic_cast<IndexExpr*>(stmt.target.get())) {
        // Valid lvalue: ptr[i]
    } else {
        diag_.error(stmt.target->range.begin, "lvalue required as left operand of assignment");
        return;
    }

    if (targetType == valueType) {
        // Match
    } else if (targetType.isFloat() && valueType.isInt()) {
        // Implicit int -> float promotion
    } else {
        diag_.error(stmt.range.begin,
            "type mismatch in assignment: cannot assign '" +
            valueType.toString() + "' to '" + targetType.toString() + "'");
    }
}

void Sema::checkIf(IfStmt& stmt) {
    Type condType = checkExpr(*stmt.cond);
    if (!isConditionType(condType)) {
        diag_.error(stmt.cond->range.begin, "condition must be a boolean or integer expression (got '" + condType.toString() + "')");
    }
    checkStmt(*stmt.thenBranch);
    if (stmt.elseBranch) {
        checkStmt(*stmt.elseBranch);
    }
}

void Sema::checkWhile(WhileStmt& stmt) {
    Type condType = checkExpr(*stmt.cond);
    if (!isConditionType(condType)) {
        diag_.error(stmt.cond->range.begin, "while condition must be a boolean or integer expression (got '" + condType.toString() + "')");
    }
    checkStmt(*stmt.body);
}

void Sema::checkFor(ForStmt& stmt) {
    symTable_.enterScope();
    if (stmt.init) {
        checkStmt(*stmt.init);
    }
    if (stmt.cond) {
        Type condType = checkExpr(*stmt.cond);
        if (!isConditionType(condType)) {
            diag_.error(stmt.cond->range.begin, "for loop condition must be a boolean or integer expression (got '" + condType.toString() + "')");
        }
    }
    if (stmt.step) {
        checkStmt(*stmt.step);
    }
    checkStmt(*stmt.body);
    symTable_.exitScope();
}

void Sema::checkBlock(BlockStmt& stmt) {
    symTable_.enterScope();
    for (auto& s : stmt.stmts) {
        checkStmt(*s);
    }
    symTable_.exitScope();
}

Type Sema::checkExpr(Expr& expr) {
    if (auto* lit = dynamic_cast<IntLiteralExpr*>(&expr)) {
        return checkIntLiteral(*lit, /*negated=*/false);
    }
    if (dynamic_cast<FloatLiteralExpr*>(&expr)) {
        expr.resolvedType = Type::makeFloat();
        return expr.resolvedType;
    }
    if (dynamic_cast<BoolLiteralExpr*>(&expr)) {
        expr.resolvedType = Type::makeBool();
        return expr.resolvedType;
    }
    if (dynamic_cast<BuiltinExpr*>(&expr)) {
        expr.resolvedType = Type::makeInt();
        return expr.resolvedType;
    }
    if (auto* ident = dynamic_cast<IdentExpr*>(&expr)) {
        return checkIdentExpr(*ident);
    }
    if (auto* bin = dynamic_cast<BinaryExpr*>(&expr)) {
        return checkBinaryExpr(*bin);
    }
    if (auto* unary = dynamic_cast<UnaryExpr*>(&expr)) {
        return checkUnaryExpr(*unary);
    }
    if (auto* idx = dynamic_cast<IndexExpr*>(&expr)) {
        return checkIndexExpr(*idx);
    }
    if (auto* cast = dynamic_cast<CastExpr*>(&expr)) {
        return checkCastExpr(*cast);
    }
    return Type::makeVoid();
}

Type Sema::checkIdentExpr(IdentExpr& expr) {
    const Type* t = symTable_.lookup(expr.name);
    if (!t) {
        diag_.error(expr.range.begin, "undeclared variable '" + expr.name + "'");
        expr.resolvedType = Type::makeVoid();
        return expr.resolvedType;
    }
    expr.resolvedType = *t;
    return expr.resolvedType;
}

// Range rule for 32-bit int literals (same as Java, JLS 3.10.1):
//  - decimal: at most 2147483647, or 2147483648 when it is the operand of unary minus;
//  - hex: at most 0xFFFFFFFF, read as a two's-complement bit pattern (0xFFFFFFFF == -1).
// The literal still has type 'int' when out of range, so no follow-on errors are reported.
Type Sema::checkIntLiteral(IntLiteralExpr& expr, bool negated) {
    bool isHex = expr.spelling.size() > 1 && (expr.spelling[1] == 'x' || expr.spelling[1] == 'X');
    int64_t max = isHex ? 0xFFFFFFFFLL : (negated ? 2147483648LL : 2147483647LL);
    if (expr.value > max) {
        diag_.error(expr.range.begin, "integer literal '" + expr.spelling +
                    "' is out of range for type 'int'");
    }
    expr.resolvedType = Type::makeInt();
    return expr.resolvedType;
}

Type Sema::checkBinaryExpr(BinaryExpr& expr) {
    Type lhsType = checkExpr(*expr.lhs);
    Type rhsType = checkExpr(*expr.rhs);

    if (lhsType.isVoid() || rhsType.isVoid()) {
        expr.resolvedType = Type::makeVoid();
        return expr.resolvedType;
    }

    switch (expr.op) {
        case BinaryOp::Add:
        case BinaryOp::Sub:
        case BinaryOp::Mul:
        case BinaryOp::Div: {
            if (lhsType.isInt() && rhsType.isInt()) {
                expr.resolvedType = Type::makeInt();
            } else if (lhsType.isFloat() && rhsType.isFloat()) {
                expr.resolvedType = Type::makeFloat();
            } else if ((lhsType.isInt() && rhsType.isFloat()) || (lhsType.isFloat() && rhsType.isInt())) {
                // Mixed promotion
                expr.resolvedType = Type::makeFloat();
            } else {
                diag_.error(expr.range.begin, "invalid operands to binary operator '" +
                    std::string(binaryOpToString(expr.op)) + "' ('" +
                    lhsType.toString() + "' and '" + rhsType.toString() + "')");
                expr.resolvedType = Type::makeVoid();
            }
            return expr.resolvedType;
        }

        case BinaryOp::Mod: {
            if (lhsType.isInt() && rhsType.isInt()) {
                expr.resolvedType = Type::makeInt();
            } else {
                diag_.error(expr.range.begin, "operator '%' requires integer operands (got '" +
                    lhsType.toString() + "' and '" + rhsType.toString() + "')");
                expr.resolvedType = Type::makeVoid();
            }
            return expr.resolvedType;
        }

        case BinaryOp::Lt:
        case BinaryOp::Gt:
        case BinaryOp::Le:
        case BinaryOp::Ge:
        case BinaryOp::Eq:
        case BinaryOp::Ne: {
            bool numeric = (lhsType.isInt() || lhsType.isFloat()) && (rhsType.isInt() || rhsType.isFloat());
            // bools are equality-comparable with each other only; they have no ordering.
            bool boolEquality = lhsType.isBool() && rhsType.isBool() &&
                                (expr.op == BinaryOp::Eq || expr.op == BinaryOp::Ne);
            if (numeric || boolEquality) {
                expr.resolvedType = Type::makeBool();
            } else {
                diag_.error(expr.range.begin, "invalid operands to comparison operator '" +
                    std::string(binaryOpToString(expr.op)) + "' ('" +
                    lhsType.toString() + "' and '" + rhsType.toString() + "')");
                expr.resolvedType = Type::makeVoid();
            }
            return expr.resolvedType;
        }

        case BinaryOp::And:
        case BinaryOp::Or: {
            if (isConditionType(lhsType) && isConditionType(rhsType)) {
                expr.resolvedType = Type::makeBool();
            } else {
                diag_.error(expr.range.begin, "logical operator requires boolean or integer operands");
                expr.resolvedType = Type::makeVoid();
            }
            return expr.resolvedType;
        }
    }
    return Type::makeVoid();
}

Type Sema::checkUnaryExpr(UnaryExpr& expr) {
    // "-2147483648" lexes as minus applied to 2147483648, so the literal's range depends on
    // whether it is negated.
    auto* lit = dynamic_cast<IntLiteralExpr*>(expr.operand.get());
    Type operandType = (expr.op == UnaryOp::Neg && lit) ? checkIntLiteral(*lit, /*negated=*/true)
                                                        : checkExpr(*expr.operand);
    if (operandType.isVoid()) {
        expr.resolvedType = Type::makeVoid();
        return expr.resolvedType;
    }

    if (expr.op == UnaryOp::Neg) {
        if (operandType.isInt() || operandType.isFloat()) {
            expr.resolvedType = operandType;
        } else {
            diag_.error(expr.range.begin, "unary '-' requires integer or float operand");
            expr.resolvedType = Type::makeVoid();
        }
    } else if (expr.op == UnaryOp::Not) {
        if (isConditionType(operandType)) {
            expr.resolvedType = Type::makeBool();
        } else {
            diag_.error(expr.range.begin, "unary '!' requires boolean or integer operand");
            expr.resolvedType = Type::makeVoid();
        }
    }
    return expr.resolvedType;
}

Type Sema::checkIndexExpr(IndexExpr& expr) {
    Type baseType = checkExpr(*expr.base);
    Type indexType = checkExpr(*expr.index);

    if (!baseType.isPtr()) {
        diag_.error(expr.base->range.begin, "subscripted value is not a pointer (type is '" + baseType.toString() + "')");
        expr.resolvedType = Type::makeVoid();
        return expr.resolvedType;
    }

    if (!indexType.isInt()) {
        diag_.error(expr.index->range.begin, "array subscript is not an integer (type is '" + indexType.toString() + "')");
        expr.resolvedType = Type::makeVoid();
        return expr.resolvedType;
    }

    if (baseType.pointeeKind == Type::Int) {
        expr.resolvedType = Type::makeInt();
    } else if (baseType.pointeeKind == Type::Float) {
        expr.resolvedType = Type::makeFloat();
    } else {
        expr.resolvedType = Type::makeVoid();
    }
    return expr.resolvedType;
}

Type Sema::checkCastExpr(CastExpr& expr) {
    Type operandType = checkExpr(*expr.operand);
    if (operandType.isVoid()) {
        expr.resolvedType = Type::makeVoid();
        return expr.resolvedType;
    }

    // Explicit casts convert freely among the scalar types: int, float and bool.
    auto isScalar = [](const Type& t) { return t.isInt() || t.isFloat() || t.isBool(); };
    if (isScalar(operandType) && isScalar(expr.targetType)) {
        expr.resolvedType = expr.targetType;
    } else {
        diag_.error(expr.range.begin, "invalid cast from '" + operandType.toString() + "' to '" + expr.targetType.toString() + "'");
        expr.resolvedType = Type::makeVoid();
    }
    return expr.resolvedType;
}

} // namespace kernc
