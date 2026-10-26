#include "kernc/Parser.h"
#include <iostream>

namespace kernc {

Parser::Parser(Lexer& lexer, DiagnosticsEngine& diag)
    : lexer_(lexer), diag_(diag) {
    current_ = lexer_.next();
}

Token Parser::advance() {
    Token prev = current_;
    current_ = lexer_.next();
    return prev;
}

bool Parser::match(TokenKind kind) {
    if (check(kind)) {
        advance();
        return true;
    }
    return false;
}

bool Parser::check(TokenKind kind) const {
    return current_.kind == kind;
}

Token Parser::consume(TokenKind kind, const std::string& errorMsg) {
    if (check(kind)) {
        return advance();
    }
    diag_.error(current_.loc, errorMsg + " (got '" + std::string(current_.text) + "')");
    hasError_ = true;
    return current_;
}

Type Parser::parseType() {
    if (match(TokenKind::KW_void)) {
        return Type::makeVoid();
    }
    if (match(TokenKind::KW_int)) {
        if (match(TokenKind::Star)) {
            return Type::makeIntPtr();
        }
        return Type::makeInt();
    }
    if (match(TokenKind::KW_float)) {
        if (match(TokenKind::Star)) {
            return Type::makeFloatPtr();
        }
        return Type::makeFloat();
    }
    if (match(TokenKind::KW_bool)) {
        // bool lives in .pred registers, which have no memory representation to point at.
        if (check(TokenKind::Star)) {
            diag_.error(current_.loc, "pointers to 'bool' are not supported");
            hasError_ = true;
            advance();
        }
        return Type::makeBool();
    }
    diag_.error(current_.loc, "expected type ('int', 'float' or 'bool')");
    hasError_ = true;
    return Type::makeVoid();
}

Param Parser::parseParam() {
    SourceLocation loc = current_.loc;
    Type type = parseType();
    Token nameTok = consume(TokenKind::Identifier, "expected parameter name");
    return Param{type, std::string(nameTok.text), loc};
}

std::unique_ptr<FunctionDecl> Parser::parseKernel() {
    SourceLocation startLoc = current_.loc;
    consume(TokenKind::KW_kernel, "expected 'kernel'");
    Token nameTok = consume(TokenKind::Identifier, "expected kernel name");
    consume(TokenKind::LParen, "expected '(' after kernel name");

    std::vector<Param> params;
    if (!check(TokenKind::RParen)) {
        do {
            params.push_back(parseParam());
        } while (match(TokenKind::Comma));
    }
    consume(TokenKind::RParen, "expected ')' after parameter list");

    consume(TokenKind::LBrace, "expected '{' before kernel body");
    std::vector<std::unique_ptr<Stmt>> body;
    while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
        auto stmt = parseStmt();
        if (stmt) {
            body.push_back(std::move(stmt));
        } else if (hasError_) {
            // Error occurred, stop
            return nullptr;
        }
    }
    SourceLocation endLoc = current_.loc;
    consume(TokenKind::RBrace, "expected '}' after kernel body");

    auto decl = std::make_unique<FunctionDecl>();
    decl->name = std::string(nameTok.text);
    decl->params = std::move(params);
    decl->body = std::move(body);
    decl->range = {startLoc, endLoc};
    return decl;
}

std::unique_ptr<FunctionDecl> Parser::parseFunction() {
    SourceLocation startLoc = current_.loc;
    consume(TokenKind::KW_func, "expected 'func'");
    Type returnType = parseType();
    Token nameTok = consume(TokenKind::Identifier, "expected function name");
    consume(TokenKind::LParen, "expected '(' after function name");

    std::vector<Param> params;
    if (!check(TokenKind::RParen)) {
        do {
            params.push_back(parseParam());
        } while (match(TokenKind::Comma));
    }
    consume(TokenKind::RParen, "expected ')' after parameter list");

    consume(TokenKind::LBrace, "expected '{' before function body");
    std::vector<std::unique_ptr<Stmt>> body;
    while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
        auto stmt = parseStmt();
        if (stmt) {
            body.push_back(std::move(stmt));
        } else if (hasError_) {
            return nullptr;
        }
    }
    SourceLocation endLoc = current_.loc;
    consume(TokenKind::RBrace, "expected '}' after function body");

    auto decl = std::make_unique<FunctionDecl>();
    decl->isKernel = false;
    decl->returnType = returnType;
    decl->name = std::string(nameTok.text);
    decl->params = std::move(params);
    decl->body = std::move(body);
    decl->range = {startLoc, endLoc};
    return decl;
}

std::unique_ptr<Program> Parser::parseProgram() {
    auto prog = std::make_unique<Program>();
    SourceLocation startLoc = current_.loc;
    while (!check(TokenKind::Eof)) {
        std::unique_ptr<FunctionDecl> fn = nullptr;
        if (check(TokenKind::KW_func)) {
            fn = parseFunction();
        } else if (check(TokenKind::KW_kernel)) {
            fn = parseKernel();
        } else {
            consume(TokenKind::KW_kernel, "expected 'kernel' or 'func'");
            return nullptr;
        }
        if (!fn) {
            return nullptr;
        }
        prog->functions.push_back(std::move(fn));
    }
    prog->range = {startLoc, current_.loc};
    return prog;
}

std::unique_ptr<Stmt> Parser::parseStmt() {
    if (check(TokenKind::KW_int) || check(TokenKind::KW_float) || check(TokenKind::KW_bool)) {
        return parseVarDecl();
    }
    if (check(TokenKind::KW_if)) {
        return parseIfStmt();
    }
    if (check(TokenKind::KW_while)) {
        return parseWhileStmt();
    }
    if (check(TokenKind::KW_for)) {
        return parseForStmt();
    }
    if (check(TokenKind::KW_return)) {
        return parseReturnStmt();
    }
    if (check(TokenKind::KW_shared)) {
        return parseSharedDecl();
    }
    if (check(TokenKind::LBrace)) {
        return parseBlock();
    }
    if (match(TokenKind::Semicolon)) {
        // Empty statement
        return std::make_unique<BlockStmt>();
    }
    return parseExprOrAssignStmt();
}

std::unique_ptr<Stmt> Parser::parseVarDecl() {
    SourceLocation startLoc = current_.loc;
    Type type = parseType();
    Token nameTok = consume(TokenKind::Identifier, "expected variable name");
    std::unique_ptr<Expr> init = nullptr;
    if (match(TokenKind::Eq)) {
        init = parseExpr();
    }
    consume(TokenKind::Semicolon, "expected ';' after variable declaration");
    auto stmt = std::make_unique<VarDeclStmt>(type, std::string(nameTok.text), std::move(init));
    stmt->range = {startLoc, current_.loc};
    return stmt;
}

std::unique_ptr<BlockStmt> Parser::parseBlock() {
    SourceLocation startLoc = current_.loc;
    consume(TokenKind::LBrace, "expected '{'");
    std::vector<std::unique_ptr<Stmt>> stmts;
    while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
        auto stmt = parseStmt();
        if (stmt) {
            stmts.push_back(std::move(stmt));
        } else if (hasError_) {
            return nullptr;
        }
    }
    consume(TokenKind::RBrace, "expected '}'");
    auto block = std::make_unique<BlockStmt>(std::move(stmts));
    block->range = {startLoc, current_.loc};
    return block;
}

std::unique_ptr<Stmt> Parser::parseIfStmt() {
    SourceLocation startLoc = current_.loc;
    consume(TokenKind::KW_if, "expected 'if'");
    consume(TokenKind::LParen, "expected '(' after 'if'");
    auto cond = parseExpr();
    consume(TokenKind::RParen, "expected ')' after if condition");
    auto thenBranch = parseStmt();
    std::unique_ptr<Stmt> elseBranch = nullptr;
    if (match(TokenKind::KW_else)) {
        elseBranch = parseStmt();
    }
    auto stmt = std::make_unique<IfStmt>(std::move(cond), std::move(thenBranch), std::move(elseBranch));
    stmt->range = {startLoc, current_.loc};
    return stmt;
}

std::unique_ptr<Stmt> Parser::parseWhileStmt() {
    SourceLocation startLoc = current_.loc;
    consume(TokenKind::KW_while, "expected 'while'");
    consume(TokenKind::LParen, "expected '(' after 'while'");
    auto cond = parseExpr();
    consume(TokenKind::RParen, "expected ')' after while condition");
    auto body = parseStmt();
    auto stmt = std::make_unique<WhileStmt>(std::move(cond), std::move(body));
    stmt->range = {startLoc, current_.loc};
    return stmt;
}

// shared <type> <name> [ <integer literal> ] ;
std::unique_ptr<Stmt> Parser::parseSharedDecl() {
    SourceLocation startLoc = current_.loc;
    consume(TokenKind::KW_shared, "expected 'shared'");
    Type elemType = parseType(); // Sema restricts it to int/float
    Token nameTok = consume(TokenKind::Identifier, "expected shared array name");
    consume(TokenKind::LBracket, "expected '[' after shared array name");
    int64_t size = 0;
    if (check(TokenKind::IntLiteral)) {
        size = advance().intVal;
    } else {
        // Static shared memory is laid out by ptxas, so its size must be known at compile time.
        diag_.error(current_.loc, "expected integer literal as shared array size");
        hasError_ = true;
    }
    consume(TokenKind::RBracket, "expected ']' after shared array size");
    consume(TokenKind::Semicolon, "expected ';' after shared array declaration");
    auto stmt = std::make_unique<SharedDeclStmt>(elemType, std::string(nameTok.text), size);
    stmt->range = {startLoc, current_.loc};
    return stmt;
}

std::unique_ptr<Stmt> Parser::parseReturnStmt() {
    SourceLocation startLoc = current_.loc;
    consume(TokenKind::KW_return, "expected 'return'");
    std::unique_ptr<Expr> value = nullptr;
    if (!check(TokenKind::Semicolon)) {
        value = parseExpr();
        if (!value) return nullptr;
    }
    consume(TokenKind::Semicolon, "expected ';' after 'return'");
    auto stmt = std::make_unique<ReturnStmt>(std::move(value));
    stmt->range = {startLoc, current_.loc};
    return stmt;
}

std::unique_ptr<Stmt> Parser::parseForStmt() {
    SourceLocation startLoc = current_.loc;
    consume(TokenKind::KW_for, "expected 'for'");
    consume(TokenKind::LParen, "expected '(' after 'for'");

    // Init
    std::unique_ptr<Stmt> init = nullptr;
    if (check(TokenKind::KW_int) || check(TokenKind::KW_float) || check(TokenKind::KW_bool)) {
        init = parseVarDecl();
    } else if (match(TokenKind::Semicolon)) {
        init = nullptr;
    } else {
        init = parseExprOrAssignStmt();
    }

    // Cond
    std::unique_ptr<Expr> cond = nullptr;
    if (!check(TokenKind::Semicolon)) {
        cond = parseExpr();
    }
    consume(TokenKind::Semicolon, "expected ';' after for condition");

    // Step
    std::unique_ptr<Stmt> step = nullptr;
    if (!check(TokenKind::RParen)) {
        // Step statement without semicolon
        SourceLocation stepLoc = current_.loc;
        auto expr = parseExpr();
        if (match(TokenKind::Eq)) {
            // Assignment: target = value
            auto val = parseExpr();
            step = std::make_unique<AssignStmt>(std::move(expr), std::move(val));
        } else {
            step = std::make_unique<ExprStmt>(std::move(expr));
        }
        step->range = {stepLoc, current_.loc};
    }
    consume(TokenKind::RParen, "expected ')' after for clauses");

    auto body = parseStmt();
    auto stmt = std::make_unique<ForStmt>(std::move(init), std::move(cond), std::move(step), std::move(body));
    stmt->range = {startLoc, current_.loc};
    return stmt;
}

std::unique_ptr<Stmt> Parser::parseExprOrAssignStmt() {
    SourceLocation startLoc = current_.loc;

    if (match(TokenKind::KW_barrier)) {
        consume(TokenKind::LParen, "expected '(' after 'barrier'");
        consume(TokenKind::RParen, "expected ')' after '(' in barrier call");
        consume(TokenKind::Semicolon, "expected ';' after barrier()");
        auto stmt = std::make_unique<BarrierStmt>();
        stmt->range = {startLoc, current_.loc};
        return stmt;
    }

    // Parse primary or indexed target
    auto expr = parseExpr();
    if (!expr) return nullptr;

    if (match(TokenKind::Eq)) {
        // Target is expr
        auto value = parseExpr();
        consume(TokenKind::Semicolon, "expected ';' after assignment");
        auto stmt = std::make_unique<AssignStmt>(std::move(expr), std::move(value));
        stmt->range = {startLoc, current_.loc};
        return stmt;
    }

    consume(TokenKind::Semicolon, "expected ';' after expression");
    auto stmt = std::make_unique<ExprStmt>(std::move(expr));
    stmt->range = {startLoc, current_.loc};
    return stmt;
}

int Parser::getBinaryPrecedence(TokenKind kind) const {
    switch (kind) {
        case TokenKind::PipePipe:       return 1;
        case TokenKind::AmpAmp:         return 2;
        case TokenKind::Pipe:           return 3;
        case TokenKind::Caret:          return 4;
        case TokenKind::Amp:            return 5;
        case TokenKind::EqEq:
        case TokenKind::BangEq:         return 6;
        case TokenKind::Less:
        case TokenKind::LessEq:
        case TokenKind::Greater:
        case TokenKind::GreaterEq:      return 7;
        case TokenKind::LessLess:
        case TokenKind::GreaterGreater: return 8;
        case TokenKind::Plus:
        case TokenKind::Minus:          return 9;
        case TokenKind::Star:
        case TokenKind::Slash:
        case TokenKind::Percent:        return 10;
        default: return 0;
    }
}

BinaryOp Parser::tokenToBinaryOp(TokenKind kind) const {
    switch (kind) {
        case TokenKind::PipePipe:       return BinaryOp::Or;
        case TokenKind::AmpAmp:         return BinaryOp::And;
        case TokenKind::Pipe:           return BinaryOp::BitOr;
        case TokenKind::Caret:          return BinaryOp::BitXor;
        case TokenKind::Amp:            return BinaryOp::BitAnd;
        case TokenKind::EqEq:           return BinaryOp::Eq;
        case TokenKind::BangEq:         return BinaryOp::Ne;
        case TokenKind::Less:           return BinaryOp::Lt;
        case TokenKind::LessEq:         return BinaryOp::Le;
        case TokenKind::Greater:        return BinaryOp::Gt;
        case TokenKind::GreaterEq:      return BinaryOp::Ge;
        case TokenKind::LessLess:       return BinaryOp::Shl;
        case TokenKind::GreaterGreater: return BinaryOp::Shr;
        case TokenKind::Plus:           return BinaryOp::Add;
        case TokenKind::Minus:          return BinaryOp::Sub;
        case TokenKind::Star:           return BinaryOp::Mul;
        case TokenKind::Slash:          return BinaryOp::Div;
        case TokenKind::Percent:        return BinaryOp::Mod;
        default: return BinaryOp::Add;
    }
}

std::unique_ptr<Expr> Parser::parseExpr(int minPrecedence) {
    auto lhs = parseUnary();
    if (!lhs) return nullptr;

    while (true) {
        int prec = getBinaryPrecedence(current_.kind);
        if (prec <= minPrecedence) {
            break;
        }
        Token opTok = advance();
        BinaryOp op = tokenToBinaryOp(opTok.kind);
        auto rhs = parseExpr(prec);
        if (!rhs) return nullptr;
        SourceRange range = {lhs->range.begin, rhs->range.end};
        auto bin = std::make_unique<BinaryExpr>(op, std::move(lhs), std::move(rhs));
        bin->range = range;
        lhs = std::move(bin);
    }
    return lhs;
}

std::unique_ptr<Expr> Parser::parseUnary() {
    SourceLocation startLoc = current_.loc;
    if (match(TokenKind::Minus)) {
        auto operand = parseUnary();
        if (!operand) return nullptr;
        auto expr = std::make_unique<UnaryExpr>(UnaryOp::Neg, std::move(operand));
        expr->range = {startLoc, expr->operand->range.end};
        return expr;
    }
    if (match(TokenKind::Bang)) {
        auto operand = parseUnary();
        if (!operand) return nullptr;
        auto expr = std::make_unique<UnaryExpr>(UnaryOp::Not, std::move(operand));
        expr->range = {startLoc, expr->operand->range.end};
        return expr;
    }
    if (match(TokenKind::Tilde)) {
        auto operand = parseUnary();
        if (!operand) return nullptr;
        auto expr = std::make_unique<UnaryExpr>(UnaryOp::BitNot, std::move(operand));
        expr->range = {startLoc, expr->operand->range.end};
        return expr;
    }
    return parsePrimary();
}

std::unique_ptr<Expr> Parser::parseCast() {
    SourceLocation startLoc = current_.loc;
    consume(TokenKind::KW_cast, "expected 'cast'");
    consume(TokenKind::Less, "expected '<' after 'cast'");
    Type targetType = parseType();
    consume(TokenKind::Greater, "expected '>' after type in cast");
    consume(TokenKind::LParen, "expected '(' after cast type");
    auto operand = parseExpr();
    SourceLocation endLoc = current_.loc;
    consume(TokenKind::RParen, "expected ')' after cast expression");
    auto castExpr = std::make_unique<CastExpr>(targetType, std::move(operand));
    castExpr->range = {startLoc, endLoc};
    return castExpr;
}

std::unique_ptr<Expr> Parser::parsePrimary() {
    SourceLocation startLoc = current_.loc;
    std::unique_ptr<Expr> expr = nullptr;

    if (check(TokenKind::IntLiteral)) {
        Token tok = advance();
        expr = std::make_unique<IntLiteralExpr>(tok.intVal, std::string(tok.text));
        expr->range = {tok.loc, tok.loc};
    } else if (check(TokenKind::KW_true) || check(TokenKind::KW_false)) {
        Token tok = advance();
        expr = std::make_unique<BoolLiteralExpr>(tok.kind == TokenKind::KW_true);
        expr->range = {tok.loc, tok.loc};
    } else if (check(TokenKind::FloatLiteral)) {
        Token tok = advance();
        expr = std::make_unique<FloatLiteralExpr>(tok.floatVal);
        expr->range = {tok.loc, tok.loc};
    } else if (check(TokenKind::KW_cast)) {
        expr = parseCast();
    } else if (check(TokenKind::Identifier)) {
        Token tok = advance();
        if (tok.text == "thread_idx") {
            expr = std::make_unique<BuiltinExpr>(BuiltinKind::ThreadIdx);
            expr->range = {tok.loc, tok.loc};
        } else if (tok.text == "block_idx") {
            expr = std::make_unique<BuiltinExpr>(BuiltinKind::BlockIdx);
            expr->range = {tok.loc, tok.loc};
        } else if (tok.text == "block_dim") {
            expr = std::make_unique<BuiltinExpr>(BuiltinKind::BlockDim);
            expr->range = {tok.loc, tok.loc};
        } else if (match(TokenKind::LParen)) {
            std::vector<std::unique_ptr<Expr>> args;
            if (!check(TokenKind::RParen)) {
                do {
                    auto arg = parseExpr();
                    if (!arg) return nullptr;
                    args.push_back(std::move(arg));
                } while (match(TokenKind::Comma));
            }
            SourceLocation endLoc = current_.loc;
            consume(TokenKind::RParen, "expected ')' after argument list");
            auto call = std::make_unique<CallExpr>(std::string(tok.text), std::move(args));
            call->range = {tok.loc, endLoc};
            expr = std::move(call);
        } else {
            expr = std::make_unique<IdentExpr>(std::string(tok.text));
            expr->range = {tok.loc, tok.loc};
        }
    } else if (match(TokenKind::LParen)) {
        expr = parseExpr();
        consume(TokenKind::RParen, "expected ')' after expression");
    } else {
        diag_.error(current_.loc, "expected expression");
        hasError_ = true;
        return nullptr;
    }

    // Handle postfix indexing: [index]
    while (match(TokenKind::LBracket)) {
        auto index = parseExpr();
        SourceLocation endLoc = current_.loc;
        consume(TokenKind::RBracket, "expected ']' after index");
        auto idxExpr = std::make_unique<IndexExpr>(std::move(expr), std::move(index));
        idxExpr->range = {startLoc, endLoc};
        expr = std::move(idxExpr);
    }

    return expr;
}

} // namespace kernc
