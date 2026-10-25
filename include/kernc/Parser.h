#pragma once

#include "kernc/AST.h"
#include "kernc/Lexer.h"
#include "kernc/Diagnostics.h"
#include <memory>
#include <string>

namespace kernc {

class Parser {
public:
    Parser(Lexer& lexer, DiagnosticsEngine& diag);

    std::unique_ptr<Program> parseProgram();
    std::unique_ptr<KernelDecl> parseKernel();

private:
    Token current() const { return current_; }
    Token advance();
    bool match(TokenKind kind);
    bool check(TokenKind kind) const;
    Token consume(TokenKind kind, const std::string& errorMsg);

    // Types & Params
    Type parseType();
    Param parseParam();

    // Statements
    std::unique_ptr<Stmt> parseStmt();
    std::unique_ptr<Stmt> parseVarDecl();
    std::unique_ptr<Stmt> parseIfStmt();
    std::unique_ptr<Stmt> parseWhileStmt();
    std::unique_ptr<Stmt> parseReturnStmt();
    std::unique_ptr<Stmt> parseSharedDecl();
    std::unique_ptr<Stmt> parseForStmt();
    std::unique_ptr<BlockStmt> parseBlock();
    std::unique_ptr<Stmt> parseExprOrAssignStmt();

    // Expressions
    std::unique_ptr<Expr> parseExpr(int minPrecedence = 0);
    std::unique_ptr<Expr> parsePrimary();
    std::unique_ptr<Expr> parseUnary();
    std::unique_ptr<Expr> parseCast();

    int getBinaryPrecedence(TokenKind kind) const;
    BinaryOp tokenToBinaryOp(TokenKind kind) const;

    Lexer& lexer_;
    DiagnosticsEngine& diag_;
    Token current_;
    bool hasError_ = false;
};

} // namespace kernc
