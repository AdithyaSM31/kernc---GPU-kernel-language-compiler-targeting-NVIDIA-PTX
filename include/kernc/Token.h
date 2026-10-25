#pragma once

#include "kernc/SourceLocation.h"
#include <string_view>
#include <string>

namespace kernc {

enum class TokenKind {
    // Literals
    IntLiteral,
    FloatLiteral,

    // Identifiers & keywords
    Identifier,
    KW_kernel,
    KW_int,
    KW_float,
    KW_if,
    KW_else,
    KW_while,
    KW_for,
    KW_return,
    KW_cast,
    KW_shared,
    KW_barrier,

    // Operators
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    Less,       // <
    Greater,    // >
    LessEq,     // <=
    GreaterEq,  // >=
    EqEq,       // ==
    BangEq,     // !=
    AmpAmp,     // &&
    PipePipe,   // ||
    Bang,       // !
    Eq,         // =

    // Delimiters
    LParen,     // (
    RParen,     // )
    LBrace,     // {
    RBrace,     // }
    LBracket,   // [
    RBracket,   // ]
    Semicolon,  // ;
    Comma,      // ,

    // Special
    Eof,
    Error
};

struct Token {
    TokenKind kind = TokenKind::Eof;
    std::string_view text;
    SourceLocation loc;
    int64_t intVal = 0;
    double floatVal = 0.0;

    std::string toString() const;
    static const char* kindToString(TokenKind kind);
};

} // namespace kernc
