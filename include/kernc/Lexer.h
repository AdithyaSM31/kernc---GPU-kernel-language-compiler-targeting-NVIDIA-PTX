#pragma once

#include "kernc/Token.h"
#include "kernc/Diagnostics.h"
#include <string_view>
#include <string>

namespace kernc {

class Lexer {
public:
    explicit Lexer(std::string_view source, DiagnosticsEngine* diag = nullptr);

    Token next();
    Token peek();

private:
    char peekChar() const;
    char peekNextChar() const;
    char getChar();
    bool isAtEnd() const;
    void skipWhitespaceAndComments();
    Token lexNumber(SourceLocation startLoc);
    Token lexIdentifierOrKeyword(SourceLocation startLoc);

    std::string_view source_;
    DiagnosticsEngine* diag_;
    size_t cursor_ = 0;
    uint32_t line_ = 1;
    uint32_t column_ = 1;

    bool hasPeeked_ = false;
    Token peekedToken_;
};

} // namespace kernc
