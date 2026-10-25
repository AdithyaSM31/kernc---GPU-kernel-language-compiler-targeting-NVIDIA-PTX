#include "kernc/Lexer.h"
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace kernc {

const char* Token::kindToString(TokenKind kind) {
    switch (kind) {
        case TokenKind::IntLiteral:   return "IntLiteral";
        case TokenKind::FloatLiteral: return "FloatLiteral";
        case TokenKind::Identifier:   return "Identifier";
        case TokenKind::KW_kernel:    return "kernel";
        case TokenKind::KW_int:       return "int";
        case TokenKind::KW_float:     return "float";
        case TokenKind::KW_if:        return "if";
        case TokenKind::KW_else:      return "else";
        case TokenKind::KW_while:     return "while";
        case TokenKind::KW_for:       return "for";
        case TokenKind::KW_return:    return "return";
        case TokenKind::KW_bool:      return "bool";
        case TokenKind::KW_true:      return "true";
        case TokenKind::KW_false:     return "false";
        case TokenKind::KW_cast:      return "cast";
        case TokenKind::KW_shared:    return "shared";
        case TokenKind::KW_barrier:   return "barrier";
        case TokenKind::Plus:         return "+";
        case TokenKind::Minus:        return "-";
        case TokenKind::Star:         return "*";
        case TokenKind::Slash:        return "/";
        case TokenKind::Percent:      return "%";
        case TokenKind::Less:         return "<";
        case TokenKind::Greater:      return ">";
        case TokenKind::LessEq:       return "<=";
        case TokenKind::GreaterEq:    return ">=";
        case TokenKind::EqEq:         return "==";
        case TokenKind::BangEq:       return "!=";
        case TokenKind::AmpAmp:       return "&&";
        case TokenKind::PipePipe:     return "||";
        case TokenKind::Bang:         return "!";
        case TokenKind::Eq:           return "=";
        case TokenKind::LParen:       return "(";
        case TokenKind::RParen:       return ")";
        case TokenKind::LBrace:       return "{";
        case TokenKind::RBrace:       return "}";
        case TokenKind::LBracket:     return "[";
        case TokenKind::RBracket:     return "]";
        case TokenKind::Semicolon:    return ";";
        case TokenKind::Comma:        return ",";
        case TokenKind::Eof:          return "<EOF>";
        case TokenKind::Error:        return "<ERROR>";
    }
    return "<UNKNOWN>";
}

std::string Token::toString() const {
    std::ostringstream ss;
    ss << kindToString(kind) << "('" << text << "') at " << loc.line << ":" << loc.column;
    return ss.str();
}

Lexer::Lexer(std::string_view source, DiagnosticsEngine* diag)
    : source_(source), diag_(diag) {}

char Lexer::peekChar() const {
    if (isAtEnd()) return '\0';
    return source_[cursor_];
}

char Lexer::peekNextChar() const {
    if (cursor_ + 1 >= source_.size()) return '\0';
    return source_[cursor_ + 1];
}

char Lexer::getChar() {
    if (isAtEnd()) return '\0';
    char c = source_[cursor_++];
    if (c == '\n') {
        line_++;
        column_ = 1;
    } else {
        column_++;
    }
    return c;
}

bool Lexer::isAtEnd() const {
    return cursor_ >= source_.size();
}

void Lexer::skipWhitespaceAndComments() {
    while (!isAtEnd()) {
        char c = peekChar();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            getChar();
        } else if (c == '/' && peekNextChar() == '/') {
            // Line comment
            getChar(); // '/'
            getChar(); // '/'
            while (!isAtEnd() && peekChar() != '\n') {
                getChar();
            }
        } else if (c == '/' && peekNextChar() == '*') {
            // Block comment
            SourceLocation commentLoc{line_, column_};
            getChar(); // '/'
            getChar(); // '*'
            bool closed = false;
            while (!isAtEnd()) {
                if (peekChar() == '*' && peekNextChar() == '/') {
                    getChar(); // '*'
                    getChar(); // '/'
                    closed = true;
                    break;
                }
                getChar();
            }
            if (!closed && diag_) {
                diag_->error(commentLoc, "unclosed block comment");
            }
        } else {
            break;
        }
    }
}

Token Lexer::lexNumber(SourceLocation startLoc) {
    size_t startOffset = cursor_ - 1; // already consumed the first digit or '.'
    bool isHex = false;
    bool isFloat = (source_[startOffset] == '.');

    if (source_[startOffset] == '0' && (peekChar() == 'x' || peekChar() == 'X')) {
        isHex = true;
        getChar(); // consume 'x' or 'X'
        while (std::isxdigit(peekChar())) {
            getChar();
        }
    } else {
        // Decimal int or float
        while (std::isdigit(peekChar())) {
            getChar();
        }

        if (!isFloat && peekChar() == '.') {
            isFloat = true;
            getChar(); // consume '.'
            while (std::isdigit(peekChar())) {
                getChar();
            }
        }

        if (peekChar() == 'e' || peekChar() == 'E') {
            isFloat = true;
            getChar(); // consume 'e'
            if (peekChar() == '+' || peekChar() == '-') {
                getChar();
            }
            while (std::isdigit(peekChar())) {
                getChar();
            }
        }

        if (peekChar() == 'f' || peekChar() == 'F') {
            isFloat = true;
            getChar(); // consume 'f' or 'F'
        }
    }

    size_t length = cursor_ - startOffset;
    std::string_view numText = source_.substr(startOffset, length);

    Token tok;
    tok.loc = startLoc;
    tok.text = numText;

    std::string str(numText);
    char* endptr = nullptr;

    if (isFloat) {
        tok.kind = TokenKind::FloatLiteral;
        tok.floatVal = std::strtod(str.c_str(), &endptr);
    } else if (isHex) {
        tok.kind = TokenKind::IntLiteral;
        tok.intVal = std::strtoll(str.c_str(), &endptr, 16);
    } else {
        tok.kind = TokenKind::IntLiteral;
        tok.intVal = std::strtoll(str.c_str(), &endptr, 10);
    }

    return tok;
}

Token Lexer::lexIdentifierOrKeyword(SourceLocation startLoc) {
    size_t startOffset = cursor_ - 1; // already consumed first char
    while (std::isalnum(peekChar()) || peekChar() == '_') {
        getChar();
    }
    size_t length = cursor_ - startOffset;
    std::string_view text = source_.substr(startOffset, length);

    Token tok;
    tok.loc = startLoc;
    tok.text = text;

    if (text == "kernel")      tok.kind = TokenKind::KW_kernel;
    else if (text == "int")    tok.kind = TokenKind::KW_int;
    else if (text == "float")  tok.kind = TokenKind::KW_float;
    else if (text == "if")     tok.kind = TokenKind::KW_if;
    else if (text == "else")   tok.kind = TokenKind::KW_else;
    else if (text == "while")  tok.kind = TokenKind::KW_while;
    else if (text == "for")    tok.kind = TokenKind::KW_for;
    else if (text == "return") tok.kind = TokenKind::KW_return;
    else if (text == "bool")   tok.kind = TokenKind::KW_bool;
    else if (text == "true")   tok.kind = TokenKind::KW_true;
    else if (text == "false")  tok.kind = TokenKind::KW_false;
    else if (text == "cast")   tok.kind = TokenKind::KW_cast;
    else if (text == "__shared__" || text == "shared") tok.kind = TokenKind::KW_shared;
    else if (text == "barrier") tok.kind = TokenKind::KW_barrier;
    else tok.kind = TokenKind::Identifier;

    return tok;
}

Token Lexer::peek() {
    if (!hasPeeked_) {
        peekedToken_ = next();
        hasPeeked_ = true;
    }
    return peekedToken_;
}

Token Lexer::next() {
    if (hasPeeked_) {
        hasPeeked_ = false;
        return peekedToken_;
    }

    skipWhitespaceAndComments();

    SourceLocation loc{line_, column_};
    if (isAtEnd()) {
        Token tok;
        tok.kind = TokenKind::Eof;
        tok.loc = loc;
        tok.text = "";
        return tok;
    }

    size_t startOffset = cursor_;
    char c = getChar();

    // Numbers: digits or '.' followed by a digit
    if (std::isdigit(c) || (c == '.' && std::isdigit(peekChar()))) {
        return lexNumber(loc);
    }

    // Identifiers or keywords
    if (std::isalpha(c) || c == '_') {
        return lexIdentifierOrKeyword(loc);
    }

    // Operators and delimiters
    Token tok;
    tok.loc = loc;

    auto makeToken = [&](TokenKind kind, size_t len) {
        tok.kind = kind;
        tok.text = source_.substr(startOffset, len);
        return tok;
    };

    switch (c) {
        case '+': return makeToken(TokenKind::Plus, 1);
        case '-': return makeToken(TokenKind::Minus, 1);
        case '*': return makeToken(TokenKind::Star, 1);
        case '/': return makeToken(TokenKind::Slash, 1);
        case '%': return makeToken(TokenKind::Percent, 1);
        case '(': return makeToken(TokenKind::LParen, 1);
        case ')': return makeToken(TokenKind::RParen, 1);
        case '{': return makeToken(TokenKind::LBrace, 1);
        case '}': return makeToken(TokenKind::RBrace, 1);
        case '[': return makeToken(TokenKind::LBracket, 1);
        case ']': return makeToken(TokenKind::RBracket, 1);
        case ';': return makeToken(TokenKind::Semicolon, 1);
        case ',': return makeToken(TokenKind::Comma, 1);

        case '=':
            if (peekChar() == '=') {
                getChar();
                return makeToken(TokenKind::EqEq, 2);
            }
            return makeToken(TokenKind::Eq, 1);

        case '!':
            if (peekChar() == '=') {
                getChar();
                return makeToken(TokenKind::BangEq, 2);
            }
            return makeToken(TokenKind::Bang, 1);

        case '<':
            if (peekChar() == '=') {
                getChar();
                return makeToken(TokenKind::LessEq, 2);
            }
            return makeToken(TokenKind::Less, 1);

        case '>':
            if (peekChar() == '=') {
                getChar();
                return makeToken(TokenKind::GreaterEq, 2);
            }
            return makeToken(TokenKind::Greater, 1);

        case '&':
            if (peekChar() == '&') {
                getChar();
                return makeToken(TokenKind::AmpAmp, 2);
            }
            break;

        case '|':
            if (peekChar() == '|') {
                getChar();
                return makeToken(TokenKind::PipePipe, 2);
            }
            break;
    }

    // Unknown/Error token
    tok.kind = TokenKind::Error;
    tok.text = source_.substr(startOffset, 1);
    if (diag_) {
        diag_->error(loc, std::string("unexpected character '") + c + "'");
    }
    return tok;
}

} // namespace kernc
