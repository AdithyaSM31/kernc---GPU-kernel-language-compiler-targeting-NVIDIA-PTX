#include <gtest/gtest.h>
#include "kernc/Lexer.h"
#include "kernc/Diagnostics.h"

using namespace kernc;

TEST(LexerTest, Keywords) {
    std::string src = "kernel int float if else while for cast shared barrier";
    Lexer lex(src);

    EXPECT_EQ(lex.next().kind, TokenKind::KW_kernel);
    EXPECT_EQ(lex.next().kind, TokenKind::KW_int);
    EXPECT_EQ(lex.next().kind, TokenKind::KW_float);
    EXPECT_EQ(lex.next().kind, TokenKind::KW_if);
    EXPECT_EQ(lex.next().kind, TokenKind::KW_else);
    EXPECT_EQ(lex.next().kind, TokenKind::KW_while);
    EXPECT_EQ(lex.next().kind, TokenKind::KW_for);
    EXPECT_EQ(lex.next().kind, TokenKind::KW_cast);
    EXPECT_EQ(lex.next().kind, TokenKind::KW_shared);
    EXPECT_EQ(lex.next().kind, TokenKind::KW_barrier);
    EXPECT_EQ(lex.next().kind, TokenKind::Eof);
}

TEST(LexerTest, IntegerLiterals) {
    std::string src = "0 42 1000 0x10 0xFF 0Xabcd";
    Lexer lex(src);

    Token t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::IntLiteral);
    EXPECT_EQ(t.intVal, 0);

    t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::IntLiteral);
    EXPECT_EQ(t.intVal, 42);

    t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::IntLiteral);
    EXPECT_EQ(t.intVal, 1000);

    t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::IntLiteral);
    EXPECT_EQ(t.intVal, 16);

    t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::IntLiteral);
    EXPECT_EQ(t.intVal, 255);

    t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::IntLiteral);
    EXPECT_EQ(t.intVal, 0xabcd);
}

TEST(LexerTest, FloatLiterals) {
    std::string src = "3.14 1.0f .5 1e-3 2.5e+2";
    Lexer lex(src);

    Token t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::FloatLiteral);
    EXPECT_NEAR(t.floatVal, 3.14, 1e-6);

    t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::FloatLiteral);
    EXPECT_NEAR(t.floatVal, 1.0, 1e-6);

    t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::FloatLiteral);
    EXPECT_NEAR(t.floatVal, 0.5, 1e-6);

    t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::FloatLiteral);
    EXPECT_NEAR(t.floatVal, 0.001, 1e-7);

    t = lex.next();
    EXPECT_EQ(t.kind, TokenKind::FloatLiteral);
    EXPECT_NEAR(t.floatVal, 250.0, 1e-6);
}

TEST(LexerTest, OperatorsAndPunctuation) {
    std::string src = "+ - * / % < <= > >= == != && || ! = ( ) { } [ ] ; ,";
    Lexer lex(src);

    EXPECT_EQ(lex.next().kind, TokenKind::Plus);
    EXPECT_EQ(lex.next().kind, TokenKind::Minus);
    EXPECT_EQ(lex.next().kind, TokenKind::Star);
    EXPECT_EQ(lex.next().kind, TokenKind::Slash);
    EXPECT_EQ(lex.next().kind, TokenKind::Percent);
    EXPECT_EQ(lex.next().kind, TokenKind::Less);
    EXPECT_EQ(lex.next().kind, TokenKind::LessEq);
    EXPECT_EQ(lex.next().kind, TokenKind::Greater);
    EXPECT_EQ(lex.next().kind, TokenKind::GreaterEq);
    EXPECT_EQ(lex.next().kind, TokenKind::EqEq);
    EXPECT_EQ(lex.next().kind, TokenKind::BangEq);
    EXPECT_EQ(lex.next().kind, TokenKind::AmpAmp);
    EXPECT_EQ(lex.next().kind, TokenKind::PipePipe);
    EXPECT_EQ(lex.next().kind, TokenKind::Bang);
    EXPECT_EQ(lex.next().kind, TokenKind::Eq);
    EXPECT_EQ(lex.next().kind, TokenKind::LParen);
    EXPECT_EQ(lex.next().kind, TokenKind::RParen);
    EXPECT_EQ(lex.next().kind, TokenKind::LBrace);
    EXPECT_EQ(lex.next().kind, TokenKind::RBrace);
    EXPECT_EQ(lex.next().kind, TokenKind::LBracket);
    EXPECT_EQ(lex.next().kind, TokenKind::RBracket);
    EXPECT_EQ(lex.next().kind, TokenKind::Semicolon);
    EXPECT_EQ(lex.next().kind, TokenKind::Comma);
    EXPECT_EQ(lex.next().kind, TokenKind::Eof);
}

TEST(LexerTest, SourceLocations) {
    std::string src = "int x;\nfloat y;";
    Lexer lex(src);

    Token t1 = lex.next();
    EXPECT_EQ(t1.kind, TokenKind::KW_int);
    EXPECT_EQ(t1.loc.line, 1);
    EXPECT_EQ(t1.loc.column, 1);

    Token t2 = lex.next();
    EXPECT_EQ(t2.kind, TokenKind::Identifier);
    EXPECT_EQ(t2.loc.line, 1);
    EXPECT_EQ(t2.loc.column, 5);

    lex.next(); // ';'

    Token t3 = lex.next();
    EXPECT_EQ(t3.kind, TokenKind::KW_float);
    EXPECT_EQ(t3.loc.line, 2);
    EXPECT_EQ(t3.loc.column, 1);
}

TEST(LexerTest, Comments) {
    std::string src = "// single line comment\nint /* inline comment */ x;";
    Lexer lex(src);

    Token t1 = lex.next();
    EXPECT_EQ(t1.kind, TokenKind::KW_int);
    EXPECT_EQ(t1.loc.line, 2);

    Token t2 = lex.next();
    EXPECT_EQ(t2.kind, TokenKind::Identifier);
    EXPECT_EQ(t2.text, "x");

    Token t3 = lex.next();
    EXPECT_EQ(t3.kind, TokenKind::Semicolon);
    EXPECT_EQ(lex.next().kind, TokenKind::Eof);
}
