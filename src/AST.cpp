#include "kernc/AST.h"

namespace kernc {

const char* binaryOpToString(BinaryOp op) {
    switch (op) {
        case BinaryOp::Add: return "+";
        case BinaryOp::Sub: return "-";
        case BinaryOp::Mul: return "*";
        case BinaryOp::Div: return "/";
        case BinaryOp::Mod: return "%";
        case BinaryOp::Lt:  return "<";
        case BinaryOp::Gt:  return ">";
        case BinaryOp::Le:  return "<=";
        case BinaryOp::Ge:  return ">=";
        case BinaryOp::Eq:  return "==";
        case BinaryOp::Ne:  return "!=";
        case BinaryOp::And: return "&&";
        case BinaryOp::Or:  return "||";
    }
    return "??";
}

const char* unaryOpToString(UnaryOp op) {
    switch (op) {
        case UnaryOp::Neg: return "-";
        case UnaryOp::Not: return "!";
    }
    return "?";
}

} // namespace kernc
