#pragma once

#include <cstdint>
#include <string>

namespace kernc {

struct SourceLocation {
    uint32_t line = 1;
    uint32_t column = 1;

    bool operator==(const SourceLocation& other) const {
        return line == other.line && column == other.column;
    }
    bool operator!=(const SourceLocation& other) const {
        return !(*this == other);
    }
};

struct SourceRange {
    SourceLocation begin;
    SourceLocation end;
};

} // namespace kernc
