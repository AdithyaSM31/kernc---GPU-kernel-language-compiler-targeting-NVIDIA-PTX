#include "kernc/Diagnostics.h"
#include <sstream>

namespace kernc {

void DiagnosticsEngine::report(DiagnosticLevel level, SourceLocation loc, const std::string& message) {
    if (level == DiagnosticLevel::Error) {
        errorCount_++;
    }
    diagnostics_.push_back({level, loc, message});
}

std::string DiagnosticsEngine::getSourceLine(uint32_t line) const {
    if (source_.empty() || line == 0) return "";
    uint32_t curLine = 1;
    size_t start = 0;
    while (curLine < line && start < source_.size()) {
        if (source_[start] == '\n') {
            curLine++;
        }
        start++;
    }
    if (curLine != line || start >= source_.size()) {
        return "";
    }
    size_t end = start;
    while (end < source_.size() && source_[end] != '\n' && source_[end] != '\r') {
        end++;
    }
    return source_.substr(start, end - start);
}

std::string DiagnosticsEngine::formatDiagnostic(const Diagnostic& diag) const {
    std::ostringstream ss;
    ss << filename_ << ":" << diag.loc.line << ":" << diag.loc.column << ": ";
    switch (diag.level) {
        case DiagnosticLevel::Error:   ss << "error: "; break;
        case DiagnosticLevel::Warning: ss << "warning: "; break;
        case DiagnosticLevel::Note:    ss << "note: "; break;
    }
    ss << diag.message << "\n";

    std::string lineStr = getSourceLine(diag.loc.line);
    if (!lineStr.empty()) {
        ss << "  " << lineStr << "\n";
        ss << "  ";
        uint32_t col = diag.loc.column > 0 ? diag.loc.column - 1 : 0;
        for (uint32_t i = 0; i < col && i < lineStr.size(); ++i) {
            if (lineStr[i] == '\t') ss << '\t';
            else ss << ' ';
        }
        ss << "^\n";
    }
    return ss.str();
}

std::string DiagnosticsEngine::formatAll() const {
    std::string result;
    for (const auto& diag : diagnostics_) {
        result += formatDiagnostic(diag);
    }
    return result;
}

void DiagnosticsEngine::printAll(std::ostream& os) const {
    os << formatAll();
}

} // namespace kernc
