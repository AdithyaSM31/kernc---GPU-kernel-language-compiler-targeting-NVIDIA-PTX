#pragma once

#include "kernc/SourceLocation.h"
#include <string>
#include <vector>
#include <iostream>

namespace kernc {

enum class DiagnosticLevel {
    Error,
    Warning,
    Note
};

struct Diagnostic {
    DiagnosticLevel level;
    SourceLocation loc;
    std::string message;
};

class DiagnosticsEngine {
public:
    DiagnosticsEngine(std::string filename = "<input>", std::string source = "")
        : filename_(std::move(filename)), source_(std::move(source)) {}

    void setSource(std::string filename, std::string source) {
        filename_ = std::move(filename);
        source_ = std::move(source);
    }

    void report(DiagnosticLevel level, SourceLocation loc, const std::string& message);
    void error(SourceLocation loc, const std::string& message) {
        report(DiagnosticLevel::Error, loc, message);
    }
    void warning(SourceLocation loc, const std::string& message) {
        report(DiagnosticLevel::Warning, loc, message);
    }
    void note(SourceLocation loc, const std::string& message) {
        report(DiagnosticLevel::Note, loc, message);
    }

    bool hasErrors() const { return errorCount_ > 0; }
    size_t errorCount() const { return errorCount_; }
    const std::vector<Diagnostic>& getDiagnostics() const { return diagnostics_; }

    void clear() {
        diagnostics_.clear();
        errorCount_ = 0;
    }

    std::string formatDiagnostic(const Diagnostic& diag) const;
    std::string formatAll() const;
    void printAll(std::ostream& os = std::cerr) const;

private:
    std::string getSourceLine(uint32_t line) const;

    std::string filename_;
    std::string source_;
    std::vector<Diagnostic> diagnostics_;
    size_t errorCount_ = 0;
};

} // namespace kernc
