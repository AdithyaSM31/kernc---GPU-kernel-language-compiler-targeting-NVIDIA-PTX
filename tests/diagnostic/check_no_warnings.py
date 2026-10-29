#!/usr/bin/env python3
"""Every shipped kernel must compile without warnings.

Usage: check_no_warnings.py <kernc_binary> <file.kern>...

A kernel that is meant to trigger warnings (for example one that reads an uninitialized
variable on purpose) must say so with one or more "// Expected warning: <text>" lines; it
must then produce each of those warnings. Every other kernel must produce no warnings at all,
so a diagnostic such as the barrier-divergence check also guards the project's own kernels.
"""
import os
import subprocess
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    kernc, files = sys.argv[1], sys.argv[2:]
    failures = []
    for path in files:
        with open(path) as fh:
            expected = [line.split("Expected warning:", 1)[1].strip()
                        for line in fh if "Expected warning:" in line]
        proc = subprocess.run([kernc, path, "-o", os.devnull],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            failures.append(f"{path}: kernc failed\n{proc.stderr}")
            continue
        warnings = [line for line in proc.stderr.splitlines() if "warning:" in line]
        name = path
        if expected:
            for text in expected:
                if not any(text in w for w in warnings):
                    failures.append(f"{name}: expected warning '{text}' was not reported")
        elif warnings:
            failures.append(f"{name}: unexpected warning(s):\n  " + "\n  ".join(warnings))

    if failures:
        print("\n".join(failures))
        sys.exit(1)
    print(f"{len(files)} kernels checked, no unexpected warnings")


if __name__ == "__main__":
    main()
