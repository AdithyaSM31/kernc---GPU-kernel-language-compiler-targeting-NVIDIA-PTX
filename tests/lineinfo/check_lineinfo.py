#!/usr/bin/env python3
"""Checks kernc's .file/.loc output end to end through ptxas and nvdisasm.

Usage: check_lineinfo.py <kernc> <ptxas> <nvdisasm> <file.kern> <workdir> <opt>

Compiles <file.kern> with `kernc -lineinfo <opt>`, assembles it with ptxas, and reads the
SASS line table with `nvdisasm --print-line-info`. Then:
  * every "//@ REGEX" line in the source must have a SASS instruction matching REGEX
    attributed to it;
  * every attributed line must contain code (not blank, not a comment-only line);
  * the file name recorded in the table must be the .kern file.
"""
import os
import re
import subprocess
import sys


def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{proc.stdout}{proc.stderr}")
    return proc.stdout


def main():
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    kernc, ptxas, nvdisasm, src, workdir, opt = sys.argv[1:]
    name = os.path.splitext(os.path.basename(src))[0]
    ptx = os.path.join(workdir, f"{name}{opt}.ptx")
    cubin = os.path.join(workdir, f"{name}{opt}.cubin")
    os.makedirs(workdir, exist_ok=True)

    run([kernc, src, "-lineinfo", opt, "-o", ptx])
    run([ptxas, "-arch=sm_86", ptx, "-o", cubin])
    listing = run([nvdisasm, "--print-line-info", "-c", cubin])

    # Map source line -> SASS instructions attributed to it.
    sass_by_line = {}
    files = set()
    current = None
    for text in listing.splitlines():
        marker = re.search(r'//## File "([^"]+)", line (\d+)', text)
        if marker:
            files.add(marker.group(1))
            current = int(marker.group(2))
            continue
        insn = re.search(r"/\*[0-9a-f]{4}\*/\s+(.*?)\s*;", text)
        if insn and current is not None:
            sass_by_line.setdefault(current, []).append(insn.group(1))

    with open(src) as fh:
        source = fh.read().splitlines()

    failures = []
    if not sass_by_line:
        failures.append("nvdisasm reported no line information")
    for f in files:
        if os.path.basename(f) != os.path.basename(src):
            failures.append(f"line table names file '{f}', expected {os.path.basename(src)}")

    for line in sorted(sass_by_line):
        code = source[line - 1].strip() if 0 < line <= len(source) else ""
        if not code or code.startswith("//"):
            failures.append(f"line {line} has no code but SASS is attributed to it: "
                            f"{sass_by_line[line]}")

    for lineno, text in enumerate(source, start=1):
        # Annotations are trailing comments on code lines; comment-only lines are prose.
        if text.strip().startswith("//"):
            continue
        want = re.search(r"//@\s*(.+?)\s*$", text)
        if not want:
            continue
        pattern = want.group(1)
        got = sass_by_line.get(lineno, [])
        if not any(re.search(pattern, insn) for insn in got):
            failures.append(f"line {lineno} expects SASS matching /{pattern}/, got {got}")

    if failures:
        print(f"[FAIL] {name} {opt}")
        for f in failures:
            print("  " + f)
        print("--- line table ---")
        for line in sorted(sass_by_line):
            print(f"  {line:3}: {', '.join(sass_by_line[line])}")
        sys.exit(1)
    print(f"[PASS] {name} {opt}: {len(sass_by_line)} source lines mapped")


if __name__ == "__main__":
    main()
