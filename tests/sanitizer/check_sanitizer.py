#!/usr/bin/env python3
"""NVIDIA compute-sanitizer on kernc output, end to end.

Usage: check_sanitizer.py <compute-sanitizer> <kernc> <kernc-runner> <examples_dir> <sanitizer_dir> <workdir>

1. memcheck: an out-of-bounds write compiled with `kernc -lineinfo` must be reported against
   its .kern file and line (the .loc line info survives the driver JIT).
2. racecheck: examples/reduce.kern must have no shared-memory hazards.
3. racecheck: the same reduction with its barrier() calls removed must report a race.

Exits 77 (CTest "skipped") when the GPU debugger interface is not enabled. On WSL that needs
the Windows registry value HKLM\\SOFTWARE\\NVIDIA Corporation\\GPUDebugger\\EnableInterface = 1.
"""
import os
import re
import subprocess
import sys

SKIP = 77


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    if "Failed to initialize WDDM debugger interface" in p.stdout or "Device not supported" in p.stdout:
        print("SKIPPED: compute-sanitizer cannot attach (GPU debugger interface not enabled)")
        sys.exit(SKIP)
    return p.stdout


def main():
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    sanitizer, kernc, runner, examples, sandir, work = sys.argv[1:]
    os.makedirs(work, exist_ok=True)
    failures = []

    # 1. memcheck with line info
    oob_src = os.path.join(sandir, "oob.kern")
    with open(oob_src) as fh:
        oob_line = next(n for n, line in enumerate(fh, 1) if "out[i + 4096]" in line)
    oob_ptx = os.path.join(work, "oob.ptx")
    subprocess.run([kernc, oob_src, "-lineinfo", "-o", oob_ptx], check=True)
    out = run([sanitizer, "--tool", "memcheck", runner, oob_ptx, "oob", "-g", "1", "-b", "32",
               "--arg", "buf:i32:16:zeros", "--arg", "i32:16"])
    if "Invalid __global__ write" not in out or f"oob.kern:{oob_line}" not in out:
        failures.append(f"memcheck did not report the write at oob.kern:{oob_line}:\n{out[:2000]}")

    # 2./3. racecheck on the shipped reduction, with and without its barriers
    reduce_src = os.path.join(examples, "reduce.kern")
    with open(reduce_src) as fh:
        source = fh.read()
    no_barrier_src = os.path.join(work, "reduce_nobar.kern")
    with open(no_barrier_src, "w") as fh:
        fh.write(source.replace("barrier();", ""))
    args = ["reduce_sum", "-g", "4", "-b", "256",
            "--arg", "buf:f32:1024:seq", "--arg", "buf:f32:4:zeros", "--arg", "i32:1024"]
    for src, expect_race in [(reduce_src, False), (no_barrier_src, True)]:
        ptx = os.path.join(work, os.path.basename(src).replace(".kern", ".ptx"))
        subprocess.run([kernc, src, "-lineinfo", "-o", ptx], check=True)
        out = run([sanitizer, "--tool", "racecheck", runner, ptx] + args)
        summary = re.search(r"RACECHECK SUMMARY: (\d+) hazard", out)
        if not summary:
            failures.append(f"no racecheck summary for {src}:\n{out[:2000]}")
        elif expect_race and (int(summary.group(1)) == 0 or "Race reported" not in out):
            failures.append(f"racecheck missed the race in {src} (barriers removed):\n{out[:2000]}")
        elif not expect_race and int(summary.group(1)) != 0:
            failures.append(f"racecheck found hazards in {src}:\n{out[:2000]}")

    if failures:
        print("\n\n".join(failures))
        sys.exit(1)
    print(f"memcheck reported oob.kern:{oob_line}; racecheck: reduce.kern clean, "
          "barrier-free variant reported a race")


if __name__ == "__main__":
    main()
