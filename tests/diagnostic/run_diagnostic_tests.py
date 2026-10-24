#!/usr/bin/env python3
import sys
import os
import subprocess
import glob

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <kernc_binary> <diagnostic_dir>")
        sys.exit(1)

    kernc = sys.argv[1]
    diag_dir = sys.argv[2]

    kern_files = sorted(glob.glob(os.path.join(diag_dir, "err_*.kern")))
    if not kern_files:
        print(f"No diagnostic test files found in {diag_dir}")
        sys.exit(1)

    all_passed = True
    print(f"Running {len(kern_files)} diagnostic tests...")

    for f in kern_files:
        expected_error = None
        with open(f, "r") as fh:
            for line in fh:
                if "Expected error:" in line:
                    expected_error = line.split("Expected error:")[1].strip()
                    break

        cmd = [kernc, f]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if proc.returncode == 0:
            print(f"[FAIL] {os.path.basename(f)}: expected error, but kernc succeeded")
            all_passed = False
            continue

        if expected_error and expected_error not in proc.stderr:
            print(f"[FAIL] {os.path.basename(f)}: expected error substring '{expected_error}' not in stderr:\n{proc.stderr}")
            all_passed = False
            continue

        print(f"[PASS] {os.path.basename(f)}: correctly rejected with expected error")

    if all_passed:
        print("All diagnostic tests passed successfully.")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
