#!/usr/bin/env python3
import sys
import os
import subprocess
import re

def run_cmd(cmd, expect_zero=True):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if expect_zero and proc.returncode != 0:
        print(f"Command failed (exit code {proc.returncode}): {' '.join(cmd)}")
        print("STDOUT:\n" + proc.stdout)
        print("STDERR:\n" + proc.stderr)
        sys.exit(1)
    if not expect_zero and proc.returncode == 0:
        print(f"Command unexpectedly succeeded (exit code 0): {' '.join(cmd)}")
        print("STDOUT:\n" + proc.stdout)
        print("STDERR:\n" + proc.stderr)
        sys.exit(1)
    return proc

def main():
    if len(sys.argv) < 5:
        print("Usage: test_runner.py <kernc_bin> <runner_bin> <examples_dir> <temp_dir>")
        sys.exit(1)

    kernc = sys.argv[1]
    runner = sys.argv[2]
    examples_dir = sys.argv[3]
    temp_dir = sys.argv[4]

    os.makedirs(temp_dir, exist_ok=True)

    print("=== Test 1: --help CLI output ===")
    p = run_cmd([runner, "--help"], expect_zero=True)
    for flag in ["--arg", "--grid", "--block", "--dump"]:
        assert flag in p.stdout, f"Expected '{flag}' in --help output"

    print("=== Test 2: No arguments prints usage and returns non-zero ===")
    p = run_cmd([runner], expect_zero=False)
    combined = p.stdout + p.stderr
    for flag in ["--arg", "--grid", "--block", "--dump"]:
        assert flag in combined, f"Expected '{flag}' in usage output when run with no arguments"

    print("=== Test 3: VecAdd execution and buffer dump ===")
    vecadd_kern = os.path.join(examples_dir, "vecadd.kern")
    vecadd_ptx = os.path.join(temp_dir, "vecadd.ptx")
    run_cmd([kernc, vecadd_kern, "-o", vecadd_ptx], expect_zero=True)

    p = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--grid", "2", "--block", "256",
        "--arg", "buf:i32:512:seq",
        "--arg", "buf:i32:512:seq",
        "--arg", "buf:i32:512:zeros",
        "--arg", "i32:512",
        "--dump", "2:8"
    ], expect_zero=True)

    assert "[Runner] Loaded GPU:" in p.stdout
    assert "[Runner] Found kernel: vecadd" in p.stdout
    assert "[Runner] Kernel execution completed successfully." in p.stdout
    assert "[Runner] Success!" in p.stdout

    # Verify dumped values: a[i]=i, b[i]=i => c[i] = 2*i
    for i in range(8):
        expected_line = f"[{i}] = {2 * i}"
        assert expected_line in p.stdout, f"Missing '{expected_line}' in dump output:\n{p.stdout}"

    print("=== Test 4: SAXPY execution with float scalars & buffers ===")
    saxpy_kern = os.path.join(examples_dir, "saxpy.kern")
    saxpy_ptx = os.path.join(temp_dir, "saxpy.ptx")
    run_cmd([kernc, saxpy_kern, "-o", saxpy_ptx], expect_zero=True)

    p = run_cmd([
        runner, saxpy_ptx, "saxpy",
        "--grid", "2", "--block", "256",
        "--arg", "f32:2.5",
        "--arg", "buf:f32:512:seq",
        "--arg", "buf:f32:512:zeros",
        "--arg", "i32:512",
        "--dump", "2:4"
    ], expect_zero=True)

    assert "[Runner] Found kernel: saxpy" in p.stdout
    assert "[Runner] Success!" in p.stdout

    # Parse float dump values: y[i] = 2.5 * i + 0
    dump_matches = re.findall(r"\[(\d+)\]\s*=\s*([0-9\.\-eE]+)", p.stdout)
    assert len(dump_matches) >= 4, f"Expected at least 4 dump elements, found {len(dump_matches)}"
    for idx_str, val_str in dump_matches[:4]:
        idx = int(idx_str)
        val = float(val_str)
        expected = 2.5 * idx
        assert abs(val - expected) < 1e-4, f"Mismatch at index {idx}: got {val}, expected {expected}"

    print("=== Test 5: Flagged args and verbose JIT logging ===")
    p = run_cmd([
        runner,
        "--ptx", vecadd_ptx,
        "--kernel", "vecadd",
        "-v",
        "-g", "1",
        "-b", "128",
        "--arg", "buf:i32:128:rand",
        "--arg", "buf:i32:128:zeros",
        "--arg", "buf:i32:128:zeros",
        "--arg", "i32:128",
        "--dump", "0:4"
    ], expect_zero=True)
    assert "[Runner] JIT compilation successful." in p.stdout
    assert "[Runner] Success!" in p.stdout

    print("=== Test 6: Default dump count and dump count 0 ===")
    p = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--arg", "buf:i32:32:seq",
        "--arg", "buf:i32:32:seq",
        "--arg", "buf:i32:32:zeros",
        "--arg", "i32:32",
        "--dump", "2"
    ], expect_zero=True)
    assert "[Runner] Dump arg 2 (i32, count=16):" in p.stdout
    assert "[15] = 30" in p.stdout

    # Dump count 0 prints header and zero elements
    p_zero = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:zeros",
        "--arg", "i32:16",
        "--dump", "2:0"
    ], expect_zero=True)
    assert "[Runner] Dump arg 2 (i32, count=0):" in p_zero.stdout
    assert "[0] =" not in p_zero.stdout

    print("=== Test 7: 32-bit Hex scalar argument support ===")
    # 0xFFFFFFFF (-1) and 0x80000000
    p = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:zeros",
        "--arg", "i32:0xFFFFFFFF",
        "--dump", "2:2"
    ], expect_zero=True)
    assert "[Runner] Success!" in p.stdout

    p = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:zeros",
        "--arg", "i32:0x80000000",
        "--dump", "2:2"
    ], expect_zero=True)
    assert "[Runner] Success!" in p.stdout

    print("=== Test 8: Substring / Prefix Kernel Name Disambiguation ===")
    multi_entry_ptx = os.path.join(temp_dir, "multi_entry.ptx")
    with open(multi_entry_ptx, "w") as f:
        f.write("""
.version 7.8
.target sm_86
.address_size 64

.visible .entry vecadd_scalar(
    .param .u64 p0,
    .param .u64 p1,
    .param .u32 p2
)
{
    ret;
}

.visible .entry vecadd(
    .param .u64 p0,
    .param .u64 p1,
    .param .u64 p2,
    .param .u32 p3
)
{
    ret;
}
""")
    p = run_cmd([
        runner, multi_entry_ptx, "vecadd",
        "--arg", "buf:i32:16:zeros",
        "--arg", "buf:i32:16:zeros",
        "--arg", "buf:i32:16:zeros",
        "--arg", "i32:16"
    ], expect_zero=True)
    assert "[Runner] Found kernel: vecadd" in p.stdout
    assert "[Runner] Success!" in p.stdout

    # Now run vecadd_scalar which expects 3 arguments
    p_sc = run_cmd([
        runner, multi_entry_ptx, "vecadd_scalar",
        "--arg", "buf:i32:16:zeros",
        "--arg", "buf:i32:16:zeros",
        "--arg", "i32:16"
    ], expect_zero=True)
    assert "[Runner] Found kernel: vecadd_scalar" in p_sc.stdout
    assert "[Runner] Success!" in p_sc.stdout

    print("=== Test 9: PTX Comments Handling ===")
    commented_ptx = os.path.join(temp_dir, "commented.ptx")
    with open(commented_ptx, "w") as f:
        f.write("""
.version 7.8
.target sm_86
.address_size 64

// .visible .entry k_commented(.param .u64 p0, .param .u64 p1)
/*
.visible .entry k_commented2(
    .param .u64 p0
)
*/

.visible .entry k(
    // .param .u64 commented_param,
    /* .param .u64 commented_param2, */
    .param .u64 p0,
    .param .u32 p1
)
{
    ret;
}
""")
    p = run_cmd([
        runner, commented_ptx, "k",
        "--arg", "buf:i32:16:zeros",
        "--arg", "i32:16"
    ], expect_zero=True)
    assert "[Runner] Found kernel: k" in p.stdout
    assert "[Runner] Success!" in p.stdout

    print("=== Test 10: Zero-parameter kernel ===")
    zero_param_ptx = os.path.join(temp_dir, "zero_param.ptx")
    with open(zero_param_ptx, "w") as f:
        f.write("""
.version 7.8
.target sm_86
.address_size 64

.visible .entry init_system()
{
    ret;
}
""")
    p = run_cmd([runner, zero_param_ptx, "init_system"], expect_zero=True)
    assert "[Runner] Found kernel: init_system" in p.stdout
    assert "[Runner] Success!" in p.stdout

    print("=== Test 11: Error handling (invalid PTX, bad kernel, bad dump) ===")
    # Non-existent file
    run_cmd([runner, "nonexistent.ptx", "vecadd"], expect_zero=False)

    # Missing kernel symbol
    run_cmd([runner, vecadd_ptx, "nonexistent_kernel"], expect_zero=False)

    # Argument count mismatch
    run_cmd([runner, vecadd_ptx, "vecadd", "--arg", "i32:10"], expect_zero=False)

    # Dump index out of range
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:i32:16:seq", "--arg", "buf:i32:16:seq",
             "--arg", "buf:i32:16:zeros", "--arg", "i32:16",
             "--dump", "10"], expect_zero=False)

    # Dump scalar argument
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:i32:16:seq", "--arg", "buf:i32:16:seq",
             "--arg", "buf:i32:16:zeros", "--arg", "i32:16",
             "--dump", "3"], expect_zero=False)

    # Negative dump index
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:i32:16:seq", "--arg", "buf:i32:16:seq",
             "--arg", "buf:i32:16:zeros", "--arg", "i32:16",
             "--dump", "-1"], expect_zero=False)

    # Negative dump count
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:i32:16:seq", "--arg", "buf:i32:16:seq",
             "--arg", "buf:i32:16:zeros", "--arg", "i32:16",
             "--dump", "2:-5"], expect_zero=False)

    # Negative buffer element count
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:i32:-1:zeros", "--arg", "buf:i32:16:zeros",
             "--arg", "buf:i32:16:zeros", "--arg", "i32:16"], expect_zero=False)

    # Extra colon in buffer spec
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:i32:16:zeros:extra", "--arg", "buf:i32:16:zeros",
             "--arg", "buf:i32:16:zeros", "--arg", "i32:16"], expect_zero=False)

    # Invalid buffer type
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:invalid:16:zeros", "--arg", "buf:i32:16:zeros",
             "--arg", "buf:i32:16:zeros", "--arg", "i32:16"], expect_zero=False)

    # Invalid buffer init mode
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:i32:16:invalid_mode", "--arg", "buf:i32:16:zeros",
             "--arg", "buf:i32:16:zeros", "--arg", "i32:16"], expect_zero=False)

    # Invalid PTX JIT error log capture
    bad_ptx = os.path.join(temp_dir, "bad.ptx")
    with open(bad_ptx, "w") as f:
        f.write(".version 7.8\n.target sm_86\n.address_size 64\n.visible .entry bad() { bad_instruction; }\n")
    p_bad = run_cmd([runner, bad_ptx, "bad"], expect_zero=False)
    assert "JIT Error Log:" in p_bad.stderr or "CUDA Error" in p_bad.stderr

    print("=== Test 12: Directory PTX path rejection without crash ===")
    p_dir = run_cmd([runner, temp_dir, "vecadd"], expect_zero=False)
    assert "is a directory" in p_dir.stderr.lower() or "failed to read" in p_dir.stderr.lower()

    print("=== Test 13: Decimal i32 overflow & float overflow rejection ===")
    # Positive decimal > INT32_MAX must be rejected
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:i32:16:zeros", "--arg", "buf:i32:16:zeros",
             "--arg", "buf:i32:16:zeros", "--arg", "i32:3000000000"], expect_zero=False)
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:i32:16:zeros", "--arg", "buf:i32:16:zeros",
             "--arg", "buf:i32:16:zeros", "--arg", "i32:2147483648"], expect_zero=False)
    # Float overflow (> ~3.4e38) must be rejected
    run_cmd([runner, saxpy_ptx, "saxpy",
             "--arg", "f32:1e45", "--arg", "buf:f32:16:zeros",
             "--arg", "buf:f32:16:zeros", "--arg", "i32:16"], expect_zero=False)

    print("=== Test 14: Block comments without newlines token separation ===")
    block_comment_ptx = os.path.join(temp_dir, "block_comment.ptx")
    with open(block_comment_ptx, "w") as f:
        f.write("""
.version 7.8
.target sm_86
.address_size 64

.visible/*inline_comment*/.entry inline_block_k(
    .param .u64 p0
)
{
    ret;
}
""")
    p = run_cmd([runner, block_comment_ptx, "inline_block_k", "--arg", "buf:i32:16:zeros"], expect_zero=True)
    assert "[Runner] Found kernel: inline_block_k" in p.stdout
    assert "[Runner] Success!" in p.stdout

    print("=== Test 15: String literal containing .entry or .param directives ===")
    string_ptx = os.path.join(temp_dir, "string_ptx.ptx")
    with open(string_ptx, "w") as f:
        f.write("""
.version 7.8
.target sm_86
.address_size 64
.file 1 "/custom/.entry string_kernel/test.ptx"
.file 2 "/path/.param not_a_param/foo.ptx"

.visible .entry string_kernel(
    .param .u64 p0,
    .param .u32 p1
)
{
    ret;
}
""")
    p = run_cmd([runner, string_ptx, "string_kernel",
                 "--arg", "buf:i32:16:zeros", "--arg", "i32:16"], expect_zero=True)
    assert "[Runner] Found kernel: string_kernel" in p.stdout
    assert "[Runner] Success!" in p.stdout

    print("=== Test 16: Random float buffer initialization ===")
    p = run_cmd([
        runner, saxpy_ptx, "saxpy",
        "--arg", "f32:1.0",
        "--arg", "buf:f32:64:rand",
        "--arg", "buf:f32:64:zeros",
        "--arg", "i32:64",
        "--dump", "1:4"
    ], expect_zero=True)
    assert "[Runner] Success!" in p.stdout
    assert "[Runner] Dump arg 1 (f32, count=4):" in p.stdout

    print("=== Test 17: Multiple dumps, count clamping, and small buffer default ===")
    # Multiple dumps in a single invocation
    p = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:zeros",
        "--arg", "i32:16",
        "--dump", "0:2",
        "--dump", "2:2"
    ], expect_zero=True)
    assert "[Runner] Dump arg 0 (i32, count=2):" in p.stdout
    assert "[Runner] Dump arg 2 (i32, count=2):" in p.stdout

    # Dump count > N clamps to N without error
    p_clamp = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--arg", "buf:i32:8:seq",
        "--arg", "buf:i32:8:seq",
        "--arg", "buf:i32:8:zeros",
        "--arg", "i32:8",
        "--dump", "2:100"
    ], expect_zero=True)
    assert "[Runner] Dump arg 2 (i32, count=8):" in p_clamp.stdout

    # Small buffer (<16) default dump count is N, not 16
    p_small = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--arg", "buf:i32:5:seq",
        "--arg", "buf:i32:5:seq",
        "--arg", "buf:i32:5:zeros",
        "--arg", "i32:5",
        "--dump", "2"
    ], expect_zero=True)
    assert "[Runner] Dump arg 2 (i32, count=5):" in p_small.stdout

    print("=== Test 18: Decimal integers with leading zeros (08, 09, 010) ===")
    # 08 and 09 must be accepted as decimals 8 and 9, not rejected as bad octal
    p_dec08 = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:zeros",
        "--arg", "i32:08",
        "--dump", "2:8"
    ], expect_zero=True)
    assert "[7] = 14" in p_dec08.stdout

    # 010 must be treated as decimal 10 (computing 10 elements), not octal 8
    p_dec010 = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:zeros",
        "--arg", "i32:010",
        "--dump", "2:10"
    ], expect_zero=True)
    assert "[9] = 18" in p_dec010.stdout

    # Negative leading zero decimal
    p_neg08 = run_cmd([
        runner, vecadd_ptx, "vecadd",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:seq",
        "--arg", "buf:i32:16:zeros",
        "--arg", "i32:-08"
    ], expect_zero=True)
    assert "[Runner] Success!" in p_neg08.stdout

    print("=== Test 19: PTX with unterminated block comment at EOF ===")
    eof_comment_ptx = os.path.join(temp_dir, "eof_comment.ptx")
    with open(eof_comment_ptx, "w") as f:
        f.write("""
.version 7.8
.target sm_86
.address_size 64

.visible .entry eof_kernel(
    .param .u64 p0
)
{
    ret;
}
/* unterminated block comment at end of file without closing delimiter
""")
    p_eof = run_cmd([runner, eof_comment_ptx, "eof_kernel", "--arg", "buf:i32:8:zeros"], expect_zero=False)
    assert "JIT Error Log:" in p_eof.stderr or "CUDA Error" in p_eof.stderr

    print("=== Test 20: Rejection of embedded whitespace in scalar arguments ===")
    run_cmd([runner, vecadd_ptx, "vecadd",
             "--arg", "buf:i32:16:zeros", "--arg", "buf:i32:16:zeros",
             "--arg", "buf:i32:16:zeros", "--arg", "i32: 10"], expect_zero=False)
    run_cmd([runner, saxpy_ptx, "saxpy",
             "--arg", "f32: 2.5", "--arg", "buf:f32:16:zeros",
             "--arg", "buf:f32:16:zeros", "--arg", "i32:16"], expect_zero=False)

    print("=== Test 21: --arg types must match the PTX .param types ===")
    # saxpy's parameters are (.f32 a, .u64 x, .u64 y, .s32 n). A mismatched --arg used to be
    # launched anyway: an i32 for a pointer faulted with CUDA_ERROR_ILLEGAL_ADDRESS, and an i32
    # for a float or a buffer for an int reported Success! with silently wrong results.
    mismatches = [
        (["i32:2", "buf:f32:8:seq", "buf:f32:8:seq", "i32:8"], 0, "float"),
        (["f32:2", "i32:5", "buf:f32:8:seq", "i32:8"], 1, "pointer"),
        (["f32:2", "buf:f32:8:seq", "buf:f32:8:seq", "buf:i32:8:seq"], 3, "int"),
        (["f32:2", "buf:f32:8:seq", "buf:f32:8:seq", "f32:8"], 3, "int"),
    ]
    for specs, bad_index, expected_kind in mismatches:
        cmd = [runner, saxpy_ptx, "saxpy", "--grid", "1", "--block", "8"]
        for s in specs:
            cmd += ["--arg", s]
        p_bad = run_cmd(cmd, expect_zero=False)
        err = p_bad.stderr
        assert f"argument {bad_index}" in err and expected_kind in err, \
            f"Expected a type error for argument {bad_index} ({expected_kind}), got:\n{err}"
        assert "Launching kernel" not in p_bad.stdout, \
            f"Mismatched arguments must be rejected before launch:\n{p_bad.stdout}"

    print("All runner tests passed cleanly!")

if __name__ == "__main__":
    main()
