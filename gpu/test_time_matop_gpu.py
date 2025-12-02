import argparse
import cupy as cp


def benchmark_matvec(n, repeats, warmup):
    # Fixed random matrix/vector on GPU
    A = (cp.random.randn(n, n, dtype=cp.float32) +
        1j * cp.random.randn(n, n, dtype=cp.float32))
    x = (cp.random.randn(n, dtype=cp.float32) +
        1j * cp.random.randn(n, dtype=cp.float32))

    total_us = 0.0
    effective_runs = repeats - warmup
    if effective_runs <= 0:
        raise ValueError("warmup_matvec must be < repeats_matvec")

    for r in range(repeats):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        y = A @ x
        end.record()
        end.synchronize()

        # elapsed_time gives milliseconds as float
        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        elapsed_us = elapsed_ms * 1000.0

        if r >= warmup:
            total_us += elapsed_us

    avg_us = total_us / effective_runs
    return avg_us


def benchmark_inv(n, repeats, warmup):
    total_us = 0.0
    effective_runs = repeats - warmup
    if effective_runs <= 0:
        raise ValueError("warmup_inv must be < repeats_inv")

    max_residual = 0.0

    for r in range(repeats):
        # fresh random matrix each time
        A = (cp.random.randn(n, n, dtype=cp.float32) +
             1j * cp.random.randn(n, n, dtype=cp.float32))

        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        Ainv = cp.linalg.inv(A)
        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        elapsed_us = elapsed_ms * 1000.0

        if r >= warmup:
            total_us += elapsed_us

        # correctness check: ||A * A^{-1} - I||_F
        I_approx = A @ Ainv
        I = cp.eye(n, dtype=cp.complex64)
        residual = cp.linalg.norm(I_approx - I, ord="fro").item()
        if residual > max_residual:
            max_residual = residual

    avg_us = total_us / effective_runs
    return avg_us, max_residual


def main():
    parser = argparse.ArgumentParser(
        description="GPU benchmark: matrix-vector and matrix inversion using CuPy"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="Matrix sizes to test (default: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)",
    )
    parser.add_argument(
        "--repeats-matvec",
        type=int,
        default=110,
        help="Total repetitions for mat-vec (default: 110)",
    )
    parser.add_argument(
        "--repeats-inv",
        type=int,
        default=110,
        help="Total repetitions for inversion (default: 110)",
    )
    parser.add_argument(
        "--warmup-matvec",
        type=int,
        default=10,
        help="Number of mat-vec warm-up runs to drop (default: 10)",
    )
    parser.add_argument(
        "--warmup-inv",
        type=int,
        default=10,
        help="Number of inversion warm-up runs to drop (default: 10)",
    )
    parser.add_argument(
        "--max-inv-size",
        type=int,
        default=9000,
        help="Skip inversion for sizes larger than this (default: 9000)",
    )

    args = parser.parse_args()

    print("GPU linear algebra benchmark (CuPy)")
    print(f"Device: {cp.cuda.runtime.getDevice()} - {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"Mat-vec: {args.repeats_matvec} runs, dropping first {args.warmup_matvec} as warm-up.")
    print(f"Inversion: {args.repeats_inv} runs, dropping first {args.warmup_inv} as warm-up.")
    print()

    # Just to initialize the cuda context upfront
    cp.zeros((1,), dtype=cp.float32).sum()

    for n in args.sizes:
        print(f"Matrix size: {n} x {n}")

        # Mat-vec benchmark
        avg_mv_us = benchmark_matvec(n, args.repeats_matvec, args.warmup_matvec)
        print(f"  Mat-vec: avg {avg_mv_us:.3f} us over "
              f"{args.repeats_matvec - args.warmup_matvec} runs "
              f"({args.warmup_matvec} warm-up dropped)")

        # Inversion benchmark
        if n > args.max_inv_size:
            print(f"  Inversion: skipped (n > {args.max_inv_size})\n")
            continue

        avg_inv_us, max_residual = benchmark_inv(n, args.repeats_inv, args.warmup_inv)
        print(f"  Inversion: avg {avg_inv_us:.3f} us over "
              f"{args.repeats_inv - args.warmup_inv} runs "
              f"({args.warmup_inv} warm-up dropped)")
        print(f"  Max Frobenius residual ||A*A^(-1)-I||_F: {max_residual:.3e}\n")


if __name__ == "__main__":
    main()
