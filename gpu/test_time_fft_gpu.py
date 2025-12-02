import argparse
import cupy as cp


def benchmark_fft(n: int, repeats: int, warmup: int) -> float:
    """Benchmark 1D FFT on a complex vector of length n."""
    if warmup >= repeats:
        raise ValueError("warmup_fft must be < repeats_fft")

    # Fixed input on GPU
    x = (cp.random.randn(n, dtype=cp.float32) +
         1j * cp.random.randn(n, dtype=cp.float32))

    effective_runs = repeats - warmup
    total_us = 0.0

    for r in range(repeats):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        y = cp.fft.fft(x)
        end.record()
        end.synchronize()

        # elapsed_time returns milliseconds
        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        elapsed_us = elapsed_ms * 1000.0

        # Touch result so it cannot be fully optimized away
        _ = y[0].real + y[0].imag

        if r >= warmup:
            total_us += elapsed_us

    avg_us = total_us / effective_runs
    return avg_us


def correctness_check(n: int) -> float:
    """Compute relative reconstruction error of ifft(fft(x)) / n vs x."""
    x = (cp.random.randn(n, dtype=cp.float32) +
         1j * cp.random.randn(n, dtype=cp.float32))

    y = cp.fft.fft(x)
    x_rec = cp.fft.ifft(y)

    # FFT convention may differ by factor n; adjust accordingly
    x_rec_scaled = x_rec / float(n)

    num = cp.linalg.norm(x - x_rec_scaled)
    den = cp.linalg.norm(x)
    rel_err = (num / den).item()
    return rel_err


def main():
    parser = argparse.ArgumentParser(
        description="GPU 1D FFT benchmark using CuPy"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="Vector lengths to test (default: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)",
    )
    parser.add_argument(
        "--repeats-fft",
        type=int,
        default=110,
        help="Total repetitions per size for FFT (default: 110)",
    )
    parser.add_argument(
        "--warmup-fft",
        type=int,
        default=10,
        help="Number of warm-up runs to drop for FFT (default: 10)",
    )

    args = parser.parse_args()

    device_id = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    dev_name = props["name"].decode()

    print("GPU FFT benchmark (CuPy cp.fft.fft)")
    print(f"Device {device_id}: {dev_name}")
    print(f"FFT: {args.repeats_fft} runs per size, "
          f"dropping first {args.warmup_fft} as warm-up.\n")

    # Force CUDA context creation upfront
    cp.zeros((1,), dtype=cp.float32).sum()

    for n in args.sizes:
        print(f"Vector size: {n}")

        avg_fft_us = benchmark_fft(
            n,
            repeats=args.repeats_fft,
            warmup=args.warmup_fft,
        )
        print(f"  FFT: avg {avg_fft_us:.3f} us over "
              f"{args.repeats_fft - args.warmup_fft} runs "
              f"({args.warmup_fft} warm-up dropped)")

        # correctness check (not timed)
        rel_err = correctness_check(n)
        print("  Relative reconstruction error "
              "||x - ifft(fft(x))/n||_2 / ||x||_2: "
              f"{rel_err:.3e}\n")


if __name__ == "__main__":
    main()