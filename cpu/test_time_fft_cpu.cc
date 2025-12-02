#include <armadillo>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using Clock    = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::micro>;  // microseconds

int main() {
    // ----------------------------------------------------
    // 0. Force single-thread execution (best-effort)
    // ----------------------------------------------------
    // These affect BLAS/OpenMP libraries and possibly FFT backends
    setenv("OMP_NUM_THREADS",      "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS",      "1", 1);
    setenv("BLIS_NUM_THREADS",     "1", 1);

    // ----------------------------------------------------
    // 1. Parse warm-up / repeat settings
    // ----------------------------------------------------
    int repeats_fft  = 110;  // total FFT runs per size
    int warmup_fft   = 10;   // how many runs to drop for warm-up

    if (warmup_fft >= repeats_fft) {
        std::cerr << "warmup_fft must be < repeats_fft; adjusting.\n";
        warmup_fft = repeats_fft - 1;
    }

    int effective_runs_fft = repeats_fft - warmup_fft;

    // Initialize RNG
    arma::arma_rng::set_seed_random();

    // FFT sizes to test (feel free to tweak)
    std::vector<std::size_t> sizes = {
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
    };

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU FFT benchmark using Armadillo (arma::fft)\n";
    std::cout << "Armadillo version: " << arma::arma_version::as_string() << "\n";
    std::cout << "Threads forced to 1 (where possible).\n";
    std::cout << "FFT: " << repeats_fft << " runs per size, dropping first "
              << warmup_fft << " as warm-up.\n\n";

    // A volatile sink to keep the compiler from optimizing away FFT results
    volatile float sink = 0.0f;

    for (std::size_t n : sizes) {
        std::cout << "Vector size: " << n << "\n";

        // Pre-generate input data (complex vector)
        arma::cx_fvec x = arma::randn<arma::cx_fvec>(n);

        Duration total_fft(0);

        for (int r = 0; r < repeats_fft; ++r) {
            auto t0 = Clock::now();
            arma::cx_fvec y = arma::fft(x);
            auto t1 = Clock::now();

            // Touch the result so the call can't be optimized away
            sink += std::abs(y(0));

            if (r >= warmup_fft) {
                total_fft += std::chrono::duration_cast<Duration>(t1 - t0);
            }
        }

        double avg_fft_us = total_fft.count() /
                            static_cast<double>(effective_runs_fft);

        std::cout << "  FFT: avg " << avg_fft_us << " us over "
                  << effective_runs_fft << " runs ("
                  << warmup_fft << " warm-up dropped)\n";

        // ----------------------------------------------------
        // 2. Simple correctness check (not timed)
        //    Check if ifft(fft(x)) ≈ x (up to scaling by n)
        // ----------------------------------------------------
        arma::cx_fvec y = arma::fft(x);
        arma::cx_fvec x_rec = arma::ifft(y);

        // Many FFT conventions differ by a factor of n; scale accordingly
        arma::cx_fvec x_rec_scaled = x_rec / static_cast<float>(n);

        double rel_error = arma::norm(x - x_rec_scaled, 2) /
                           arma::norm(x, 2);

        std::cout << "  Relative reconstruction error "
                     "||x - ifft(fft(x))/n||_2 / ||x||_2: "
                  << rel_error << "\n\n";

        std::cout.flush();
    }

    // Prevent optimizer from removing the entire loop
    if (sink == 123456.789f) {
        std::cerr << "Impossible sink value: " << sink << "\n";
    }

    return 0;
}