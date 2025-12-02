#include <armadillo>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::micro>; // us

int main() {
    // -------------------------------
    // 0. Single-core / single-thread setup
    // -------------------------------

    // Try to hint BLAS/OpenMP libraries via environment variables (POSIX)
    // These are best set *before* program start, but setting here often still helps.
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("BLIS_NUM_THREADS", "1", 1);

    // Initialize Armadillo RNG
    arma::arma_rng::set_seed_random();

    // Matrix sizes to test
    std::vector<std::size_t> sizes = {
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
    };

    // -------------------------------
    // 1. Set warm-up and repetition parameters
    // -------------------------------
    // Number of repetitions for averaging runtime
    const int repeats_matvec = 110;
    const int repeats_inv    = 110;

    // Default number of iterations to drop for warm-up
    int warmup_matvec = 10;  // dropped from mat-vec average
    int warmup_inv    = 10;  // dropped from inversion average

    // Safety: make sure warmup < repeats
    if (warmup_matvec >= repeats_matvec) {
        warmup_matvec = repeats_matvec - 1;
    }
    if (warmup_inv >= repeats_inv) {
        warmup_inv = repeats_inv - 1;
    }

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Testing matrix-vector multiplication and matrix inversion\n";
    std::cout << "Armardillo version: " << arma::arma_version::as_string()
              << "\n\n";
    std::cout << "Threads forced to 1 (where possible).\n";
    std::cout << "Mat-vec: " << repeats_matvec << " runs, dropping first "
              << warmup_matvec << " as warm-up.\n";
    std::cout << "Inversion: " << repeats_inv << " runs, dropping first "
              << warmup_inv << " as warm-up.\n\n";


    for (int n : sizes) {
        std::cout << "Matrix size: " << n << " x " << n << "\n";

        // Generate a random complex-float matrix and vector
        arma::cx_fmat A = arma::randn<arma::cx_fmat>(n, n);
        arma::cx_fvec x = arma::randn<arma::cx_fvec>(n);

        // -------------------------------
        // 2. Matrix-vector multiplication benchmark
        // -------------------------------
        Duration total_mv(0);
        int effective_runs_mv = repeats_matvec - warmup_matvec;

        for (int r = 0; r < repeats_matvec; ++r) {
            auto t0 = Clock::now();
            arma::cx_fvec y = A * x;
            auto t1 = Clock::now();

            if (r >= warmup_matvec) {
                total_mv += std::chrono::duration_cast<Duration>(t1 - t0);
            }
        }

        double avg_mv_us = total_mv.count() / static_cast<double>(effective_runs_mv);
        std::cout << "  MVM: avg " << avg_mv_us << " us over "
                  << effective_runs_mv << " runs ("
                  << warmup_matvec << " warm-up dropped)\n";

        // -------------------------------
        // 3. Matrix inversion benchmark
        // -------------------------------
        Duration total_inv(0);
        double max_residual = 0.0;
        int effective_runs_inv = repeats_inv - warmup_inv;

        for (int r = 0; r < repeats_inv; ++r) {

            arma::cx_fmat A_curr = arma::randn<arma::cx_fmat>(n, n);

            auto t0 = Clock::now();
            arma::cx_fmat Ainv = arma::inv(A_curr);
            auto t1 = Clock::now();

            if (r >= warmup_inv) {
                total_inv += std::chrono::duration_cast<Duration>(t1 - t0);
            }

            // Simple correctness check: ||A * A^{-1} - I||_F
            arma::cx_fmat I_approx = A_curr * Ainv;
            arma::cx_fmat I        = arma::eye<arma::cx_fmat>(n, n);
            double residual    = arma::norm(I_approx - I, "fro");
            if (residual > max_residual) {
                max_residual = residual;
            }
        }

        double avg_inv_us = total_inv.count() / static_cast<double>(effective_runs_inv);
        std::cout << "  MatInv: avg " << avg_inv_us << " us over "
                  << effective_runs_inv << " runs ("
                  << warmup_inv << " warm-up dropped)\n";
        std::cout << "  Max Frobenius residual ||A*A^{-1}-I||_F: "
                  << max_residual << "\n\n";

        std::cout.flush();
    }

    return 0;
}