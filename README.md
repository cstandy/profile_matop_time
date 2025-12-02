# Profile Matrix Operation Time on CPU/GPU

Small benchmark suite for linear-algebra operations (MVM/matinv/FFT) on CPU/GPU.

This repo provides two separate profiling methods:

- **GPU profiling (CuPy, Python)**: Python scripts under `gpu/` (`test_time_fft_gpu.py`, `test_time_matop_gpu.py`) that use CuPy to run benchmarks on an NVIDIA GPU. These require CUDA drivers/toolkit and a CuPy wheel compatible with your CUDA version. Use the `Pipfile` / `pipenv` instructions above to set up the environment and run the scripts.

- **CPU profiling (Armadillo, C++)**: Native C++ benchmarks under `cpu/`. The sources are `test_time_matop_cpu.cc` and `test_time_fft_cpu.cc`. The repository also contains prebuilt binaries named `test_time_matop_cpu` and `test_time_fft_cpu` (these may be built for another machine/ABI). Prefer rebuilding on your machine using the provided `Makefile`.


## Prerequisites

### GPU Prerequisites

- **CUDA driver & toolkit**: Required for GPU execution. Verify with `nvidia-smi` and optionally `nvcc --version`.
- **Python 3.10** (project Pipfile targets `3.10`).
- **Pipenv**: used to create the virtual environment and manage Python dependencies.

The repository `Pipfile` contains `cupy = "*"`. Installing plain `cupy` with pipenv may attempt to build from source which can be slow or fail. It's usually easier to install the prebuilt CuPy wheel that matches your CUDA version inside the pipenv environment (examples below).

Steps:
- Install pipenv (if needed):
```bash
python3 -m pip install --user pipenv
```
- Use Pipenv to create a virtualenv and open a shell:
```bash
pipenv --python 3.10
pipenv shell
```
- Install CuPy inside the pipenv shell:
```bash
# read from Pipfile
pipenv install
# OR install cupy directly
pipenv install cupy
```
- Verify installation:
```bash
pipenv shell
python -c "import cupy as cp; print(cp.__version__)"
```


### CPU Prerequisites
The `Makefile` uses `g++` and links `-larmadillo`.
Install `libarmadillo-dev` (or equivalent) on your distribution.

## Execution

### GPU profiling (CuPy)

Quick run (inside a pipenv shell with a matching `cupy-cuda***` wheel installed):
```bash
python gpu/test_time_matop_gpu.py --sizes 64 128 --repeats-matvec 110 --repeats-inv 110 --warmup-matvec 10 --warmup-inv 10
python gpu/test_time_fft_gpu.py --sizes 64 128 256 --repeats-fft 110 --warmup-fft 10
```

Notes:
- Ensure the active Python environment has `cupy` installed (prefer the prebuilt `cupy-cuda***` wheel matching your CUDA toolkit).
- GPU timings measure work done on the device; warm-up runs help stabilize setup/allocator overhead.

### CPU profiling (Armadillo C++)

Build and run from the `cpu/` directory:
```bash
cd cpu
make # compiles test_time_matop_cpu and test_time_fft_cpu (requires Armadillo and a C++ toolchain)
./test_time_matop_cpu
./test_time_fft_cpu
```

Notes and tips:
- The programs attempt to force single-threaded execution by setting environment variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`) but it's best to set these before running to ensure BLAS and FFT backends honor them:
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
./test_time_matop_cpu
```
- The C++ programs perform the same operations as the GPU scripts (complex float inputs, matrix-vector multiply, matrix inversion, and correctness checks) and print microsecond timings.
- If you already have the included binaries, you can run them directly; however rebuilding on your machine ensures ABI compatibility and will pick up optimized BLAS/FFT libraries present on your system.

