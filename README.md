# FDTD 2D — Parallel Electromagnetic Simulation

2D Finite-Difference Time-Domain (FDTD) solver for Maxwell's equations,
parallelized with **OpenMP** (CPU) and **CUDA** (GPU).

---

## Features

- Three implementations sharing a common interface: Sequential, OpenMP, CUDA
- Two boundary conditions: Dirichlet PEC cavity and Perfectly Matched Layer (PML)
- Runtime grid resizing — no recompilation needed for different sizes
- VTK output (`.vti`) compatible with ParaView
- Full benchmark suite with Roofline analysis

---

## Requirements

| Tool | Version |
|------|---------|
| GCC | ≥ 10 (C++17, `-fopenmp`) |
| CUDA Toolkit | ≥ 11.0 |
| GNU Make | any |
| ParaView | ≥ 5.10 (visualization) |

**Before compiling**, check that the GPU architecture in the `makefile` matches your card:

```makefile
NVCCFLAGS = -arch=sm_89   # sm_75=Turing, sm_86=Ampere, sm_89=Ada, sm_90=Hopper
```

---

## Build

```bash
make            # full build → produces ./fdtd2d
make clean      # remove all .o and binary
```

---

## Usage

```bash
./fdtd2d <mode>
```

| Mode | Description |
|------|-------------|
| `seq` | Sequential CPU — Dirichlet (PEC cavity) |
| `omp` | OpenMP CPU — Dirichlet (PEC cavity) |
| `cuda` | CUDA GPU — Dirichlet (PEC cavity) |
| `seq-pml` | Sequential CPU — PML (open space) |
| `omp-pml` | OpenMP CPU — PML (open space) |
| `cuda-pml` | CUDA GPU — PML (open space) |
| `bench` | Full benchmark suite (all implementations, multiple grid sizes) |

**Examples:**

```bash
# Run sequential simulation with Dirichlet BC
./fdtd2d seq

# Run CUDA simulation with PML absorbing boundary
./fdtd2d cuda-pml

# Full benchmark (redirect to file for later plotting)
./fdtd2d bench 2>&1 | tee bench_results.txt
```

---

## Output

VTK files are written to `output/` (created automatically):

```
output/
  seq_000000.vti    ← step 0
  seq_000200.vti    ← step 200
  ...
  cuda_000000.vti
  ...
```

Each `.vti` file contains the field arrays `Ez`
as double-precision data on the full grid.

---

## Visualization (ParaView)

1. **File → Open** → select `output/seq_000000.vti`
   ParaView auto-detects the full time series.
2. Click **Apply**, then set coloring to `Ez`.
3. Click **Rescale to Data Range**.
4. Press **Play** to animate.
5. **File → Save Animation** to export as MP4 or PNG series.

---

## Parameters

Key simulation parameters are in `fdtd_simple.h` and `main.cpp`:

```cpp
// fdtd_simple.h
inline int NX = 200;       // grid points along x
inline int NY = 200;       // grid points along y
constexpr double DX = 1e-3; // spatial step [m]

// main.cpp
int total_steps   = 5000;
int save_interval = 200;   // write VTK every N steps
```

---


## Project Structure

```
fdtd_simple.h/cpp   — shared types (Grid, Source, BC, VTK writer)
fdtd_seq.h/cpp      — sequential implementation
fdtd_omp.h/cpp      — OpenMP implementation
fdtd_cuda.h/.cu     — CUDA kernels and implementation
main.cpp            — entry point, all modes
makefile            — build rules
```
