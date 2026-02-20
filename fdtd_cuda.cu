
#include "fdtd_cuda.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// -------------------------------------------------------------------
// Macro de vérification d'erreur CUDA
// -------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__    \
                      << "  " << cudaGetErrorString(_err) << std::endl;    \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// -------------------------------------------------------------------
// Kernels CUDA
// -------------------------------------------------------------------

// Hx(i,j) -= coeff * (Ez(i,j+1) - Ez(i,j))
__global__ void updateHx_kernel(double* __restrict__ Hx,
                                 const double* __restrict__ Ez, double coeff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NX && j < NY - 1) {
        int idx = i * NY + j;
        Hx[idx] -= coeff * (Ez[idx + 1] - Ez[idx]);
    }
}

// Hy(i,j) += coeff * (Ez(i+1,j) - Ez(i,j))
__global__ void updateHy_kernel(double* __restrict__ Hy,
                                 const double* __restrict__ Ez, double coeff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NX - 1 && j < NY) {
        int idx = i * NY + j;
        Hy[idx] += coeff * (Ez[idx + NY] - Ez[idx]);
    }
}

// Ez(i,j) += coeffE * [ (Hy-Hy_left)/DX - (Hx-Hx_below)/DY ]
// Lancé sur la grille intérieure avec offset +1
__global__ void updateEz_kernel(double* __restrict__ Ez,
                                 const double* __restrict__ Hx,
                                 const double* __restrict__ Hy,
                                 double coeffE, double invDX, double invDY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < NX - 1 && j < NY - 1) {
        int idx      = i * NY + j;
        int idx_left = idx - NY;
        int idx_down = idx - 1;
        double curl  = (Hy[idx] - Hy[idx_left]) * invDX
                     - (Hx[idx] - Hx[idx_down]) * invDY;
        Ez[idx] += coeffE * curl;
    }
}

// Source ponctuelle : 1 thread, écriture directe à src_idx pré-calculé
__global__ void applySource_kernel(double* Ez, double value, int src_idx) {
    Ez[src_idx] += value;
}

// -------------------------------------------------------------------
// Kernels Dirichlet (PEC) — parallèles sur les bords
// -------------------------------------------------------------------

__global__ void applyDirichlet_Ez_kernel(double* Ez) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NY) {
        Ez[0 * NY + tid]      = 0.0;
        Ez[(NX-1) * NY + tid] = 0.0;
    }
    if (tid < NX) {
        Ez[tid * NY + 0]      = 0.0;
        Ez[tid * NY + (NY-1)] = 0.0;
    }
}

__global__ void applyDirichlet_H_kernel(double* Hx, double* Hy) {
    // Reproduit exactement Dirichlet::applyH du CPU :
    //   Hx = 0 sur i=0 et i=NX-1 (pour tout j)
    //   Hy = 0 sur j=0 et j=NY-1 (pour tout i)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NY) {
        Hx[0 * NY + tid]      = 0.0;   // bord i=0
        Hx[(NX-1) * NY + tid] = 0.0;   // bord i=NX-1
    }
    if (tid < NX) {
        Hy[tid * NY + 0]      = 0.0;   // bord j=0
        Hy[tid * NY + (NY-1)] = 0.0;   // bord j=NY-1
    }
}

// -------------------------------------------------------------------
// Kernels PML
// -------------------------------------------------------------------

__global__ void applyPML_E_kernel(
    double* Ez,
    double* psi_Ezx, double* psi_Ezy,
    const double* be_x, const double* be_y,
    const double* ce_x, const double* ce_y,
    const double* Hx, const double* Hy,
    double invDY, double invDX, double coeffE) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX || j >= NY) return;
    int idx = i * NY + j;

    // psi_Ezx corrige partial_Hy/partial_x : difference selon i, coefficient sigma_x
    if (i > 0) {
        psi_Ezx[idx] = be_x[idx] * psi_Ezx[idx]
                     + ce_x[idx] * (Hy[idx] - Hy[idx - NY]) * invDX;
    }
    // psi_Ezy corrige partial_Hx/partial_y : difference selon j, coefficient sigma_y
    if (j > 0) {
        psi_Ezy[idx] = be_y[idx] * psi_Ezy[idx]
                     + ce_y[idx] * (Hx[idx] - Hx[idx - 1]) * invDY;
    }
    // psi_Ezx corrige +∂Hy/∂x → signe +
    // psi_Ezy corrige −∂Hx/∂y → signe − (opposé au terme standard)
    Ez[idx] += coeffE * (psi_Ezx[idx] - psi_Ezy[idx]);
}

__global__ void applyPML_H_kernel(
    double* Hx, double* Hy,
    double* psi_Hxx, double* psi_Hyy,
    const double* bh_x, const double* bh_y,
    const double* ch_x, const double* ch_y,
    const double* Ez,
    double invDY, double invDX, double coeffH) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX || j >= NY) return;
    int idx = i * NY + j;

    // psi_Hxx corrige Hx <- -∂Ez/∂y : même signe − que la mise à jour principale
    if (j < NY - 1) {
        psi_Hxx[idx] = bh_y[idx] * psi_Hxx[idx]
                     + ch_y[idx] * (Ez[idx + 1] - Ez[idx]) * invDY;
        Hx[idx] -= coeffH * psi_Hxx[idx];
    }
    // psi_Hyy corrige Hy <- +∂Ez/∂x : même signe + que la mise à jour principale
    if (i < NX - 1) {
        psi_Hyy[idx] = bh_x[idx] * psi_Hyy[idx]
                     + ch_x[idx] * (Ez[idx + NY] - Ez[idx]) * invDX;
        Hy[idx] += coeffH * psi_Hyy[idx];
    }
}
// -------------------------------------------------------------------
// Implémentation de la classe CudaFDTD
// -------------------------------------------------------------------

CudaFDTD::CudaFDTD(const Source& src, int steps, int save)
    : source(src), total_steps(steps), save_interval(save), use_pml(false),
      d_Ez(nullptr), d_Hx(nullptr), d_Hy(nullptr),
      d_psi_Ezx(nullptr), d_psi_Ezy(nullptr),
      d_psi_Hxx(nullptr), d_psi_Hyy(nullptr),
      d_be_x(nullptr), d_be_y(nullptr),
      d_ce_x(nullptr), d_ce_y(nullptr),
      d_bh_x(nullptr), d_bh_y(nullptr),
      d_ch_x(nullptr), d_ch_y(nullptr)
{
    block   = dim3(16, 16);
    gridH   = dim3((NX + block.x - 1) / block.x,
                   (NY + block.y - 1) / block.y);
    gridE   = dim3((NX - 2 + block.x - 1) / block.x,
                   (NY - 2 + block.y - 1) / block.y);
    gridPML = gridH;
}

CudaFDTD::~CudaFDTD() {
    cudaFree(d_Ez);  cudaFree(d_Hx);  cudaFree(d_Hy);
    cudaFree(d_psi_Ezx); cudaFree(d_psi_Ezy);
    cudaFree(d_psi_Hxx); cudaFree(d_psi_Hyy);
    cudaFree(d_be_x); cudaFree(d_be_y);
    cudaFree(d_ce_x); cudaFree(d_ce_y);
    cudaFree(d_bh_x); cudaFree(d_bh_y);
    cudaFree(d_ch_x); cudaFree(d_ch_y);
}

void CudaFDTD::setBoundary(std::unique_ptr<BoundaryCondition> boundary) {
    bc = std::move(boundary);
    use_pml = (dynamic_cast<PML*>(bc.get()) != nullptr);
    if (use_pml) preparePMLCoeffs();
}

void CudaFDTD::preparePMLCoeffs() {
    PML pml_cpu;
    const auto& sigma_x = pml_cpu.getSigmaX();
    const auto& sigma_y = pml_cpu.getSigmaY();

    CUDA_CHECK(cudaMalloc(&d_psi_Ezx, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_psi_Ezy, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_psi_Hxx, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_psi_Hyy, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_be_x, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_be_y, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ce_x, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ce_y, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bh_x, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bh_y, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ch_x, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ch_y, NCELLS * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_psi_Ezx, 0, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_psi_Ezy, 0, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_psi_Hxx, 0, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_psi_Hyy, 0, NCELLS * sizeof(double)));

    std::vector<double> be_x(NCELLS), be_y(NCELLS), ce_x(NCELLS), ce_y(NCELLS);
    std::vector<double> bh_x(NCELLS), bh_y(NCELLS), ch_x(NCELLS), ch_y(NCELLS);

    for (int i = 0; i < NCELLS; ++i) {
        double sx = sigma_x[i], sy = sigma_y[i];
        be_x[i] = std::exp(-sx * DT / EPS0);
        be_y[i] = std::exp(-sy * DT / EPS0);
        ce_x[i] = (sx > 1e-18) ? (be_x[i] - 1.0) : 0.0;
        ce_y[i] = (sy > 1e-18) ? (be_y[i] - 1.0) : 0.0;
        bh_x[i] = std::exp(-sx * DT / MU0);
        bh_y[i] = std::exp(-sy * DT / MU0);
        ch_x[i] = (sx > 1e-18) ? (bh_x[i] - 1.0) : 0.0;
        ch_y[i] = (sy > 1e-18) ? (bh_y[i] - 1.0) : 0.0;
    }

    CUDA_CHECK(cudaMemcpy(d_be_x, be_x.data(), NCELLS*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_be_y, be_y.data(), NCELLS*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ce_x, ce_x.data(), NCELLS*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ce_y, ce_y.data(), NCELLS*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bh_x, bh_x.data(), NCELLS*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bh_y, bh_y.data(), NCELLS*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ch_x, ch_x.data(), NCELLS*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ch_y, ch_y.data(), NCELLS*sizeof(double), cudaMemcpyHostToDevice));
}

void CudaFDTD::run() {
    // ----------------------------------------------------------------
    // Détection du GPU — si aucun disponible, on s'arrête proprement
    // ----------------------------------------------------------------
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "[CUDA] Aucun GPU détecté. Abandon.\n";
        return;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "[CUDA] GPU : " << prop.name
              << "  (sm_" << prop.major << prop.minor << ")\n";

    // ----------------------------------------------------------------
    // Allocation et initialisation à zéro sur GPU
    // ----------------------------------------------------------------
    CUDA_CHECK(cudaMalloc(&d_Ez, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hx, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hy, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Ez, 0, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Hx, 0, NCELLS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Hy, 0, NCELLS * sizeof(double)));

    const int bdim    = 256;
    const int bgrid   = (std::max(NX, NY) + bdim - 1) / bdim;
    const int src_idx = source.x * NY + source.y;

    // ----------------------------------------------------------------
    // Boucle temporelle principale — tout sur GPU
    // ----------------------------------------------------------------
    for (int n = 0; n < total_steps; ++n) {

        // Mise à jour de H (demi-pas) — parallèle sur toute la grille
        updateHx_kernel<<<gridH, block>>>(d_Hx, d_Ez, DT / (MU0 * DY));
        updateHy_kernel<<<gridH, block>>>(d_Hy, d_Ez, DT / (MU0 * DX));

        // Condition aux limites pour H
        if (use_pml) {
            applyPML_H_kernel<<<gridPML, block>>>(
                d_Hx, d_Hy, d_psi_Hxx, d_psi_Hyy,
                d_bh_x, d_bh_y, d_ch_x, d_ch_y,
                d_Ez, 1.0 / DY, 1.0 / DX, DT / MU0);
        } else {
            applyDirichlet_H_kernel<<<bgrid, bdim>>>(d_Hx, d_Hy);
        }

        // Mise à jour de E (pas entier) — parallèle sur l'intérieur
        updateEz_kernel<<<gridE, block>>>(
            d_Ez, d_Hx, d_Hy,
            DT / EPS0, 1.0 / DX, 1.0 / DY);

        // Condition aux limites pour E
        if (use_pml) {
            applyPML_E_kernel<<<gridPML, block>>>(
                d_Ez, d_psi_Ezx, d_psi_Ezy,
                d_be_x, d_be_y, d_ce_x, d_ce_y,
                d_Hx, d_Hy, 1.0 / DY, 1.0 / DX, DT / EPS0);
            // Termination PEC : Ez=0 sur les 4 bords extérieurs de la PML.
            // Sans ce reset les bords accumulent la correction PML → divergence aux coins.
            applyDirichlet_Ez_kernel<<<bgrid, bdim>>>(d_Ez);
        } else {
            applyDirichlet_Ez_kernel<<<bgrid, bdim>>>(d_Ez);
        }

        // Source ponctuelle (1 thread, index direct, sans branchement)
        applySource_kernel<<<1, 1>>>(d_Ez, source.getValue((n + 0.5) * DT), src_idx);

        // Sauvegarde : sync GPU → copie D2H → écriture fichier
        if (n % save_interval == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(host_grid.Ez.data(), d_Ez,
                                  NCELLS*sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_grid.Hx.data(), d_Hx,
                                  NCELLS*sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_grid.Hy.data(), d_Hy,
                                  NCELLS*sizeof(double), cudaMemcpyDeviceToHost));
            writeVTK(host_grid, n, "cuda");
            std::cout << "CUDA step " << n << "/" << total_steps << std::endl;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}
