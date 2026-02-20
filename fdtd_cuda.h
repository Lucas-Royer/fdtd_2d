#ifndef FDTD_CUDA_H
#define FDTD_CUDA_H

#include "fdtd_simple.h"
#include <cuda_runtime.h>

class CudaFDTD {
private:
    Grid host_grid;
    double *d_Ez, *d_Hx, *d_Hy;
    // Champs PML
    double *d_psi_Ezx, *d_psi_Ezy, *d_psi_Hxx, *d_psi_Hyy;
    double *d_be_x, *d_be_y, *d_ce_x, *d_ce_y;
    double *d_bh_x, *d_bh_y, *d_ch_x, *d_ch_y;

    std::unique_ptr<BoundaryCondition> bc;
    Source source;
    int total_steps, save_interval;
    dim3 block, gridH, gridE, gridPML;
    bool use_pml;

public:
    CudaFDTD(const Source& src, int steps, int save);
    ~CudaFDTD();

    void setBoundary(std::unique_ptr<BoundaryCondition> boundary);
    void run();
    const Grid& getGrid() const { return host_grid; }

private:
    void preparePMLCoeffs();
};

#endif