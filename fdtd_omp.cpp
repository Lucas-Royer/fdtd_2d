#include "fdtd_omp.h"
#include <omp.h>
#include <iostream>

void OpenMPFDTD::run() {
    if (num_threads > 0) omp_set_num_threads(num_threads);
    grid.reset();
    double time = 0.0;

    for (int n = 0; n < total_steps; ++n) {
        // Hx
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY - 1; ++j) {
                grid.hx(i, j) -= (DT / (MU0 * DY)) * (grid.ez(i, j+1) - grid.ez(i, j));
            }
        }
        // Hy
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NX - 1; ++i) {
            for (int j = 0; j < NY; ++j) {
                grid.hy(i, j) += (DT / (MU0 * DX)) * (grid.ez(i+1, j) - grid.ez(i, j));
            }
        }
        if (bc) bc->applyH(grid);   // à paralléliser si nécessaire

        // Ez
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < NX - 1; ++i) {
            for (int j = 1; j < NY - 1; ++j) {
                double curl = (grid.hy(i, j) - grid.hy(i-1, j)) / DX
                            - (grid.hx(i, j) - grid.hx(i, j-1)) / DY;
                grid.ez(i, j) += (DT / EPS0) * curl;
            }
        }

        // Source (un seul thread)
        time = (n + 0.5) * DT;
        grid.ez(source.x, source.y) += source.getValue(time);

        if (bc) bc->applyE(grid);

        if (n % save_interval == 0) {
            writeVTK(grid, n, "omp");
            std::cout << "Step " << n << "/" << total_steps << "\n";
        }
    }
}