#include "fdtd_seq.h"
#include <iostream>

void SequentialFDTD::run() {
    grid.reset();
    double time = 0.0;

    for (int n = 0; n < total_steps; ++n) {
        // --- Mise à jour de H (demi-pas) ---
        // Hx[i,j] <- Hx[i,j] - (DT/(MU0*DY)) * (Ez[i,j+1] - Ez[i,j])
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY - 1; ++j) {
                grid.hx(i, j) -= (DT / (MU0 * DY)) * (grid.ez(i, j+1) - grid.ez(i, j));
            }
        }
        // Hy[i,j] <- Hy[i,j] + (DT/(MU0*DX)) * (Ez[i+1,j] - Ez[i,j])
        for (int i = 0; i < NX - 1; ++i) {
            for (int j = 0; j < NY; ++j) {
                grid.hy(i, j) += (DT / (MU0 * DX)) * (grid.ez(i+1, j) - grid.ez(i, j));
            }
        }
        if (bc) bc->applyH(grid);

        // --- Mise à jour de E (pas entier) ---
        // Ez[i,j] <- Ez[i,j] + (DT/EPS0) * [ (Hy[i,j]-Hy[i-1,j])/DX - (Hx[i,j]-Hx[i,j-1])/DY ]
        for (int i = 1; i < NX - 1; ++i) {
            for (int j = 1; j < NY - 1; ++j) {
                double curl = (grid.hy(i, j) - grid.hy(i-1, j)) / DX
                            - (grid.hx(i, j) - grid.hx(i, j-1)) / DY;
                grid.ez(i, j) += (DT / EPS0) * curl;
            }
        }

        // --- Source ponctuelle (hard source) ---
        time = (n + 0.5) * DT;   // on applique au milieu du pas
        grid.ez(source.x, source.y) += source.getValue(time);

        if (bc) bc->applyE(grid);

        // --- Sauvegarde ---
        if (n % save_interval == 0) {
            writeVTK(grid, n, "seq");
            std::cout << "Step " << n << "/" << total_steps << "\n";
        }
    }
}