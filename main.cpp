#include "fdtd_simple.h"
#include "fdtd_seq.h"
#include "fdtd_omp.h"
#include "fdtd_cuda.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

// Fonction de validation : compare la solution numérique à la solution analytique TE10
// Ez(x,y,t) = A * sin(pi*x/Lx)   — uniforme selon y, sinusoïdal selon x
//
// Deux vérifications :
//   1. Profil selon x à j=NY/2 : doit correspondre à sin(pi*i/NX)
//   2. Profil selon y à i=NX/2 : doit être uniforme (écart-type ~ 0)
void validate(const Grid& grid, int step) {
    double max_val = *std::max_element(grid.Ez.begin(), grid.Ez.end(),
                        [](double a, double b){ return std::abs(a) < std::abs(b); });
    max_val = std::abs(max_val);

    if (max_val < 1e-12) {
        std::cout << "  [validate] Step " << step << " : champ nul, source pas encore établie\n";
        return;
    }

    // --- Vérification 1 : profil selon x (doit être sin(pi*i/NX)) ---
    int j_mid = NY / 2;
    double err_x = 0.0;
    for (int i = 0; i < NX; ++i) {
        double norm_num = grid.ez(i, j_mid) / max_val;
        double norm_ana = std::sin(M_PI * i / static_cast<double>(NX));
        double d = std::abs(norm_num) - std::abs(norm_ana);
        err_x += d * d;
    }
    err_x = std::sqrt(err_x / NX);

    // --- Vérification 2 : uniformité selon y à i=NX/2 (doit être constant) ---
    int i_mid = NX / 2;
    double sum_y = 0.0, sum_y2 = 0.0;
    for (int j = 0; j < NY; ++j) {
        double v = grid.ez(i_mid, j) / max_val;
        sum_y  += v;
        sum_y2 += v * v;
    }
    double mean_y = sum_y / NY;
    double std_y  = std::sqrt(std::max(0.0, sum_y2 / NY - mean_y * mean_y));

    std::cout << "  [validate] Step " << std::setw(5) << step
              << "  |  err_profil_x = " << std::scientific << std::setprecision(3) << err_x
              << "  |  std_y = " << std_y
              << "\n";
}

int main(int argc, char** argv) {
    int total_steps   = 5000;
    int save_interval = 10;

    double Lx = NX * DX;
    double f0 = C0 / (2.0 * Lx);
    std::cout << "Fréquence TE10 : " << f0 * 1e-9 << " GHz\n";

    std::string mode = "seq";
    if (argc > 1) mode = argv[1];

    bool use_pml = (mode.find("pml") != std::string::npos);

    // Source sinusoïdale continue pour les modes Dirichlet (résonance)
    // Source impulsionnelle gaussienne pour les modes PML (onde sortante)
    Source source = use_pml
        ? Source::makeGaussian(NX/2, NY/2, 1.0,
                               30.0 * DT,   // centre : 30 pas
                               10.0 * DT)   // largeur : 10 pas
        : Source(NX/2, NY/2, f0, 1.0);

    auto make_boundary = [&]() -> std::unique_ptr<BoundaryCondition> {
        if (use_pml) {
            std::cout << "BC : PML (10 couches)\n";
            return std::make_unique<PML>();
        }
        std::cout << "BC : Dirichlet (PEC)\n";
        return std::make_unique<Dirichlet>();
    };

    if (mode == "seq" || mode == "seq-pml") {
        SequentialFDTD sim(source, total_steps, save_interval);
        sim.setBoundary(make_boundary());
        sim.run();
        std::cout << "\n--- Validation finale ---\n";
        validate(sim.getGrid(), total_steps - 1);
    }
    else if (mode == "omp" || mode == "omp-pml") {
        OpenMPFDTD sim(source, total_steps, save_interval, 4);
        sim.setBoundary(make_boundary());
        sim.run();
        std::cout << "\n--- Validation finale ---\n";
        validate(sim.getGrid(), total_steps - 1);
    }
    else if (mode == "cuda" || mode == "cuda-pml") {
        CudaFDTD sim(source, total_steps, save_interval);
        sim.setBoundary(make_boundary());
        sim.run();
        std::cout << "\n--- Validation finale ---\n";
        validate(sim.getGrid(), total_steps - 1);
    }
    else {
        std::cerr << "Modes : seq, omp, cuda, seq-pml, omp-pml, cuda-pml\n";
        return 1;
    }

    return 0;
}