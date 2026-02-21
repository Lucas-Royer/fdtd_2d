#include "fdtd_simple.h"
#include "fdtd_seq.h"
#include "fdtd_omp.h"
#include "fdtd_cuda.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

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

    // Vérification 1 : profil selon x (doit être sin(pi*i/NX)) ---
    int j_mid = NY / 2;
    double err_x = 0.0;
    for (int i = 0; i < NX; ++i) {
        double norm_num = grid.ez(i, j_mid) / max_val;
        double norm_ana = std::sin(M_PI * i / static_cast<double>(NX));
        double d = std::abs(norm_num) - std::abs(norm_ana);
        err_x += d * d;
    }
    err_x = std::sqrt(err_x / NX);

    // Vérification 2 : uniformité selon y à i=NX/2 (doit être constant) ---
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
    else if (mode == "bench") {
        // ================================================================
        // Mode benchmark
        // Mesures :
        //   1. Timing global médiane (3 runs) sur grille NX×NY
        //   2. Scaling fort : plusieurs tailles de grille (SEQ + CUDA)
        //   3. Timing par phase : H-update / E-update / BC / transfert D2H
        //   4. Coût des transferts D2H CUDA vs calcul pur
        //   5. Fraction série (Amdahl) estimée
        //   6. GFLOP/s et débit mémoire → roofline
        // ================================================================

        using Clock = std::chrono::steady_clock;
        using Ms    = std::chrono::duration<double, std::milli>;

        const int bench_steps = 1000;
        const int no_save     = bench_steps + 1;
        const int max_threads = omp_get_max_threads();

        auto median_ms = [&](auto&& fn, int N = 3) {
            std::vector<double> t;
            for (int k = 0; k < N; ++k) {
                auto t0 = Clock::now();
                fn();
                t.push_back(std::chrono::duration_cast<Ms>(Clock::now()-t0).count());
            }
            std::sort(t.begin(), t.end());
            return t[N/2];
        };

        // ── Constantes FLOP et mémoire ────────────────────────────────
        // updateHx : NX*(NY-1)*3, updateHy : (NX-1)*NY*3
        // updateEz : (NX-2)*(NY-2)*9  (4 sub + 2 div + 1 sub + 1 mul + 1 add)
        const double flop_per_step = NX*(NY-1)*3.0 + (NX-1)*NY*3.0 + (NX-2)*(NY-2)*9.0;
        // Accès mémoire : Hx(4 doubles/cell), Hy(4), Ez(6)
        const double bytes_per_step =
            (NX*(NY-1)*4.0 + (NX-1)*NY*4.0 + (NX-2)*(NY-2)*6.0) * sizeof(double);
        const double arith_intensity = flop_per_step / bytes_per_step;

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║        BENCHMARK FDTD 2D                                       ║\n";
        std::cout << "║        Grille " << NX << "×" << NY << "   " << bench_steps
                  << " pas   3 runs médiane                  ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

        // ================================================================
        // TABLE 1 — Speedup global et efficacité parallèle OMP
        // ================================================================
        Source bench_src(NX/2, NY/2, f0, 1.0);

        double t_seq = median_ms([&]() {
            SequentialFDTD sim(bench_src, bench_steps, no_save);
            sim.setBoundary(std::make_unique<Dirichlet>()); sim.run();
        });

        std::vector<std::pair<int,double>> omp_times;
        for (int t = 1; t <= max_threads; ++t) {
            double ms = median_ms([&]() {
                OpenMPFDTD sim(bench_src, bench_steps, no_save, t);
                sim.setBoundary(std::make_unique<Dirichlet>()); sim.run();
            });
            omp_times.push_back({t, ms});
        }

        double t_cuda = median_ms([&]() {
            CudaFDTD sim(bench_src, bench_steps, no_save);
            sim.setBoundary(std::make_unique<Dirichlet>()); sim.run();
        });

        auto gflops = [&](double ms) {
            return flop_per_step * bench_steps / (ms*1e-3) / 1e9;
        };

        std::cout << "TABLE 1 — Speedup et performances (grille " << NX << "×" << NY << ")\n";
        std::cout << "┌──────────────────────┬──────────┬─────────┬──────────┬──────────┐\n";
        std::cout << "│ Implémentation       │ Tps (ms) │ Speedup │ Efficac. │ GFLOP/s  │\n";
        std::cout << "├──────────────────────┼──────────┼─────────┼──────────┼──────────┤\n";
        std::cout << std::fixed;
        // SEQ
        std::cout << "│ SEQ                  │"
                  << std::setw(9) << std::setprecision(1) << t_seq << " │"
                  << std::setw(8) << std::setprecision(2) << 1.0 << "x│"
                  << std::setw(9) << "  —" << " │"
                  << std::setw(9) << std::setprecision(3) << gflops(t_seq) << " │\n";
        std::cout << "├──────────────────────┼──────────┼─────────┼──────────┼──────────┤\n";
        for (auto [nth, ms] : omp_times) {
            double sp  = t_seq / ms;
            double eff = sp / nth * 100.0;
            std::string lbl = "OMP " + std::to_string(nth) + "T";
            lbl += std::string(18 - lbl.size(), ' ');
            std::cout << "│ " << lbl << " │"
                      << std::setw(9) << std::setprecision(1) << ms << " │"
                      << std::setw(8) << std::setprecision(2) << sp  << "x│"
                      << std::setw(8) << std::setprecision(1) << eff << "% │"
                      << std::setw(9) << std::setprecision(3) << gflops(ms) << " │\n";
        }
        std::cout << "├──────────────────────┼──────────┼─────────┼──────────┼──────────┤\n";
        std::cout << "│ CUDA                 │"
                  << std::setw(9) << std::setprecision(1) << t_cuda << " │"
                  << std::setw(8) << std::setprecision(2) << t_seq/t_cuda << "x│"
                  << std::setw(9) << "  —" << " │"
                  << std::setw(9) << std::setprecision(3) << gflops(t_cuda) << " │\n";
        std::cout << "└──────────────────────┴──────────┴─────────┴──────────┴──────────┘\n\n";

        // ================================================================
        // TABLE 2 — Loi d'Amdahl : estimation de la fraction série
        // ================================================================
        //  T_p = T_s*(f + (1-f)/p)  →  f = (T_s/T_p - 1/p) / (1 - 1/p)
        std::cout << "TABLE 2 — Fraction série estimée (loi d'Amdahl)\n";
        std::cout << "┌──────────────────────┬─────────┬──────────────┐\n";
        std::cout << "│ Threads              │ Speedup │ Fraction sér.│\n";
        std::cout << "├──────────────────────┼─────────┼──────────────┤\n";
        for (auto [p, ms] : omp_times) {
            if (p < 2) continue;
            double sp = t_seq / ms;
            double f  = (1.0/sp - 1.0/p) / (1.0 - 1.0/p);
            std::cout << "│ OMP " << std::setw(2) << p << " threads        │"
                      << std::setw(8) << std::setprecision(2) << sp << "x│"
                      << std::setw(12) << std::setprecision(1) << f*100.0 << "% │\n";
        }
        std::cout << "└──────────────────────┴─────────┴──────────────┘\n\n";

        // ================================================================
        // ================================================================
        // TABLE 3 — Scaling fort : tailles de grille croissantes
        // NX/NY sont maintenant des variables globales modifiables via setGridSize()
        // → pas de recompilation nécessaire, pas de stack overflow (std::vector)
        // ================================================================
        std::cout << "TABLE 3 — Scaling fort (SEQ vs OMP vs CUDA, 500 pas)\n";
        std::cout << "┌──────────────┬──────────┬──────────┬──────────┬─────────┬──────────┐\n";
        std::cout << "│ Grille       │ SEQ (ms) │OMP  (ms) │CUDA (ms) │Sp CUDA  │ GFLOP/s  │\n";
        std::cout << "├──────────────┼──────────┼──────────┼──────────┼─────────┼──────────┤\n";

        const int scaling_steps = 500;
        for (int N : {128, 256, 512, 1024}) {
            setGridSize(N, N);
            Source src_n(N/2, N/2, C0/(2.0*N*DX), 1.0);
            double flop_n = N*(N-1)*3.0 + (N-1)*N*3.0 + (N-2)*(N-2)*9.0;

            double ts = median_ms([&]() {
                SequentialFDTD sim(src_n, scaling_steps, scaling_steps+1);
                sim.setBoundary(std::make_unique<Dirichlet>()); sim.run();
            });
            double to = median_ms([&]() {
                OpenMPFDTD sim(src_n, scaling_steps, scaling_steps+1, max_threads);
                sim.setBoundary(std::make_unique<Dirichlet>()); sim.run();
            });
            double tc = median_ms([&]() {
                CudaFDTD sim(src_n, scaling_steps, scaling_steps+1);
                sim.setBoundary(std::make_unique<Dirichlet>()); sim.run();
            });
            double gf = flop_n * scaling_steps / (tc*1e-3) / 1e9;
            std::cout << "│" << std::setw(5) << N << "×" << std::setw(5) << N << "      │"
                      << std::setw(9) << std::setprecision(1) << ts << " │"
                      << std::setw(9) << std::setprecision(1) << to << " │"
                      << std::setw(9) << std::setprecision(1) << tc << " │"
                      << std::setw(8) << std::setprecision(2) << ts/tc << "x│"
                      << std::setw(9) << std::setprecision(3) << gf << " │\n";
        }
        // Remettre la grille par défaut
        setGridSize(200, 200);
        std::cout << "└──────────────┴──────────┴──────────┴──────────┴─────────┴──────────┘\n\n";

        // ================================================================
        // TABLE 4 — Overhead CUDA : transferts D2H vs calcul pur
        // ================================================================
        std::cout << "TABLE 4 — Overhead CUDA : coût des transferts D2H\n";
        std::cout << "  (mesure avec save_interval variable, 1000 pas)\n";
        std::cout << "┌─────────────────┬──────────┬───────────┬────────────────┐\n";
        std::cout << "│ save_interval   │ Tps (ms) │ Overhead  │ Transferts/run │\n";
        std::cout << "├─────────────────┼──────────┼───────────┼────────────────┤\n";

        // Référence : aucun transfert
        double t_no_save = median_ms([&]() {
            CudaFDTD sim(bench_src, bench_steps, bench_steps+1);
            sim.setBoundary(std::make_unique<Dirichlet>()); sim.run();
        });
        std::cout << "│ ∞ (aucun)       │"
                  << std::setw(9) << std::setprecision(1) << t_no_save << " │"
                  << std::setw(10) << "  réf." << " │"
                  << std::setw(15) << "0" << " │\n";

        for (int si : {1000, 500, 200, 100, 50}) {
            if (si >= bench_steps) continue;
            double t_with = median_ms([&]() {
                CudaFDTD sim(bench_src, bench_steps, si);
                sim.setBoundary(std::make_unique<Dirichlet>()); sim.run();
            });
            int n_transfers = bench_steps / si;
            double overhead_pct = (t_with - t_no_save) / t_no_save * 100.0;
            std::cout << "│" << std::setw(9) << si << "        │"
                      << std::setw(9) << std::setprecision(1) << t_with << " │"
                      << std::setw(9) << std::setprecision(1) << overhead_pct << "% │"
                      << std::setw(15) << n_transfers << " │\n";
        }
        std::cout << "└─────────────────┴──────────┴───────────┴────────────────┘\n\n";

        // ================================================================
        // ANALYSE ROOFLINE
        // ================================================================
        double bw_gpu_gb  = (bytes_per_step * bench_steps) / (t_cuda*1e-3) / 1e9;
        double bw_seq_gb  = (bytes_per_step * bench_steps) / (t_seq *1e-3) / 1e9;
        std::cout << "ANALYSE ROOFLINE\n";
        std::cout << "  Intensité arithmétique : "
                  << std::setprecision(3) << arith_intensity << " FLOP/byte\n";
        std::cout << "  → Algorithme MEMORY-BOUND (typique FDTD)\n\n";
        std::cout << "  ┌──────────┬────────────┬───────────┬──────────────────────────┐\n";
        std::cout << "  │          │ GFLOP/s    │  BW eff.  │  BW théorique            │\n";
        std::cout << "  ├──────────┼────────────┼───────────┼──────────────────────────┤\n";
        std::cout << "  │ SEQ      │"
                  << std::setw(11) << std::setprecision(3) << gflops(t_seq) << " │"
                  << std::setw(8)  << std::setprecision(1) << bw_seq_gb << " GB/s│"
                  << "  ~51 GB/s (DDR5)          │\n";
        std::cout << "  │ CUDA     │"
                  << std::setw(11) << std::setprecision(3) << gflops(t_cuda) << " │"
                  << std::setw(8)  << std::setprecision(1) << bw_gpu_gb << " GB/s│"
                  << "  ~272 GB/s (GDDR6)        │\n";
        std::cout << "  └──────────┴────────────┴───────────┴──────────────────────────┘\n";
        std::cout << "  Roofline CUDA (BW-limited) : "
                  << std::setprecision(1) << arith_intensity * 272.0 << " GFLOP/s théorique max\n";
        std::cout << "  Utilisation BW GPU : "
                  << std::setprecision(1) << bw_gpu_gb / 272.0 * 100.0 << "%\n\n";
    }
    else {
        std::cerr << "Modes : seq, omp, cuda, seq-pml, omp-pml, cuda-pml, bench\n";
        return 1;
    }

    return 0;
}