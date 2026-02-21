#ifndef FDTD_SIMPLE_H
#define FDTD_SIMPLE_H

#include <vector>
#include <cmath>
#include <memory>
#include <string>
#include <algorithm>

// Dimensions de la grille — variables globales modifiables au runtime.
// Utiliser setGridSize() avant toute construction d'objet.
// Note : les kernels CUDA utilisent NX/NY via des constantes de device
// recompilées, mais la version CPU lit ces globales directement.
inline int NX     = 200;
inline int NY     = 200;
inline int NCELLS = NX * NY;
inline int NPML_G = 40;

// Constantes physiques
constexpr double C0   = 299792458.0;
constexpr double EPS0 = 8.854187817e-12;
constexpr double MU0  = 4.0 * M_PI * 1e-7;

// Discrétisation (CFL = 0.99/√2)
constexpr double DX = 1.0e-3;
constexpr double DY = 1.0e-3;
constexpr double DT = 0.99 * DX / (C0 * 1.414);

// Modifier la taille de grille (appeler avant tout constructeur)
inline void setGridSize(int nx, int ny) {
    NX     = nx;
    NY     = ny;
    NCELLS = nx * ny;
    // PML : 10% de la grille, entre 10 et 40 couches
    NPML_G = std::max(10, std::min(40, static_cast<int>(0.10 * std::min(nx, ny))));
}

// -----------------------------------------------------------------
// Grid : grille de Yee TE, stockage sur le TAS (std::vector)
// pas de stack overflow quelle que soit la taille 
// la première implémentation utilisait std::array
// -----------------------------------------------------------------
class Grid {
public:
    std::vector<double> Ez;
    std::vector<double> Hx;
    std::vector<double> Hy;

    Grid() : Ez(NCELLS, 0.0), Hx(NCELLS, 0.0), Hy(NCELLS, 0.0) {}

    double& ez(int i, int j)             { return Ez[i * NY + j]; }
    double& hx(int i, int j)             { return Hx[i * NY + j]; }
    double& hy(int i, int j)             { return Hy[i * NY + j]; }
    const double& ez(int i, int j) const { return Ez[i * NY + j]; }
    const double& hx(int i, int j) const { return Hx[i * NY + j]; }
    const double& hy(int i, int j) const { return Hy[i * NY + j]; }

    void reset() {
        std::fill(Ez.begin(), Ez.end(), 0.0);
        std::fill(Hx.begin(), Hx.end(), 0.0);
        std::fill(Hy.begin(), Hy.end(), 0.0);
    }

    int getNx() const { return NX; }
    int getNy() const { return NY; }
};

// -----------------------------------------------------------------
// Conditions aux limites
// -----------------------------------------------------------------
class BoundaryCondition {
public:
    virtual ~BoundaryCondition() = default;
    virtual void applyE(Grid& g) = 0;
    virtual void applyH(Grid& g) = 0;
};

class Dirichlet : public BoundaryCondition {
public:
    void applyE(Grid& g) override;
    void applyH(Grid& g) override;
};

class PML : public BoundaryCondition {
private:
    int npml;
    std::vector<double> psi_Ezx, psi_Ezy;
    std::vector<double> psi_Hxx, psi_Hyy;
    std::vector<double> sigma_x, sigma_y;
    double sigma_max;

public:
    PML();
    void applyE(Grid& g) override;
    void applyH(Grid& g) override;
    const std::vector<double>& getSigmaX() const { return sigma_x; }
    const std::vector<double>& getSigmaY() const { return sigma_y; }
    int getNPML() const { return npml; }
};

// -----------------------------------------------------------------
// Source ponctuelle : sinus continu ou impulsion gaussienne
// -----------------------------------------------------------------
class Source {
public:
    int x, y;
    double freq, amp;
    bool   pulse;
    double t0, tau;

    Source(int xi, int yi, double f = 1e9, double a = 1.0)
        : x(xi), y(yi), freq(f), amp(a), pulse(false), t0(0.0), tau(0.0) {}

    static Source makeGaussian(int xi, int yi, double a, double t0_s, double tau_s) {
        Source s(xi, yi, 0.0, a);
        s.pulse = true; s.t0 = t0_s; s.tau = tau_s;
        return s;
    }

    double getValue(double t) const {
        if (pulse) {
            double d = t - t0;
            return amp * std::exp(-d*d / (2.0*tau*tau));
        }
        return amp * std::sin(2.0 * M_PI * freq * t);
    }
};

void writeVTK(const Grid& g, int step, const std::string& prefix);

#endif // FDTD_SIMPLE_H
