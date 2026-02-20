#ifndef FDTD_SIMPLE_H
#define FDTD_SIMPLE_H

#include <array>
#include <cmath>
#include <memory>
#include <string>

// Dimensions de la grille (fixées à la compilation)
constexpr int NX = 200;
constexpr int NY = 200;
constexpr int NCELLS = NX * NY;

// Constantes physiques (air)
constexpr double C0 = 299792458.0;
constexpr double EPS0 = 8.854187817e-12;
constexpr double MU0 = 4.0 * M_PI * 1e-7;

// Paramètres de discrétisation (CFL = 0.99)
constexpr double DX = 1.0e-3;      // 1 mm
constexpr double DY = 1.0e-3;
constexpr double DT = 0.99 * DX / (C0 * 1.414);

// -----------------------------------------------------------------
// Classe Grid : grille de Yee pour la polarisation TE
// Stockage : std::array<double, NCELLS> (allocation statique)
// -----------------------------------------------------------------
class Grid {
public:
    std::array<double, NCELLS> Ez;
    std::array<double, NCELLS> Hx;
    std::array<double, NCELLS> Hy;

    // Accès indexé (i, j) avec vérification de bornes en debug
    double& ez(int i, int j)       { return Ez[i * NY + j]; }
    double& hx(int i, int j)       { return Hx[i * NY + j]; }
    double& hy(int i, int j)       { return Hy[i * NY + j]; }

    const double& ez(int i, int j) const { return Ez[i * NY + j]; }
    const double& hx(int i, int j) const { return Hx[i * NY + j]; }
    const double& hy(int i, int j) const { return Hy[i * NY + j]; }

    // Méthodes utilitaires
    void reset();
    int getNx() const { return NX; }
    int getNy() const { return NY; }
};

// -----------------------------------------------------------------
// Condition aux limites (interface)
// -----------------------------------------------------------------
class BoundaryCondition {
public:
    virtual ~BoundaryCondition() = default;
    virtual void applyE(Grid& g) = 0;
    virtual void applyH(Grid& g) = 0;
};

// Dirichlet (PEC) : Ez = 0 sur les bords
class Dirichlet : public BoundaryCondition {
public:
    void applyE(Grid& g) override;
    void applyH(Grid& g) override;
};

// PML simplifiée (split-field, 10 couches)
class PML : public BoundaryCondition {
private:
    static constexpr int NPML = 30;
    std::array<double, NCELLS> psi_Ezx, psi_Ezy;
    std::array<double, NCELLS> psi_Hxx, psi_Hyy;
    std::array<double, NCELLS> sigma_x, sigma_y;
    double sigma_max;

public:
    PML();
    void applyE(Grid& g) override;
    void applyH(Grid& g) override;
    const std::array<double, NCELLS>& getSigmaX() const { return sigma_x; }
    const std::array<double, NCELLS>& getSigmaY() const { return sigma_y; }
};

// -----------------------------------------------------------------
// Source ponctuelle — sinusoïdale ou impulsion gaussienne
// -----------------------------------------------------------------
class Source {
public:
    int x, y;
    double freq;
    double amp;
    bool   pulse;    // true = impulsion gaussienne, false = sinus continu
    double t0;       // centre temporel de l'impulsion (s)
    double tau;      // largeur de l'impulsion (s)

    // Constructeur sinus continu (compatibilité ascendante)
    Source(int xi, int yi, double f = 1e9, double a = 1.0)
        : x(xi), y(yi), freq(f), amp(a), pulse(false),
          t0(0.0), tau(0.0) {}

    // Constructeur impulsion gaussienne (pour test PML)
    static Source makeGaussian(int xi, int yi, double a,
                                double t0_s, double tau_s) {
        Source s(xi, yi, 0.0, a);
        s.pulse = true;
        s.t0    = t0_s;
        s.tau   = tau_s;
        return s;
    }

    double getValue(double t) const {
        if (pulse) {
            double dt = t - t0;
            return amp * std::exp(-dt * dt / (2.0 * tau * tau));
        }
        return amp * std::sin(2.0 * M_PI * freq * t);
    }
};

// -----------------------------------------------------------------
// Écriture VTK simplifiée (format .vti)
// -----------------------------------------------------------------
void writeVTK(const Grid& g, int step, const std::string& prefix);

#endif // FDTD_SIMPLE_H