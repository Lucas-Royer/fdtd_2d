#include "fdtd_simple.h"
#include <fstream>
#include <iomanip>
#include <filesystem>

// -----------------------------------------------------------------
// Grid
// -----------------------------------------------------------------
void Grid::reset() {
    Ez.fill(0.0);
    Hx.fill(0.0);
    Hy.fill(0.0);
}

// -----------------------------------------------------------------
// Dirichlet
// -----------------------------------------------------------------
void Dirichlet::applyE(Grid& g) {
    for (int j = 0; j < NY; ++j) {
        g.ez(0, j) = 0.0;
        g.ez(NX-1, j) = 0.0;
    }
    for (int i = 0; i < NX; ++i) {
        g.ez(i, 0) = 0.0;
        g.ez(i, NY-1) = 0.0;
    }
}
void Dirichlet::applyH(Grid& g) {
    for (int j = 0; j < NY; ++j) {
        g.hx(0, j) = 0.0;
        g.hx(NX-1, j) = 0.0;
    }
    for (int i = 0; i < NX; ++i) {
        g.hy(i, 0) = 0.0;
        g.hy(i, NY-1) = 0.0;
    }
}

// -----------------------------------------------------------------
// PML (split-field, profil polynomial d'ordre 2)
// -----------------------------------------------------------------
PML::PML() {
    psi_Ezx.fill(0.0); psi_Ezy.fill(0.0);
    psi_Hxx.fill(0.0); psi_Hyy.fill(0.0);
    sigma_x.fill(0.0); sigma_y.fill(0.0);

    // Formule optimale : sigma_max = (m+1) * C0 * EPS0 * ln(1/R_cible) / (2 * NPML * DX)
    // avec m=2 (ordre polynomial), R_cible = 1e-6 (-120 dB théorique)
    // Un sigma trop grand crée un gradient abrupt → réflexion à l'interface PML/domaine
    constexpr double R_target = 1e-6;
    constexpr int    poly_m   = 2;
    sigma_max = (poly_m + 1) * C0 * EPS0 * std::log(1.0 / R_target)
                / (2.0 * NPML * DX);

    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            int idx = i * NY + j;
            // Profil gauche : sigma croît de 0 (interface i=NPML) vers sigma_max (bord i=0)
            if (i < NPML) {
                double r = static_cast<double>(NPML - i) / NPML;
                sigma_x[idx] = sigma_max * r * r;
            } else if (i >= NX - NPML) {
                double r = static_cast<double>(i - (NX - NPML) + 1) / NPML;
                sigma_x[idx] = sigma_max * r * r;
            }
            // Profil bas : sigma croît de 0 (interface j=NPML) vers sigma_max (bord j=0)
            if (j < NPML) {
                double r = static_cast<double>(NPML - j) / NPML;
                sigma_y[idx] = sigma_max * r * r;
            } else if (j >= NY - NPML) {
                double r = static_cast<double>(j - (NY - NPML) + 1) / NPML;
                sigma_y[idx] = sigma_max * r * r;
            }
        }
    }
}

void PML::applyE(Grid& g) {
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            int idx = i * NY + j;
            double be_x = std::exp(-sigma_x[idx] * DT / EPS0);
            double be_y = std::exp(-sigma_y[idx] * DT / EPS0);
            double ce_x = sigma_x[idx] / (sigma_x[idx] + 1e-12) * (be_x - 1.0);
            double ce_y = sigma_y[idx] / (sigma_y[idx] + 1e-12) * (be_y - 1.0);

            // psi_Ezx corrige ∂Hy/∂x → différence selon i, coefficient sigma_x
            if (i > 0)
                psi_Ezx[idx] = be_x * psi_Ezx[idx] + ce_x * (g.hy(i,j) - g.hy(i-1,j)) / DX;
            // psi_Ezy corrige ∂Hx/∂y → différence selon j, coefficient sigma_y
            if (j > 0)
                psi_Ezy[idx] = be_y * psi_Ezy[idx] + ce_y * (g.hx(i,j) - g.hx(i,j-1)) / DY;

            g.ez(i,j) += (DT / EPS0) * (psi_Ezx[idx] - psi_Ezy[idx]);
        }
    }
    // Termination PEC obligatoire : le mur extérieur de la PML doit être Ez=0.
    // Le curl (updateEz) ne touche jamais i=0, i=NX-1, j=0, j=NY-1.
    // Sans ce reset, les bords accumulent la correction PML indéfiniment → divergence.
    for (int k = 0; k < NX; ++k) { g.ez(k, 0) = 0.0; g.ez(k, NY-1) = 0.0; }
    for (int k = 0; k < NY; ++k) { g.ez(0, k) = 0.0; g.ez(NX-1, k) = 0.0; }
}

void PML::applyH(Grid& g) {
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            int idx = i * NY + j;
            double bh_x = std::exp(-sigma_x[idx] * DT / MU0);
            double bh_y = std::exp(-sigma_y[idx] * DT / MU0);
            double ch_x = sigma_x[idx] / (sigma_x[idx] + 1e-12) * (bh_x - 1.0);
            double ch_y = sigma_y[idx] / (sigma_y[idx] + 1e-12) * (bh_y - 1.0);

            // psi_Hxx corrige Hx ← ∂Ez/∂y → différence selon j, coefficient sigma_y
            if (j < NY-1)
                psi_Hxx[idx] = bh_y * psi_Hxx[idx] + ch_y * (g.ez(i,j+1) - g.ez(i,j)) / DY;
            // psi_Hyy corrige Hy ← ∂Ez/∂x → différence selon i, coefficient sigma_x
            if (i < NX-1)
                psi_Hyy[idx] = bh_x * psi_Hyy[idx] + ch_x * (g.ez(i+1,j) - g.ez(i,j)) / DX;

            g.hx(i,j) -= (DT / MU0) * psi_Hxx[idx];
            g.hy(i,j) += (DT / MU0) * psi_Hyy[idx];
        }
    }
}

// -----------------------------------------------------------------
// Écriture VTK (ImageData XML)
// -----------------------------------------------------------------
void writeVTK(const Grid& g, int step, const std::string& prefix) {
    std::filesystem::create_directories("output");
    std::ostringstream fname;
    fname << "output/" << prefix << "_" << std::setw(6) << std::setfill('0') << step << ".vti";

    std::ofstream f(fname.str());
    if (!f) return;

    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"ImageData\" version=\"0.1\">\n";
    f << "  <ImageData WholeExtent=\"0 " << NX-1 << " 0 " << NY-1 << " 0 0\" "
      << "Origin=\"0 0 0\" Spacing=\"" << DX << " " << DY << " 1\">\n";
    f << "    <Piece Extent=\"0 " << NX-1 << " 0 " << NY-1 << " 0 0\">\n";
    f << "      <PointData Scalars=\"Ez\">\n";
    f << "        <DataArray type=\"Float64\" Name=\"Ez\" format=\"ascii\">\n";
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j)
            f << g.ez(i,j) << " ";
        f << "\n";
    }
    f << "        </DataArray>\n";
    f << "      </PointData>\n";
    f << "    </Piece>\n";
    f << "  </ImageData>\n";
    f << "</VTKFile>\n";
    f.close();
}