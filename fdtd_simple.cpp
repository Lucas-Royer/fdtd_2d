#include "fdtd_simple.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>

// -----------------------------------------------------------------
// Dirichlet
// -----------------------------------------------------------------
void Dirichlet::applyE(Grid& g) {
    for (int j = 0; j < NY; ++j) { g.ez(0,j)=0.0;   g.ez(NX-1,j)=0.0; }
    for (int i = 0; i < NX; ++i) { g.ez(i,0)=0.0;   g.ez(i,NY-1)=0.0; }
}
void Dirichlet::applyH(Grid& g) {
    for (int j = 0; j < NY; ++j) { g.hx(0,j)=0.0;   g.hx(NX-1,j)=0.0; }
    for (int i = 0; i < NX; ++i) { g.hy(i,0)=0.0;   g.hy(i,NY-1)=0.0; }
}

// -----------------------------------------------------------------
// PML (split-field, profil polynomial ordre 2)
// -----------------------------------------------------------------
PML::PML()
    : npml(NPML_G),
      psi_Ezx(NCELLS,0.0), psi_Ezy(NCELLS,0.0),
      psi_Hxx(NCELLS,0.0), psi_Hyy(NCELLS,0.0),
      sigma_x(NCELLS,0.0), sigma_y(NCELLS,0.0)
{
    // sigma_max optimal : (m+1)*C0*EPS0*ln(1/R) / (2*npml*DX), m=2, R=1e-6
    constexpr double R_target = 1e-6;
    constexpr int    poly_m   = 2;
    sigma_max = (poly_m + 1) * C0 * EPS0 * std::log(1.0 / R_target)
                / (2.0 * npml * DX);

    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            int idx = i * NY + j;
            if (i < npml) {
                double r = static_cast<double>(npml - i) / npml;
                sigma_x[idx] = sigma_max * r * r;
            } else if (i >= NX - npml) {
                double r = static_cast<double>(i - (NX - npml) + 1) / npml;
                sigma_x[idx] = sigma_max * r * r;
            }
            if (j < npml) {
                double r = static_cast<double>(npml - j) / npml;
                sigma_y[idx] = sigma_max * r * r;
            } else if (j >= NY - npml) {
                double r = static_cast<double>(j - (NY - npml) + 1) / npml;
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
            if (i > 0)
                psi_Ezx[idx] = be_x * psi_Ezx[idx] + ce_x * (g.hy(i,j) - g.hy(i-1,j)) / DX;
            if (j > 0)
                psi_Ezy[idx] = be_y * psi_Ezy[idx] + ce_y * (g.hx(i,j) - g.hx(i,j-1)) / DY;
            g.ez(i,j) += (DT / EPS0) * (psi_Ezx[idx] - psi_Ezy[idx]);
        }
    }
    // Terminaison PEC : mur extérieur de la PML toujours à Ez=0
    for (int k = 0; k < NX; ++k) { g.ez(k,0)=0.0; g.ez(k,NY-1)=0.0; }
    for (int k = 0; k < NY; ++k) { g.ez(0,k)=0.0; g.ez(NX-1,k)=0.0; }
}

void PML::applyH(Grid& g) {
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            int idx = i * NY + j;
            double bh_x = std::exp(-sigma_x[idx] * DT / MU0);
            double bh_y = std::exp(-sigma_y[idx] * DT / MU0);
            double ch_x = sigma_x[idx] / (sigma_x[idx] + 1e-12) * (bh_x - 1.0);
            double ch_y = sigma_y[idx] / (sigma_y[idx] + 1e-12) * (bh_y - 1.0);
            if (j < NY-1)
                psi_Hxx[idx] = bh_y * psi_Hxx[idx] + ch_y * (g.ez(i,j+1) - g.ez(i,j)) / DY;
            if (i < NX-1)
                psi_Hyy[idx] = bh_x * psi_Hyy[idx] + ch_x * (g.ez(i+1,j) - g.ez(i,j)) / DX;
            g.hx(i,j) -= (DT / MU0) * psi_Hxx[idx];
            g.hy(i,j) += (DT / MU0) * psi_Hyy[idx];
        }
    }
}

// -----------------------------------------------------------------
// VTK
// -----------------------------------------------------------------
void writeVTK(const Grid& g, int step, const std::string& prefix) {
    std::filesystem::create_directories("output");
    std::ostringstream fname;
    fname << "output/" << prefix << "_"
          << std::setw(6) << std::setfill('0') << step << ".vti";

    std::ofstream f(fname.str());
    if (!f) return;

    f << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
      << "  <ImageData WholeExtent=\"0 " << NX-1 << " 0 " << NY-1 << " 0 0\" "
      << "Origin=\"0 0 0\" Spacing=\"" << DX << " " << DY << " 1\">\n"
      << "    <Piece Extent=\"0 " << NX-1 << " 0 " << NY-1 << " 0 0\">\n"
      << "      <PointData Scalars=\"Ez\">\n"
      << "        <DataArray type=\"Float64\" Name=\"Ez\" format=\"ascii\">\n";
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) f << g.ez(i,j) << " ";
        f << "\n";
    }
    f << "        </DataArray>\n"
      << "      </PointData>\n"
      << "    </Piece>\n"
      << "  </ImageData>\n"
      << "</VTKFile>\n";
}
