#ifndef FDTD_OMP_H
#define FDTD_OMP_H

#include "fdtd_simple.h"

class OpenMPFDTD {
private:
    Grid grid;
    std::unique_ptr<BoundaryCondition> bc;
    Source source;
    int total_steps;
    int save_interval;
    int num_threads;

public:
    OpenMPFDTD(const Source& src, int steps, int save, int threads = 0)
        : source(src), total_steps(steps), save_interval(save), num_threads(threads) {}

    void setBoundary(std::unique_ptr<BoundaryCondition> boundary) {
        bc = std::move(boundary);
    }
    void setNumThreads(int t) { num_threads = t; }

    void run();
    const Grid& getGrid() const { return grid; }
};

#endif // FDTD_OMP_H