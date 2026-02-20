#ifndef FDTD_SEQ_H
#define FDTD_SEQ_H

#include "fdtd_simple.h"

class SequentialFDTD {
private:
    Grid grid;
    std::unique_ptr<BoundaryCondition> bc;
    Source source;
    int total_steps;
    int save_interval;

public:
    SequentialFDTD(const Source& src, int steps, int save)
        : source(src), total_steps(steps), save_interval(save) {}

    void setBoundary(std::unique_ptr<BoundaryCondition> boundary) {
        bc = std::move(boundary);
    }

    void run();
    const Grid& getGrid() const { return grid; }
};

#endif // FDTD_SEQ_H