# Makefile pour un exécutable unique supportant les trois modes (séquentiel, OpenMP, CUDA)
CXX      = g++
CXXFLAGS = -std=c++17 -O3 -Wall -I.
NVCC     = nvcc
CUDA_PATH ?= /usr/local/cuda
NVCCFLAGS = -std=c++17 -O3 -arch=sm_89 -I. -Xcompiler -fopenmp

OBJS = main.o fdtd_simple.o fdtd_seq.o fdtd_omp.o fdtd_cuda.o

all: fdtd2d

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

fdtd_omp.o: fdtd_omp.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -c $< -o $@

fdtd_cuda.o: fdtd_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

fdtd2d: $(OBJS)
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $(OBJS) \
	    -L$(CUDA_PATH)/lib64 -lcudart -Wl,-rpath,$(CUDA_PATH)/lib64

clean:
	rm -f $(OBJS) fdtd2d

.PHONY: all clean