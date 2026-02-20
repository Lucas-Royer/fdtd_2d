# Makefile pour un exécutable unique supportant les trois modes (séquentiel, OpenMP, CUDA)
CXX      = g++
CXXFLAGS = -std=c++17 -O3 -Wall -I.
NVCC     = nvcc
CUDA_PATH ?= /usr/local/cuda
NVCCFLAGS = -std=c++17 -O3 -arch=sm_89 -I. -Xcompiler -fopenmp

# Liste des fichiers objets (tous nécessaires pour l'exécutable final)
OBJS = main.o fdtd_simple.o fdtd_seq.o fdtd_omp.o fdtd_cuda.o

# Cible par défaut
all: fdtd2d

# Règle générique pour les fichiers C++
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Règle spécifique pour fdtd_omp.cpp (doit être compilé avec OpenMP)
fdtd_omp.o: fdtd_omp.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -c $< -o $@

# Règle pour les fichiers CUDA
fdtd_cuda.o: fdtd_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Édition des liens : on linke tous les objets avec les bibliothèques OpenMP et CUDA
fdtd2d: $(OBJS)
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $(OBJS) \
	    -L$(CUDA_PATH)/lib64 -lcudart -Wl,-rpath,$(CUDA_PATH)/lib64

# Nettoyage
clean:
	rm -f $(OBJS) fdtd2d

.PHONY: all clean