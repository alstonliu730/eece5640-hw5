CC = gcc
NVCC = nvcc

CFLAGS = -Wall -Wextra -O3
OMP_FLAGS = -fopenmp
NVCCFLAGS = -O3

TARGETS = q1a q1b

q1a: cuda_hist.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

q1b: omp_hist.c
	$(CC) $(CFLAGS) -o $@ $< $(OMP_FLAGS)

clean:
	rm -f $(TARGETS)

