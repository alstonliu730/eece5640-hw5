#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NUM_BINS 32
#define RANGE 100000

const int bin_size = RANGE / NUM_BINS;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

// CUDA error check macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }
// Kernel function to compute histogram
__global__ void histogram_kernel(int *data, int *histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int bin = data[idx] / bin_size;
        if (bin >= NUM_BINS) {
            bin = NUM_BINS - 1; // Ensure bin index is within bounds
        }

        atomicAdd(&histogram[bin], 1); // Increment the histogram bin
    }
}

int main(int argc, char** argv) {
    // Check if the number of arguments is correct
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_samples>\n", argv[0]);
        return 1;
    }

    // Parse the number of samples from the command line argument
    int num_samples = atoi(argv[1]);
    if (num_samples <= 0) {
        fprintf(stderr, "Error: Number of samples must be a positive integer.\n");
        return 1;
    }

    // number of samples in bytes
    size_t size = sizeof(int) * num_samples;

    // define host data
    int *h_data = (int *)malloc(size);
    int *h_histogram = (int *)calloc(NUM_BINS, sizeof(int)); // initialize to 0

    // initialize host data
    srand(time(NULL)); // seed in host random number generator
    for (int i = 0; i < num_samples; i++) {
        h_data[i] = rand() % RANGE + 1; // 1 - RANGE
    }

    // define device data
    int *d_data, *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BINS * sizeof(int)));

    // copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_histogram, h_histogram, NUM_BINS * sizeof(int), cudaMemcpyHostToDevice));

    // Start benchmarking time
    double start, stop;
    start = CLOCK();

    // define block and grid sizes
    int threads = 512; // number of threads per block
    int blocks = (num_samples + threads - 1) / threads; // number of blocks
    histogram_kernel<<<blocks, threads>>>(d_data, d_histogram, num_samples);
    CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors

    // copy histogram from device to host
    CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    // Stop benchmarking time
    stop = CLOCK();

    printf("Time taken: %f ms\n", stop - start);
    printf("Num. of Samples: %d\n", num_samples);

    // print histogram
    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %d: %d\n", i, h_histogram[i]);
    }

    // free device memory
    cudaFree(d_data);
    cudaFree(d_histogram);
    
    // free host memory
    free(h_data);
    free(h_histogram);

    return 0;
}   

