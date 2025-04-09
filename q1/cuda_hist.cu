#include <stdio.h>
#include <stdlib.h>

#define NUM_BINS 32
#define RANGE 100000

const int bin_size = RANGE / NUM_BINS;
// CUDA error check macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }
// Kernel function to compute histogram
__global__ void histogram_kernel(int *data, int *histogram, int n) {
    extern __shared__ int local_hist[];

    // get current thread index
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global thread index

    // Initialize shared memory bins
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        local_hist[i] = 0;
    }

    __syncthreads(); // synchronize threads in the block

    // check if thread index is within bounds
    if (idx < n) {
        int bin = data[idx] / (RANGE / NUM_BINS);
        atomicAdd(&histogram[bin], 1);
    }

    __syncthreads(); // synchronize threads in the block

    // write the local histogram to global memory
    if (tid < NUM_BINS) {
        atomicAdd(&histogram[tid], local_hist[tid]);
    }
}

int main(int argc, char**argv) {
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
    int *h_histogram = (int *)malloc(sizeof(int) * NUM_BINS);

    // initialize host data
    srand(time(NULL)); // seed in host random number generator
    for (int i = 0; i < num_samples; i++) {
        h_data[i] = rand() % RANGE + 1; // 1 - RANGE
    }

    // define device data
    int *d_data, *d_histogram;
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_histogram, size);

    // copy data from host to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, size); // initialize histogram to 0

    // Start benchmarking time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // define block and grid sizes
    int threads = 512;
    int blocks = (num_samples + threads - 1) / threads;
    histogram_kernel<<<blocks, threads, NUM_BINS * sizeof(int)>>>(d_data, d_histogram, num_samples);

    // copy histogram from device to host
    cudaMemcpy(h_histogram, d_histogram, size, cudaMemcpyDeviceToHost);

    // Stop benchmarking time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken: %f ms\n", ms);

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

