#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Original Stencil code
/*
#define n 32
float a[n][n][n], b[n][n][n];
for (int i = 1; i < n-1; i++)
    for (int j = 1; j < n-1; j++)
        for (int k = 1; k < n-1; k++) {
            a[i][j][k] = 0.75f * (b[i-1][j][k] + b[i+1][j][k] +
                          b[i][j-1][k] + b[i][j+1][k] +
                          b[i][j][k-1] + b[i][j][k+1]);
        }
    }
}       
*/

#define TILE_SIZE 8

// untiled stencil code
__global__ void stencil_naive(float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < n - 1 && j < n - 1 && k < n - 1) {
        int idx = i * n * n + j * n + k;
        a[idx] = 0.75f * (b[(i - 1) * n * n + j * n + k] +
                         b[(i + 1) * n * n + j * n + k] +
                         b[i * n * n + (j - 1) * n + k] +
                         b[i * n * n + (j + 1) * n + k] +
                         b[i * n * n + j * n + (k - 1)] +
                         b[i * n * n + j * n + (k + 1)]);
    }
}

// tiled stencil code
__global__ void stencil_tiled(float* a, float* b, int n) {
    // Shared memory for tiles -- add padding to tile
    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2][TILE_SIZE + 2];

    // 
    int i = blockIdx.x * TILE_SIZE + threadIdx.x;
    int j = blockIdx.y * TILE_SIZE + threadIdx.y;
    int k = blockIdx.z * TILE_SIZE + threadIdx.z;

    // tile indices (to account for padding)
    int ti = threadIdx.x + 1;
    int tj = threadIdx.y + 1;
    int tk = threadIdx.z + 1;

    // global index
    int idx = i * n * n + j * n + k; // global starting index in b

    // load center data into shared memory
    if (i < n && j < n && k < n) {
        tile[ti][tj][tk] = b[idx];
    }

    // Load data into shared memory from neighbors while checking boundaries
    if (threadIdx.x == 0 && i > 0) tile[ti - 1][tj][tk] = b[(i - 1) * n * n + j * n + k];
    if (threadIdx.x == TILE_SIZE - 1 && i < n - 1) tile[ti + 1][tj][tk] = b[(i + 1) * n * n + j * n + k];
    if (threadIdx.y == 0 && j > 0) tile[ti][tj - 1][tk] = b[i * n * n + (j- 1) * n + k];
    if (threadIdx.y == TILE_SIZE - 1 && j < n - 1) tile[ti][tj + 1][tk] = b[i * n * n + (j + 1) * n + k];
    if (threadIdx.z == 0 && k > 0) tile[ti][tj][tk - 1] = b[i * n * n + j * n + (k - 1)];
    if (threadIdx.z == TILE_SIZE - 1 && k < n - 1) tile[ti][tj][tk + 1] = b[i * n * n + j * n + (k + 1)];

    // wait for all threads to load data into shared memory
    __syncthreads();

    // Compute the stencil operation
    if (i > 0 && i < n - 1 && j > 0 && j < n - 1 && k > 0 && k < n - 1) {
        a[idx] = 0.75f * (tile[ti - 1][tj][tk] + 
                            tile[ti + 1][tj][tk] +
                            tile[ti][tj - 1][tk] +
                            tile[ti][tj + 1][tk] +
                            tile[ti][tj][tk - 1] +
                            tile[ti][tj][tk + 1]);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <mode (0|1)> <TILE_SIZE>\n", argv[0]);
        return 1;
    }

    int n = 32;
    int mode = atoi(argv[1]);
    if (mode != 0 && mode != 1) {
        fprintf(stderr, "Invalid mode. Use 0 for naive and 1 for tiled.\n");
        return 1;
    }

    if (argc == 3){
        n = atoi(argv[2]);
    }

    size_t size = n * n * n * sizeof(float);
    float *h_a, *h_b;

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);

    // Initialize b with some values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++){
                h_b[i * n * n + j * n + k] = (float)(i + j + k);
            }
        }
    }

    float *d_a, *d_b;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);

    // Copy b to device
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 gridSize(n / TILE_SIZE, n / TILE_SIZE, n / TILE_SIZE);

    // Launch kernel
    if (mode == 0) {
        stencil_naive<<<gridSize, blockSize>>>(d_a, d_b, n);
    } else {
        stencil_tiled<<<gridSize, blockSize>>>(d_a, d_b, n);
    }
    
    // Copy result back to host
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
    
    // End Benchmark
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // print the result
    printf("Result of stencil operation (%s):\n", (mode == 0) ? "naive" : "tiled");
    printf("n = %d\n", n);
    printf("Elapsed time: %f ms\n", elapsedTime);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);

    // Free host memory
    free(h_a);
    free(h_b);

    return 0;
}
