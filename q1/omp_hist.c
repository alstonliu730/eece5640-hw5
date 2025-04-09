#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <math.h>

#define NUM_BINS 32
#define RANGE 10000

const int bin_size = RANGE / NUM_BINS;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main(int argc, char *argv[]) {
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

    // Initialize the histogram
    int hist[NUM_BINS] = {0};

    // Initialize the random number generator
    unsigned int seed = 0;
    int num = 0;
    
    // Benchmark parameters
    double start_time, end_time;
    start_time = CLOCK();

    // Generate random numbers and fill the histogram
    #pragma omp parallel
    {
        seed = time(NULL) + omp_get_thread_num();

        #pragma omp for reduction(+:hist[:NUM_BINS]) private (seed, num)
        for(int i = 0; i < num_samples; i++) {
            // Generate a random number in the range [0, RANGE)
            num = rand_r(&seed) % RANGE;

            // Determine the bin index
            int bin_index = num / bin_size;

            // Increment the corresponding bin
            if (bin_index < NUM_BINS) {
                #pragma omp atomic
                hist[bin_index]++;
            } else {
                #pragma omp atomic
                // If the number is out of range, increment the last bin
                hist[NUM_BINS - 1]++;
            }
        }
    }

    // End the benchmark
    end_time = CLOCK();

    // Print the histogram
    printf("Histogram:\n");
    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %d: %d\n", i, hist[i]);
    }

    // Print the execution time
    printf("Num. of Samples: %d\n", num_samples);
    printf("Execution time: %0.6f ms\n", (end_time - start_time));
}
