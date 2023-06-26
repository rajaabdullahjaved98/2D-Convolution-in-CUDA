#include <stdio.h>
#include "Timer.h"

#define WIDTH 512
#define HEIGHT 512
#define KERNEL_SIZE 15
#define BLOCK_SIZE 16

__global__ void convolutionKernel(const unsigned char* input, unsigned char* output) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Compute the global position of the thread
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // Shared memory for the convolution kernel
    __shared__ unsigned char sharedKernel[KERNEL_SIZE][KERNEL_SIZE];

    // Load the convolution kernel into shared memory
    if (ty < KERNEL_SIZE && tx < KERNEL_SIZE) {
        sharedKernel[ty][tx] = input[row * WIDTH + col];
    }
    __syncthreads();

    // Convolution operation
    int sum = 0;
    if (row < HEIGHT && col < WIDTH) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                sum += sharedKernel[i][j] * input[(row + i) * WIDTH + (col + j)];
            }
        }
        output[row * WIDTH + col] = sum / (KERNEL_SIZE * KERNEL_SIZE);
    }
}

int main() 
{
    //Declare Timer Variables
    Timer gputime;
    initTimer(&gputime, "GPU Execution Time: ");

    unsigned char *h_input, *h_output;
    unsigned char *d_input, *d_output;

    size_t size = WIDTH * HEIGHT * sizeof(unsigned char);

    // Allocate host memory
    h_input = (unsigned char*)malloc(size);
    h_output = (unsigned char*)malloc(size);

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Initialize input data with random values
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = rand() % 256;
    }

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //Start Timer
    startTimer(&gputime);

    // Launch the convolution kernel
    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output);

    //Stop Timer
    stopTimer(&gputime);

    //Print Execution Time
    printTimer(gputime);

    // Copy output data from device to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print the first few elements of the output for verification
    //for (int i = 0; i < 10; i++) {
    //    printf("%u ", h_output[i]);
    //}
    //printf("\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}


