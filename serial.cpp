#include <iostream>
#include <cstdlib>
#include <ctime>
#include "Timer.h"

// Constants
const int MATRIX_SIZE = 512;
const int KERNEL_SIZE = 15;

// Function to perform convolution
void convolution(const unsigned char* inputMatrix, const float* kernel, unsigned char* outputMatrix) {
    // Loop over the input matrix
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            float sum = 0.0;
            
            // Apply the convolution kernel
            for (int k = 0; k < KERNEL_SIZE; k++) {
                for (int l = 0; l < KERNEL_SIZE; l++) {
                    int x = i - KERNEL_SIZE / 2 + k;
                    int y = j - KERNEL_SIZE / 2 + l;
                    
                    // Handle boundary conditions by clamping
                    if (x < 0) x = 0;
                    if (x >= MATRIX_SIZE) x = MATRIX_SIZE - 1;
                    if (y < 0) y = 0;
                    if (y >= MATRIX_SIZE) y = MATRIX_SIZE - 1;
                    
                    sum += inputMatrix[x * MATRIX_SIZE + y] * kernel[k * KERNEL_SIZE + l];
                }
            }
            
            // Store the result in the output matrix
            outputMatrix[i * MATRIX_SIZE + j] = sum;
        }
    }
}

int main() 
{
    //Declare Timer Variable
    Timer t1("Serial Execution Time: ");

    // Seed the random number generator
    std::srand(std::time(nullptr));
    
    // Create the input grayscale image matrix
    unsigned char inputMatrix[MATRIX_SIZE * MATRIX_SIZE];
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        inputMatrix[i] = std::rand() % 256;  // Random pixel value between 0 and 255
    }
    
    // Create the convolution kernel
    float kernel[KERNEL_SIZE * KERNEL_SIZE];
    // Initialize the kernel with your desired values here
    
    // Create the output grayscale image matrix
    unsigned char outputMatrix[MATRIX_SIZE * MATRIX_SIZE];
    
    //Start Timer
    t1.Start();

    // Perform convolution
    convolution(inputMatrix, kernel, outputMatrix);

    //Stop Timer
    t1.Stop();

    //Print Execution Time
    t1.Print();
    
    // Print the output matrix if desired
    /*
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            std::cout << static_cast<int>(outputMatrix[i * MATRIX_SIZE + j]) << " ";
        }
        std::cout << std::endl;
    }
    */
    
    return 0;
}


