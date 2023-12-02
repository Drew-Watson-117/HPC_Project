#include <iostream>
#include <vector>
#include "helper_functions.hpp"
#include <cuda_runtime.h>

// Declare the wrapper function for the kernel
void launchLineOfSightKernel(int16_t *d_data, int16_t *d_results, int width, int height, int range);

int main() {
    const int width = 6000;
    const int height = 6000;
    const int range = 10;  // Set the range for the line of sight calculations

    std::cout << "Processing File" << std::endl;

    std::vector<int16_t> data = getData(); // Get elevation data
    std::vector<int16_t> results(width * height, 0);

    int16_t *d_data, *d_results;
    cudaMalloc(&d_data, width * height * sizeof(int16_t));
    cudaMalloc(&d_results, width * height * sizeof(int16_t));

    std::cout << "Memory Allocated" << std::endl;

    cudaMemcpy(d_data, data.data(), width * height * sizeof(int16_t), cudaMemcpyHostToDevice);

    std::cout << "Data Copied" << std::endl;

    // Call the wrapper function to launch the kernel
    launchLineOfSightKernel(d_data, d_results, width, height, range);

    std::cout << "Kernal Launched" << std::endl;

    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    // Copy results back to host
    cudaMemcpy(results.data(), d_results, width * height * sizeof(int16_t), cudaMemcpyDeviceToHost);

    std::cout << "Data Copied back" << std::endl;

    // Write results to a binary file
    bool success = writeFile(results, "output_gpu.raw");
    success ? std::cout << "Done!" << std::endl : std::cout << "Error Writing to File" << std::endl;

    // Free GPU memory
    cudaFree(d_data);
    cudaFree(d_results);

    return 0;
}
