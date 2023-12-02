#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__device__ int countLineOfSight(int16_t *data, int x1, int y1, int x2, int y2, int xDim) {
    int count = 0;

    int dx = x2 - x1;
    int dy = y2 - y1;
    const int abs_dx = abs(dx);
    const int abs_dy = abs(dy);

    int x = x1;
    int y = y1;
    double z1 = static_cast<double>(data[y1 * xDim + x1]);
    double z = z1;
    double zLineOfSightAtMax;
    double slope;
    int xMax = x1;
    int yMax = y1;
    double zMax = z1;

    if (abs_dx > abs_dy) {
        int dx_update = (dx > 0) ? 1 : -1;
        int p = 2 * abs_dy - abs_dx;

        for (int i = 0; i <= abs_dx; ++i) {
            z = static_cast<double>(data[y * xDim + x]);
            if (z > zMax) {
                xMax = x;
                yMax = y;
                zMax = z;
            }
            slope = (z - z1) / sqrt(pow(x - x1, 2) + pow(y - y1, 2));
            zLineOfSightAtMax = slope * sqrt(pow(xMax - x1, 2) + pow(yMax - y1, 2)) + z1;

            if (zMax <= zLineOfSightAtMax) {
                count++;
            }

            if (p < 0) {
                p += 2 * abs_dy;
            } else {
                y += (dy >= 0) ? 1 : -1;
                p += 2 * (abs_dy - abs_dx);
            }
            x += dx_update;
        }
    } else {
        int dy_update = (dy > 0) ? 1 : -1;
        int p = 2 * abs_dx - abs_dy;

        for (int i = 0; i <= abs_dy; ++i) {
            z = static_cast<double>(data[y * xDim + x]);
            if (z > zMax) {
                xMax = x;
                yMax = y;
                zMax = z;
            }
            slope = (z - z1) / sqrt(pow(x - x1, 2) + pow(y - y1, 2));
            zLineOfSightAtMax = slope * sqrt(pow(xMax - x1, 2) + pow(yMax - y1, 2)) + z1;

            if (zMax <= zLineOfSightAtMax) {
                count++;
            }

            if (p < 0) {
                p += 2 * abs_dx;
            } else {
                x += (dx >= 0) ? 1 : -1;
                p += 2 * (abs_dx - abs_dy);
            }
            y += dy_update;
        }
    }
    
    return count;
}


__global__ void lineOfSightKernel(int16_t *data, int16_t *results, int width, int height, int range) {
    int centerX = blockIdx.x * blockDim.x + threadIdx.x;
    int centerY = blockIdx.y * blockDim.y + threadIdx.y;

    if (centerX < width && centerY < height) {
        int count = 0;

        // Define the boundary of the square around the center pixel
        int startX = max(0, centerX - range);
        int endX = min(width - 1, centerX + range);
        int startY = max(0, centerY - range);
        int endY = min(height - 1, centerY + range);

        // Iterate over the boundary pixels
        for (int x = startX; x <= endX; x++) {
            count += countLineOfSight(data, centerX, centerY, x, startY, width);
            count += countLineOfSight(data, centerX, centerY, x, endY, width);
        }
        for (int y = startY + 1; y < endY; y++) { // +1 and < to avoid double counting corners
            count += countLineOfSight(data, centerX, centerY, startX, y, width);
            count += countLineOfSight(data, centerX, centerY, endX, y, width);
        }

        results[centerY * width + centerX] = count;
    }
}

// Wrapper function
void launchLineOfSightKernel(int16_t *d_data, int16_t *d_results, int width, int height, int range) {
    dim3 dimBlock(16, 16); 
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    int sharedMemorySize = (dimBlock.x + 2 * range) * (dimBlock.y + 2 * range) * sizeof(int16_t);

    printf("Shared Memory Size %d\n", sharedMemorySize);

    std::cout << "Launching Kernel" << std::endl;

    // Launch the kernel
    lineOfSightKernel<<<dimGrid, dimBlock, sharedMemorySize>>>(d_data, d_results, width, height, range);

    // Check for errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}




