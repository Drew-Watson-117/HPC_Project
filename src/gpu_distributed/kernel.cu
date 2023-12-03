/*
        Each MPI process will be given an equal distribution of the data. For example, if there are
        4 processes working on our 6000*6000 dataset, then each process will get a (6000, 1500)
        subset of the data. Each gpu needs to be aware of which subset it is working on. This means
        the GPU will need to factor in an offset for both its start and end point. In the MPI implementation,
        I am already calculating work distributions of the 1D array, but for CUDA Kernels, we need to
        determine the start and end points in a 2d grid. For example:
        
        given 4 ranks:
            total grid: (6000,6000)
            possible indices into data: [0, 36M]
            rank 0 - start point(0,0); index[0], end point(5999, 1499); index[9M - 1]
            rank 1 - start point(0, 1500); index[9M], end point(5999, 2999); index[18M - 1]
            rank 3 - start point(0, 3000); index[18M], end point(5999, 4499); index[27M - 1]
            rank 4 - start point(0, 4500); index[27M], end point(5999, 5999); index[36M - 1]
            
            We can see the number of rows a given rank will process is 1500 rows.

        This means we need to be careful about how we shape our grid, but each rank should have the
        same shape of grid. Using the same grid size for all ranks will ensure there is no overlapping
        calulations. An approach we can take is always keeping the width of the share of work equal to the
        width of the original data. That way, we only need to worry about the number of rows a rank
        will process.

        It seems we need the following information to account for this additional complexity:
            grid_height - the number of rows a rank needs to process
                - computed via width / comm_sz
            offsetY_begin - The starting Y position for the sub-array a rank is working on
                - computed via grid_height * my_rank
            offsetY_end - the last Y position the rank needs to compute
                - computed via (grid_height * my_rank) + grid_height

        In the lineOfSightKernel we will need to offset our centerY by this offsetY_begin,
        and verify it is never computing things past offsetY_end.

        Outside of this nuance, everything else should be pretty typical. Here is the flow
        of the program:
        1. rank 0 reads in data; all other ranks resize their data array, and then that data
           is broadcast to all other threads.
        2. rank 0 will compute the grid_height so each rank can size it's grid appropiately
        3. A local_counts vector is created so each rank can store their respective counts
            - this vector should is sized to fit the amount of work a rank will do. The vector
              is not sized the same size as the data array. We will have to account for this
              when storing the computed results.
                - computed via offsetY_begin - (centerY * width + centerX)
        4. Device allocations are made to store information in GPU
        5. The launchLineOfSightKernel_MPI function is called on each rank. Each rank will set its
           block grid to the dimensions:
           ((width + dimBlock.x - 1) / dimBlock.x, (grid_height + dinBlock.y - 1) / grid_height)
        6. The lineOfSightKernel kernel is called, and it computes the line of sight in range
            - check if in allocation using `if (centerX < width && (centerY >= offsetY_begin || centerY < offsetY_end)`
        7. The data is copied off the device, and then gathered back to rank 0 using MPI_Gather()
        8. The resulting line of sight map is saved to a .raw file

 */

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

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


__global__ void lineOfSightKernel(int16_t *data, int *results, int width, int height, int offsetY_begin, int offsetY_end, int range) {
    int centerX = blockIdx.x * blockDim.x + threadIdx.x;
    int centerY = offsetY_begin + (blockIdx.y * blockDim.y + threadIdx.y);

    if (centerX < width && centerY < offsetY_end){
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

        results[(width * offsetY_begin) - centerY * width + centerX] = count;
    }
}

// Wrapper function
void launchLineOfSightKernel_MPI(int16_t *d_data, int *d_results, int width, int height, int offsetY_begin, int offsetY_end, int range) {
    
    dim3 dimBlock(16, 16); 
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    int sharedMemorySize = (dimBlock.x + 2 * range) * (dimBlock.y + 2 * range) * sizeof(int16_t);

    printf("Shared Memory Size %d\n", sharedMemorySize);

    //std::cout << "Launching Kernel" << std::endl;

    // Launch the kernel
    lineOfSightKernel<<<dimGrid, dimBlock, sharedMemorySize>>>(d_data, d_results, width, height, offsetY_begin, offsetY_end, range);

    // Check for errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        //std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}


extern "C" void cuda_bootstrapper(std::vector<int16_t> &data, std::vector<int> &counts, int16_t range, int width, int height, int my_rank, int comm_sz, int rows_per_rank, int row_overflow){
    int16_t *d_data;
    int *d_counts;

    cudaMalloc(&d_data, width * height * sizeof(int16_t));
    int offsetY_begin = my_rank * rows_per_rank;
    int offsetY_end;
    if(my_rank != comm_sz - 1){
        cudaMalloc(&d_counts, width * rows_per_rank * sizeof(int));
        offsetY_end = (my_rank * rows_per_rank) + (rows_per_rank - 1);
    }
    else{
        cudaMalloc(&d_counts, width * (rows_per_rank + row_overflow) * sizeof(int));
        offsetY_end = (my_rank * rows_per_rank) + (rows_per_rank - 1) + row_overflow;
    }

    cudaMemcpy(d_data, data.data(), width * rows_per_rank * sizeof(int16_t), cudaMemcpyHostToDevice);
    if(my_rank != comm_sz - 1){
        launchLineOfSightKernel_MPI(d_data, d_counts, width, rows_per_rank, offsetY_begin, offsetY_end, range);
    }
    else{
        launchLineOfSightKernel_MPI(d_data, d_counts, width, rows_per_rank + row_overflow, offsetY_begin, offsetY_end, range);
    }
    cudaDeviceSynchronize();
    
    cudaError_t cudaStatus = cudaGetLastError();
    
    cudaMemcpy(counts.data(), d_counts, width * rows_per_rank * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_counts);
}

