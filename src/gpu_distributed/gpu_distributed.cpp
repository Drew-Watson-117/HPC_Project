#include "helper_functions.hpp"
#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <mpi.h>
/*  
    Given an elevation map, a starting coordinate, and an ending coordinate, count the number of pixels the two points which have line of sight with the starting point.
    May specify x dimension of the elevation map xDim and whether to display debug messages

    compile w/o build tool: mpic++ -g -Wall -o gpu_distributed gpu_distributed.cpp helper_functions.hpp helper_functions.cpp
    run: mpiexec -n [process count] ./cpu_distributed
*/
int countLineOfSight(std::vector<int16_t> &data, int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t xDim = 6000, bool debug = false);


void MPI_CUDA_line_of_sight(std::vector<int16_t> &data, std::vector<int> &counts, int16_t range, int my_rank, int comm_sz, int data_split, int data_overflow);

void launchLineOfSightKernel_MPI(int16_t *d_data, int16_t *d_results, int width, int height, int offsetY_begin, int offsetY_end, int range);
extern "C" void cuda_bootstrapper(std::vector<int16_t> &data, std::vector<int> &counts, int16_t range, int width, int height, int my_rank, int comm_sz, int rows_per_rank, int row_overflow);
//global communication
MPI_Comm comm;

const int DATA_COUNT = 6000 * 6000; 

int main(int argc, char* argv[]) {
    const int HEIGHT = 6000;
    const int WIDTH = 6000;
    //initialize MPI
    MPI_Init(&argc, &argv);
    
    int my_rank;
    int comm_sz;
   
    //get number of processes available
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    //get rank of process currently executing function
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    //communication channel
    comm = MPI_COMM_WORLD;


    std::vector<int16_t> data(DATA_COUNT);
    std::vector<int> global_counts(DATA_COUNT);
    
    //arrays to store calculated distributions & displacements
    //(needed by MPI_Gatherv)
    int work_distribution[comm_sz];
    int displacements[comm_sz];
    
    //variables to determine how to split up work
    int row_overflow = HEIGHT % comm_sz; // any work not evenly divisible by num rank
    int rows_per_rank = ((HEIGHT - (HEIGHT % comm_sz)) / comm_sz); // number of rows each rank will process (besides comm_sz - 1)
    
    int data_overflow = row_overflow * WIDTH; //number of additional data points last rank will process
    int data_split = rows_per_rank * WIDTH; //number of data points each rank will process

    //rank 0 loads data and broadcasts it to all other threads
    if (my_rank == 0){
       std::cout << "getting data from file" << std::endl;
       data = getData();

       int curr_displacement = 0;
       for(int i = 0; i < comm_sz; i++){
           if(i == (comm_sz - 1)){ // last rank potentially takes on extra work
               displacements[i] = curr_displacement;
               work_distribution[i] = data_split + data_overflow;
           }
           else{ // all other ranks take on same amt of work
               displacements[i] = curr_displacement;
               work_distribution[i] = data_split;
               curr_displacement += data_split;
           }
       }
       for(int i = 0; i < comm_sz; i++){
           if(i != comm_sz - 1){ // if I am not the last rank 
               std::cout << "rank " << i << " working on rows " << i * rows_per_rank  << " - " << (i * rows_per_rank) + (rows_per_rank - 1) << std::endl;
               std::cout << "\t number of data points for this rank: " << data_split << std::endl;
           }
           else{
               std::cout << "last rank " << i << " working on rows " << (i * rows_per_rank) << " - " << (i * rows_per_rank) + (rows_per_rank - 1) + row_overflow << std::endl;
               std::cout << "\t number of data points for this rank: " << data_overflow + data_split << std::endl;
           }
       }
    }
    // resize gloabl counts to avoid segfault down the line
    global_counts.resize(DATA_COUNT);

    //all other ranks resize their data vector in anticipation of the incoming data
    if (my_rank != 0){
        data.resize(DATA_COUNT);
    }
   
    //broadcast vector size & work distribution to all other threads
    MPI_Bcast(data.data(), DATA_COUNT, MPI_SHORT, 0, comm);
    MPI_Bcast(&work_distribution, comm_sz, MPI_INT, 0, comm);
    MPI_Bcast(&displacements, comm_sz, MPI_INT, 0, comm);
    std::vector<int> local_counts(work_distribution[my_rank]);
    
    //allocate GPU resources and calculate local line of sights
    if(my_rank != comm_sz - 1){
        cuda_bootstrapper(data, local_counts, 10, WIDTH, rows_per_rank, my_rank, comm_sz, rows_per_rank, row_overflow);
    }
    else{
        cuda_bootstrapper(data, local_counts, 10, WIDTH, rows_per_rank + row_overflow, my_rank, comm_sz, rows_per_rank, row_overflow);

    }


    MPI_Barrier(comm);
    
    MPI_Gatherv(
            &local_counts.front(),
            local_counts.size(), 
            MPI_INT, 
            &global_counts.front(), 
            work_distribution,
            displacements,
            MPI_INT, 
            0, 
            comm
            );

    if(my_rank == 0){
        //cannot use writeFile from helperfunctions as freeing buffer causes seg. fault.
        std::ofstream file;
        file.open("gpu_distributed_out.raw", std::ios::binary);
        if (file.is_open()) {
            char* buffer = reinterpret_cast<char*>(global_counts.data());
            size_t size = global_counts.size() * sizeof(int);
            file.write(buffer, size);
            file.close();
        } 
    }
    //shut down mpi
    MPI_Finalize(); 
   
    return 0;
}

void MPI_CUDA_line_of_sight(std::vector<int16_t> &data, std::vector<int> &counts, int16_t range, int my_rank, int comm_sz, int data_split, int data_overflow){
    int start_point, end_point; 
    start_point = data_split * my_rank;
    
    if(my_rank == comm_sz - 1){ // if I am the last process, then I get some extra work if needed
        end_point = start_point + data_split + data_overflow; 
    }
    else{
        end_point = start_point + data_split;
    }

    for (int i = start_point; i < end_point; i++){
        int centerX = i % 6000;
        int centerY = i / 6000;
        
        int count = 0;
        // For each pixel on the edge of a square of radius 100 around the center pixel
        // std::max and std::min ensure considerX and considerY are in the bounds of image

		// Fix considerX at centerX - 100, vary considerY
		for (short considerY = std::max(centerY - range, 0); considerY <= std::min(centerY + range,5999); ++considerY) {
			count += countLineOfSight(data,centerX,centerY,std::max(0,centerX - range),considerY);
		}
		// Fix considerX at centerX + 100, vary considerY
		for (short considerY = std::max(centerY - range, 0); considerY <= std::min(centerY + range,5999); ++considerY) {
			count += countLineOfSight(data,centerX,centerY,std::min(5999,centerX + range),considerY);
		}
		// Fix considerY at centerY - 100, vary considerX
		for (short considerX = std::max(centerX - range, 0); considerX <= std::min(centerX + range,5999); ++considerX) {
			count += countLineOfSight(data,centerX,centerY,considerX,std::min(5999,centerY + range));
		}
		// Fix considerY at centerY + 100, vary considerX
		for (short considerX = std::max(centerX - range, 0); considerX <= std::min(centerX + range,5999); ++considerX) {
			count += countLineOfSight(data,centerX,centerY,considerX,std::min(5999,centerY + range));
		}
        counts[i - start_point] = count;
   }

}


int countLineOfSight(std::vector<int16_t> &data,int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t xDim, bool debug)
{
	// count: Number of points in the straight line between (x1,y1) and (x2,y2) that have line of sight with (x1,y1)
	int count = 0;

	// Compute the differences between start and end points
    
	int dx = x2 - x1;
	int dy = y2 - y1;

	// Absolute values of the change in x and y
	const int abs_dx = abs(dx);
	const int abs_dy = abs(dy);

	// Initial point
	int x = x1;
	int y = y1;
	double z1 = static_cast<double>(data[getIndex(x1,y1,xDim)]);
	// z: the real elevation of the currently considered point
  double z = z1;
	// zLineOfSight: the z value for the "line of sight" line between (x1,y1) and (x,y) at point (xMax,yMax)
  double zLineOfSightAtMax;
	// slope: slope of the "line of sight" line between (x1,y1) and (x,y)
	double slope;
	// zMax: the largest elevation value between (x1,y1) and (x,y). Full coordinate is (xMax,yMax,zMax)
	int xMax = x1;
	int yMax = y1;
	double zMax = z1;

	// Proceed based on the absolute differences to support all octants
	if (abs_dx > abs_dy)
	{
		// If the line is moving to the left, set dx accordingly
		int dx_update;
		if (dx > 0)
		{
			dx_update = 1;
		}
		else
		{
			dx_update = -1;
		}

		// Calculate the initial decision parameter
		int p = 2 * abs_dy - abs_dx;

		// Draw the line for the x-major case
		for (int i = 0; i <= abs_dx; i++)
		{
			// Get z and determine whether it is zMax
            z = static_cast<double>(data[getIndex(x,y,xDim)]);
			if (z > zMax) {
				xMax = x;
				yMax = y;
				zMax = z;
			}
			// Compute the slope for line of sight from (x1,y1,z1) to (x,y,z)
			slope = (z-z1) / std::sqrt(std::pow((x-x1),2)+std::pow((y-y1),2));
			
			// Compute the line of sight z value at (xMax,yMax,zMax)
            zLineOfSightAtMax = slope * std::sqrt(std::pow((xMax-x1),2)+std::pow((yMax-y1),2)) + z1;
            

            // Print debug messages
            if (debug) {
                std::cout << "(" << x << "," << y << "," << z << ")" << std::endl;
				std::cout << "Slope = " << slope << std::endl;
                std::cout << "zLineOfSightAtMax = " << zLineOfSightAtMax << std::endl;
                std::cout << "z = " << z << std::endl;
				std::cout << "zMax = " << zMax << std::endl;
				if (zMax <= zLineOfSightAtMax) std::cout << "Iterating Count" << std::endl;
				std::cout << "\n" << std::endl;
            }
			
			// If the real value of zMax does not obstruct the line of sight z value at (xMax,yMax), then there is line of sight, iterate count
            if (zMax <= zLineOfSightAtMax) {
                count++;
            }
            

			// Threshold for deciding whether or not to update y
			if (p < 0)
			{
				p = p + 2 * abs_dy;
			}
			else
			{
				// Update y
				if (dy >= 0)
				{
					y += 1;
				}
				else
				{
					y += -1;
				}

				p = p + 2 * abs_dy - 2 * abs_dx;
			}

			// Always update x
			x += dx_update;
		}
	}
	else
	{
		// If the line is moving downwards, set dy accordingly
		int dy_update;
		if (dy > 0)
		{
			dy_update = 1;
		}
		else
		{
			dy_update = -1;
		}

		// Calculate the initial decision parameter
		int p = 2 * abs_dx - abs_dy;

		// Draw the line for the y-major case
		for (int i = 0; i <= abs_dy; i++)
		{
            // Get z and determine whether it is zMax
            z = static_cast<double>(data[getIndex(x,y,xDim)]);
			if (z > zMax) {
				xMax = x;
				yMax = y;
				zMax = z;
			}
			// Compute the slope for line of sight from (x1,y1,z1) to (x,y,z)
			slope = (z-z1) / std::sqrt(std::pow((x-x1),2)+std::pow((y-y1),2));

			// Compute the line of sight z value at (xMax,yMax,zMax)
            zLineOfSightAtMax = slope * std::sqrt(std::pow((xMax-x1),2)+std::pow((yMax-y1),2)) + z1;
            

            // Print debug messages
            if (debug) {
                std::cout << "(" << x << "," << y << "," << z << ")" << std::endl;
				std::cout << "Slope = " << slope << std::endl;
                std::cout << "zLineOfSightAtMax = " << zLineOfSightAtMax << std::endl;
                std::cout << "z = " << z << std::endl;
				std::cout << "zMax = " << zMax << std::endl;
				if (zMax <= zLineOfSightAtMax) std::cout << "Iterating Count" << std::endl;
				std::cout << "\n" << std::endl;
            }
			
			// If the real value of zMax does not obstruct the line of sight z value at (xMax,yMax), then there is line of sight, iterate count
            if (zMax <= zLineOfSightAtMax) {
                count++;
            }

			// Threshold for deciding whether or not to update x
			if (p < 0)
			{
				p = p + 2 * abs_dx;
			}
			else
			{
				// Update x
				if (dx >= 0)
				{
					x += 1;
				}
				else
				{
					x += -1;
				}

				p = p + 2 * abs_dx - 2 * abs_dy;
			}

			// Always update y
			y += dy_update;
		}
	}
	if (debug) {
		std::cout << "Returning " << count << std::endl;
	}
    return count;
}


