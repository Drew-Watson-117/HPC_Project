#include "helper_functions.hpp"
#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
/*  
    Given an elevation map, a starting coordinate, and an ending coordinate, count the number of pixels the two points which have line of sight with the starting point.
    May specify x dimension of the elevation map xDim and whether to display debug messages

    compile w/o build tool: mpic++ -g -Wall -o gpu_distributed gpu_distributed.cpp helper_functions.hpp helper_functions.cpp
    run: mpiexec -n [process count] ./cpu_distributed
*/
void get_args(int argc, char* argv[], int &range);
extern "C++" void cuda_bootstrapper(std::vector<int16_t> &data, std::vector<int> &counts, int16_t range, int width, int height, int my_rank, int comm_sz, int rows_per_rank, int row_overflow);

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

    //grab desired line of sight range from args and broadcast it to all other ranks
    int range;
    if (my_rank == 0){
    	get_args(argc, argv, range);
	std::cout << "Checking line of sight with range: " << range << std::endl;
    }
    MPI_Bcast(&range, 1, MPI_INT, 0, comm);


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
        cuda_bootstrapper(data, local_counts, range, WIDTH, HEIGHT, my_rank, comm_sz, rows_per_rank, row_overflow);
    }
    else{
        cuda_bootstrapper(data, local_counts, range, WIDTH, HEIGHT, my_rank, comm_sz, rows_per_rank, row_overflow);

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
    //     for(int i = 50; i < 100; i++){
	// 	std::cout << global_counts[i] << std::endl;
	// }
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

//grabs arguments from CLI and processes them for program
void get_args(int argc, char* argv[], int &range) {
   if(argc != 2){
        std::cout << "insufficiant number of arguments. Exiting";
        exit(1); //exit code 1 to indicate error
   }
   else{
    	range = atoi(argv[1]);    
   }
}  /* Get_args */



