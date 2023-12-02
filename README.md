# HPC Project: AwannaCU

## Serial Algorithm

### Instructions for build and execution

- To run the serial implementation, run `$ bash ./run_serial.sh`

### Approach for the serial program

Consider two points 1 `(x1,y1,z1)` and 2 `(x2,y2,z2)`. Point 2 is in the line of sight of point 1 if and only if the maximum elevation value `zMax` between `(x1,y1)` and `(x2,y2)` is less than or equal to the z value of the **line of sight** `zLineOfSightAtMax` at that point. The **line of sight** is the straight line between `(x1,y1,z1)` and `(x2,y2,z2)`. In the case of `countLineOfSight`, the line of sight is the straight line between `(x1,y1,z1)` and `(x,y,z)`. 

- `int countLineOfSight(data, x1, y1, x2, y2)`
    - Some variables:
        - x1, y1: Initial coordinate
        - x2, y2: End coordinate
        - x, y: Current coordinates being considered by the algorithm
        - zMax: The maximum elevation encountered on the line between (x1,y1) and (x,y).
        - xMax, yMax: Coordinates such that (xMax,yMax,zMax).
        - zLineOfSightAtMax: The z value of the line of sight line at (xMax,yMax)
    - This function computes a straight line between (x1,y1) and (x2,y2) and along that line checks zMax against zLineOfSightAtMax. If at some point (x,y) (x1,y1) to (x2,y2), zMax <= zLineOfSightAtMax, then there is line of sight between (x1,y1) and (x,y), so the function iterates a count. The function returns that count at the end of execution.

In `serial(int range)`, the program will read in the data from the file (a helper function to do this is written in `helper_functions.cpp`), then for each pixel (centerX, centerY) in the image, compute `countLineOfSight(data,centerX,centerY,considerX,considerY)`, where (considerX,considerY) iterates over every pixel in the perimeter of 100 pixel square radius. Add the results for all `countLineOfSight` calculations for each (centerX,centerY), and write to an array. After the count has been computed for every pixel, the array holding the counts is written to a file `output_serial.raw`

The `main()` function simply calls the `serial()` function, specifying a range (which is how far away each pixel should look for line of sight)

## CPU Shared Memory Algorithm

### Instructions for build and execution

- To run the cpu shared memory implementation, run `$ bash ./run_cpu_shared.sh {thread_count}`, where `{thread_count}` is the number of threads you want to run the program.

### Approach for the cpu shared memory program

- Start with serial implementation, with `serial(range)` method being changed to `cpu_shared(thread_count, range)`, which must be given a thread count. `countLineOfSight` will remain unchanged.
- Use OpenMP directives to parallelize the outermost for loop, which iterates over the rows (centerY). This allows the algorithm to take advantage of caching. In practice, this will be a `#pragma omp parallel for`. 
- Inside the for loops, when the local count for a pixel wants to be written to the array, there must be a critical section. For this, we can use `#pragma omp critical` to wrap the write operation.

## CPU Distributed Memory Algorithm

### Instructions for build and execution

- To run the CPU Distributed memory implementaiton, run `$ bash ./run_cpu_shared.sh {process_count}`, where `{process_count}` is the number of processes you want to run the program concurrently.

### Approach for the CPU distributed memory program

- Initialize MPI
- Rank 0 reads in the dataset and then calculates what work each thread will do. Each rank will do the same amount of work, except the last rank which also takes on any work that is remaining using `DATA_COUNT % comm_sz`.
- Broadcast data, work distribution, and displacements for each rank.
- Each rank calculates the line of sight for the pixels it has been assigned. The implementation for this is heavily inspired by our serial implementation.
- Use `MPI_Gatherv` to collect results from each thread to the root thread(rank 0).
- Rank 0 writes results to `cpu_distributed_out.raw`
- MPI is shut down.

## GPU Shared Memory Algorithm

### Instruction for build and execution on CHPC
- To run the GPU share memory implementation, get a gpu allocation. Make sure the CudaToolkit module is loaded with `module load cuda/12.2.0`. Then run `$ bash ./run_gpu_shared.sh`.


## Scaling Study

### Comparing Serial, Shared CPU, and Shared GPU

The execution time of the serial algorithm ranges from about 265,799 ms to 287,679 ms. 

Shown below is a table which examines the **strong scalability** of the shared cpu (OpenMP) implementation.

| Number of Cores | Execution Time     | Speedup | Efficiency |
|-----------------|--------------------|---------|------------|
| 1               | 287439 ms          |         |            |
| 2               | 147115 ms          | 1.95    |  0.977     |
| 4               | 106258 ms          | 2.71    |  0.676     |
| 8               |  67013 ms          | 4.29    |  0.536     |
| 16              |  42623 ms          | 6.74    |  0.421     |

It is clear from the table above that the algorithm is not strongly scalable because the efficiency does not remain constant as the number of cores increases. However, it should be noted that adding cores still greatly benefits the execution time, at least up to 8 cores. 

Weak scalability can be examined by increasing the range that each pixel counts line of sight for proportionally to the increase in threads. This is because the execution time of our algorithm is proportional to the range (algorithm iterates over the perimeter of a square of side length $2*range$). If the execution time remains constant, then the problem weakly scales. 

Shown below is a table which examines the **weak scalability** of the shared cpu (OpenMP) implementation. 

| Number of Cores | Range | Execution Time |
|-----------------|-------|----------------|
| 1               | 2     |  18426 ms      |
| 2               | 4     |  29895 ms      |
| 4               | 8     |  53854 ms      |
| 8               | 16    | 119472 ms      |
| 16              | 32    | 380313 ms      |

As shown above, the execution time does not stay constant as the range (and problem size by extension) is increased proportionally to the number of threads, so the algorithm does not weakly scale well. 


### Scaling Study Conclusion

- The shared cpu (OpenMP) solution does not scale strongly or weakly, but its strong scaling is better than its weak scaling. 
- The shared cpu solution is better than the serial solution, because it parallelizes the problem. 
- 