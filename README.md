# HPC Project: AwannaCU

## Serial algorithm

### Instructions for build and execution

- In the directory `src/serial/`, there is a file called `CMakeLists.txt`. This allows us to build the program using CMake. Ensure that you have cmake installed on your machine to build the program.
- Navigate to the top directory of the project
- Make a build folder for the serial program by running the following command:

`$ mkdir build && mkdir build/serial`

- Create the build system by running the following command

`$ cmake -B build/serial -S src/serial`

- Once the build system is created, navigate to the build directory by running `$ cd build/serial`

- To build the program, run `$ make`
- To run the program, run `$ ./serial`
- Output file will be `output.raw`

### Approach for the serial program

Consider two points 1 (x1,y1,z1) and 2 (x2,y2,z2). point 2 is in the line of sight of point 1 if and only if for every point (x,y,zLine) along the line from (x1,y1,z1) to (x2,y2,z2) z <= zLine, where z is the elevation map value at point (x,y). From this, we can modify Bresenham's algorithm to check z against zLine (called zLineOfSight in serial.cpp) at all points (x,y) in a straight line between (x1,y1) and (x2,y2). Define the following function

- `bool isLineOfSight(data, x1, y1, x2, y2)`
    - This function computes a straight line between (x1,y1) and (x2,y2) and then along that line checks z against zLineOfSight. If at every point (x,y) on the line from (x1,y1) to (x2,y2) z <= zLineOfSight, then the function returns true. Otherwise, false is returned.

In `main()`, the program will read in the data from the file (a helper function to do this is written in `helper_functions.cpp`), then for each pixel (centerX, centerY) in the image, compute `isLineOfSight(data,centerX,centerY,considerX,considerY)`, where (considerX,considerY) iterates over every pixel in a 100 pixel square radius. A count is iterated for every time `isLineOfSight` returns true, and then the count is stored in an array. After the count has been computed for every pixel, the array holding the counts is written to a file `output.raw`


