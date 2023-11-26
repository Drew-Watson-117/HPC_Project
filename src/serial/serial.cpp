#include "helper_functions.hpp"
#include <algorithm>
#include <chrono>

/*  
    Given an elevation map, a starting coordinate, and an ending coordinate, count the number of pixels the two points which have line of sight with the starting point.
    May specify x dimension of the elevation map xDim and whether to display debug messages
*/
int countLineOfSight(std::vector<int16_t> &data, int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t xDim = 6000, bool debug = false);

/*
    Functions for testing countLineOfSight on a dataset
*/
void test(bool debug=false);
// Tests an iteration of the algorithm with a specified range (radius to consider)
void timing(int16_t range=100);

/*
	Function which contains the logic for the serial implementation of the algorithm
*/
void serial(int16_t range=100);

int main() {   
    // test();
	// timing(100);
    serial(10);

    return 0;
}

void serial(int16_t range) {
	std::vector<int16_t> data = getData();

    std::vector<int> counts(6000*6000,0);

    // For each pixel
    for (int16_t centerY = 0; centerY < 6000; ++centerY) {
        for (int16_t centerX = 0; centerX < 6000; ++centerX) {
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
            counts[getIndex(centerX,centerY)] = count;
        }
    }

    bool success = writeFile(counts,"output_serial.raw");
    success ? std::cout << "Done!" << std::endl : std::cout << "Error Writing to File" << std::endl;
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

void test(bool debug) {
    std::vector<int16_t> testData{0, 0, 0, 0,
                                  1, 1, 1, 1,
                                  2, 2, 3, 2,
                                  3, 3, 3, 3};
    int count;

	int x1 = 0, y1 = 0;
	int x2 = 0, y2 = 0;
	std::cout << "Testing from (" << x1 << "," << y1 << ") to (" << x2 << "," << y2 << "): ";
    count = countLineOfSight(testData,x1,y1,x2,y2,4,debug);
    std::cout << count << std::endl;
	x2 = 0, y2 = 3;
	std::cout << "Testing from (" << x1 << "," << y1 << ") to (" << x2 << "," << y2 << "): ";
    count = countLineOfSight(testData,x1,y1,x2,y2,4,debug);
    std::cout << count << std::endl;
	x2 = 3, y2 = 0;
	std::cout << "Testing from (" << x1 << "," << y1 << ") to (" << x2 << "," << y2 << "): ";
    count = countLineOfSight(testData,x1,y1,x2,y2,4,debug);
    std::cout << count << std::endl;
	x2 = 3, y2 = 3;
	std::cout << "Testing from (" << x1 << "," << y1 << ") to (" << x2 << "," << y2 << "): ";
    count = countLineOfSight(testData,x1,y1,x2,y2,4,debug);
    std::cout << count << std::endl;

}

void timing(int16_t range) {
	std::vector<int16_t> data = getData();
	int count;
	std::cout << "Timing double implementation" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; i++) count = countLineOfSight(data,200,200,200,200-range);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << duration.count() / 100 << " us" << std::endl;
}