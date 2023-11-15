#include "helper_functions.hpp"
#include <algorithm>

/*  
    Given an elevation map, a starting coordinate, and an ending coordinate, determine whether there is line of sight between the two points.
    May specify x dimension of the elevation map xDim and whether to display debug messages
*/
bool isLineOfSight(std::vector<int16_t> data, int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t xDim = 6000, bool debug = false);

/*
    Function for testing isLineOfSight on a dataset
*/
void test(bool debug=false);

int main() {   
    // test();
    std::vector<int16_t> data = getData();

    std::vector<int> counts(6000*6000,0);

    // For each pixel
    for (int16_t centerX = 0; centerX < 6000; ++centerX) {
        for (int16_t centerY = 0; centerY < 6000; ++centerY) {
            int count = 0;
            // For each pixel in a 100 pixel radius (square radius) in the image boundaries
            // std::max and std::min ensure considerX and considerY are in the bounds of image
            for (int16_t considerX = std::max(centerX - 100,0); considerX <= std::min(centerX + 100,5999); ++considerX) { 
                for (int16_t considerY = std::max(centerY - 100,0); considerY <= std::min(centerY + 100,5999); ++considerY) {
                    bool isSight = isLineOfSight(data,centerX,centerY,considerX,considerY);
                    if (isSight) {
                        count++;
                    }
                }
            }
            counts[getIndex(centerX,centerY)] = count;
        }
    }

    writeFile(counts,"output.raw");
    std::cout << "Done!" << std::endl;
    std::cout << "Visible pixels from (0,0): " << counts[getIndex(0,0)] << std::endl; 

    return 0;
}

bool isLineOfSight(std::vector<int16_t> data,int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t xDim, bool debug)
{
	// Compute the differences between start and end points
    
	int dx = x2 - x1;
	int dy = y2 - y1;
    float z2 = static_cast<float>(data[getIndex(x2,y2,xDim)]);
    float z1 = static_cast<float>(data[getIndex(x1,y1,xDim)]);
    float dz = z2 - z1;
    float SLOPE = dz / std::sqrt(std::pow(dx,2)+std::pow(dy,2));


	// Absolute values of the change in x and y
	const int abs_dx = abs(dx);
	const int abs_dy = abs(dy);

	// Initial point
	int x = x1;
	int y = y1;
    float z = z1;
    float zLineOfSight;

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
            
            zLineOfSight = SLOPE * std::sqrt(std::pow((x-x1),2)+std::pow((y-y1),2)) + z1;
            z = static_cast<float>(data[getIndex(x,y,xDim)]);

            // Print debug messages
            if (debug) {
                std::cout << "(" << x << "," << y << "," << z << ")" << std::endl; // TODELETE
                std::cout << "zLineOfSight = " << zLineOfSight << std::endl;
                std::cout << "z = " << z << std::endl;
                std::cout << "\n";
            }
			

            if (z > zLineOfSight) {
                if (debug) {
                    std::cout << "Returning false" << std::endl;
                }
                return false;
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
            zLineOfSight = SLOPE * std::sqrt(std::pow((x-x1),2)+std::pow((y-y1),2)) + z1;
            z = static_cast<float>(data[getIndex(x,y,xDim)]);
            
            // Print debug messages
            if (debug) {
                std::cout << "(" << x << "," << y << "," << z << ")" << std::endl; // TODELETE
                std::cout << "zLineOfSight = " << zLineOfSight << std::endl;
                std::cout << "z = " << z << std::endl;
                std::cout << "\n";
            }
			

            if (z > zLineOfSight) {
                if (debug) {
                    std::cout << "Returning false" << std::endl;
                }
                return false;
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
    return true;
}


void test(bool debug) {
    std::vector<int16_t> testData{0, 0, 0, 0,
                                  1, 1, 1, 1,
                                  2, 2, 3, 2,
                                  3, 3, 3, 3};
    bool sight;
    for (int x1 = 1; x1 < 4; ++x1) {
        for (int y1 = 1; y1 < 4; ++y1) {
            for (int x2 = 1; x2 < 4; ++x2) {
                for (int y2 = 1; y2 < 4; ++y2) {
                    std::cout << "Testing from (" << x1 << "," << y1 << ") to (" << x2 << "," << y2 << "): ";
                    sight = isLineOfSight(testData,x1,y1,x2,y2,4,debug);
                    if (sight) std::cout << "True" << std::endl;
                    else std::cout << "False" << std::endl;
                }
            }
        }
    }

    // auto sight = isLineOfSight(testData,1,1,3,3,4,true) ? "true" : "false";
    // std::cout <<sight<<std::endl;


}
