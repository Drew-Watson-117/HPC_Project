// CS 5030 Project - Serial Bresenham implementation
#include <cmath>
#include <iostream>
#include <chrono>

// Bresenham's Line Algorithm: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

// Traces a line between points (x1, y1) and (x2, y2) and prints the intermediate coordinates
void plot_line(int x1, int y1, int x2, int y2)
{
	// Compute the differences between start and end points
	int dx = x2 - x1;
	int dy = y2 - y1;

	// Absolute values of the change in x and y
	const int abs_dx = abs(dx);
	const int abs_dy = abs(dy);

	// Initial point
	int x = x1;
	int y = y1;

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
			// Print the current coordinate
			// std::cout << "(" << x << "," << y << ")" << std::endl;

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
			// Print the current coordinate
			// std::cout << "(" << x << "," << y << ")" << std::endl;

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
}

// Example of how to use plot_line, I show that the algorithm will work from any starting point to any stopping point, regardless of direction
int main()
{
	// Starting: (0, 0)
	int x1 = 0;
	int y1 = 0;

	// Stopping: (7, 6)
	int x2 = 100;
	int y2 = 0;

	// Up and to the right
	std::cout << "up-right: Starting at (" << x1 << ", " << y1 << ") and stopping at (" << x2 << ", " << y2 << ")" << std::endl;
	plot_line(x1, y1, x2, y2);

	// Up and to the left
	std::cout << "\nup-left: Starting at (" << x1 << ", " << y1 << ") and stopping at (" << -x2 << ", " << y2 << ")" << std::endl;
	plot_line(x1, y1, -x2, y2);

	// Down and to the left
	std::cout << "\ndown-left: Starting at (" << x1 << ", " << y1 << ") and stopping at (" << -x2 << ", " << -y2 << ")" << std::endl;
	plot_line(x1, y1, -x2, -y2);

	// Down and to the left
	std::cout << "\ndown-right: Starting at (" << x1 << ", " << y1 << ") and stopping at (" << x2 << ", " << -y2 << ")" << std::endl;
	plot_line(x1, y1, x2, -y2);

	std::cout << "Timing Bresenham" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; i++) plot_line(x1,y1,x2,y2);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << duration.count() / 100 << " us" << std::endl;

	return 0;
}