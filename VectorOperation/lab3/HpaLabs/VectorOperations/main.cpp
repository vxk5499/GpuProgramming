#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include "common.h"
#include <time.h>
const int size = 2;
const int iterations=1000;
/* Entry point for the program. Allocates space for three vectors,
calls a function to add them, subracting them and scale a vector, and displays the results. */
int main()
{
// Timing data
	
	
	float tcpu, tgpu;
	clock_t start, end;
// The element i,j is represented by index (i*SIZE + j)
	srand (time(NULL));
	float* a = new float[size ];
	float* b = new float[size ];
	float* c = new float[size ];
	float* cgpu = new float[size ];
	float scaleFactor= (float)(rand() % 100);

// Initialize M and N to random integers
	for (int i = 0; i < size; i++) {
		a[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		b[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

//Display the operating vector length

std::cout << "Operating on a vector length " << size << std::endl;

// Performs vector addition in CPU and compute CPU computation time
	start = clock();
	for (int i = 0; i < iterations; i++) {
	addVectorCPU(a, b, c, size);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "CPU Addition took " << tcpu << " ms" << std::endl;
// Checks for kernel error	
	bool success = addVectorGPU(a, b, cgpu, size);
	if (!success){
		std::cout << "\n * Device error! Add *\n" << std::endl;
		return 1;
	}

// Performs vector addition in GPU and compute GPU computation time
	start = clock();
	for (int i = 0; i < iterations; i++) {
		addVectorGPU(a, b, cgpu, size);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;	std::cout << "GPU Addition took " << tgpu << " ms" << std::endl;/*Computes the GPU speedup w.r.t GPU and error between the GPU and CPU results */	std::cout << "Addition Speedup = " << tcpu/tgpu <<  std::endl;	float sum = 0, delta = 0;
	for (int i = 0; i < size; i++) {
		delta += (c[i] - cgpu[i]) * (c[i] - cgpu[i]);
		sum += (c[i] * cgpu[i]);
	}
	float L2norm = sqrt(delta / sum);
	std::cout << "Addition error = " << L2norm << "\n" <<
	((L2norm < 1e-6) ? " " : "Failed") << std::endl;	
// Performs vector subtraction in CPU and Compute the CPU computation time
	start = clock();
	for (int i = 0; i < iterations; i++) {
	subtractVectorCPU(a, b, c, size);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "CPU Subraction took " << tcpu << " ms" << std::endl;

// Checks for Kernel error
	success = subtractVectorGPU(a, b, cgpu, size);
	if (!success){
		std::cout << "\n * Device error! Sub*\n" << std::endl;
		return 1;
	}
	
// Performs vector subtraction in GPU and Compute the GPU computation time	
	start = clock();
	for (int i = 0; i < iterations; i++) {
		subtractVectorGPU(a, b, cgpu, size);
		}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;	std::cout << "GPU Subraction took " << tgpu << " ms" << std::endl;

	std::cout << "Subraction Speedup = " << tcpu/tgpu <<  std::endl;
	sum = 0, delta = 0;
	for (int i = 0; i < size; i++) {
		delta += (c[i] - cgpu[i]) * (c[i] - cgpu[i]);
		sum += (c[i] * cgpu[i]);
	}
	L2norm = sqrt(delta / sum);
	std::cout << "Subraction Error = " << L2norm << "\n" <<
	((L2norm < 1e-6) ? "" : "Failed") << std::endl;
	
// Performs vector scaling in CPU and Compute the CPU computation time	
	start = clock();
	for (int i = 0; i < iterations; i++) {
	scaleVectorCPU(a, c, scaleFactor, size);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "CPU Scaling took " << tcpu << " ms" << std::endl;

// Checks for Kernel Error		
	success = scaleVectorGPU(a, cgpu,  scaleFactor, size);
	if (!success){
		std::cout << "\n * Device error! Scale*\n" << std::endl;
		return 1;
	}

// Performs vector scaling in GPU and Compute the GPU computation time
	start = clock();
	for (int i = 0; i < iterations; i++) {
		scaleVectorGPU(a, cgpu,  scaleFactor, size);
			}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "GPU scaling took " << tgpu << " ms" << std::endl;

//Calculates CPU to GPU speedup and checks for error between CPU and GPU results
	std::cout << "Scale Speedup = " << tcpu/tgpu <<  std::endl;
	sum = 0, delta = 0;
	for (int i = 0; i < size; i++) {
		delta += (c[i] - cgpu[i]) * (c[i] - cgpu[i]);
		sum += (c[i] * cgpu[i]);
	}
	L2norm = sqrt(delta / sum);
	std::cout << "Scaling error = " << L2norm << "\n" <<
	((L2norm < 1e-6) ? " " : "Failed") << std::endl;

// Release the vectors
	delete[] a; delete[] b; delete[] c, delete[] cgpu;
	
	return 0;
}