#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include "common.h"
#include <time.h>
#include <fstream> 

const int numberOfRows= 512;
const int numberOfColumns= 513;
const int iterations=5;



int main(){

	// Timing data
	float tcpu, tgpu;
	clock_t start, end;

	srand (time(NULL));

	std::cout<<"Operating on a "<<numberOfRows<<"x"<<numberOfRows<<" matrix"<<std::endl;

	// Allocation of input matrix with respect to the number of rows and columns
	float** matrix = new float*[numberOfRows];
	for(int i=0; i< numberOfRows; ++i)
		matrix[i]= new float[numberOfColumns];

	// Allocation of output matrix of CPU implementation with respect to the number of rows and columns
	float** outputMatrix = new float*[numberOfRows];
	for(int i=0; i< numberOfRows; ++i)
		outputMatrix[i]= new float[numberOfColumns];

	// Allocation of output matrix of GPU implementation with respect to the number of rows and columns
	float** outputMatrix1 = new float*[numberOfRows];
	for(int i=0; i< numberOfRows; ++i)
		outputMatrix1[i]= new float[numberOfColumns];

	// partialPivot variable initialization
	bool partialPivot= false;

	// Initialize the matrix to random float values

	for( int i = 0; i < numberOfRows; i++){
		for( int j = 0; j < numberOfColumns; j++){
			matrix[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			outputMatrix[i][j]=0;
			outputMatrix1[i][j]=0;
		}
	}

	
	// CPU implemenation of Guass Jordan Implementation.
	start = clock();
	for (int i = 0; i < iterations; i++) {
	GaussianEliminationCPU(  matrix,  numberOfRows,  numberOfColumns, outputMatrix,  partialPivot );
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result took " << tcpu << " ms" << std::endl; // Time taken by CPU for computation

	
	// Warm up of GPU	
	bool success = true;
	success = GaussianEliminationGPU(matrix, numberOfRows, numberOfColumns, outputMatrix1, partialPivot );
	if (!success){
		std::cout << "\n * Device error! Add *\n" << std::endl;
		return 1;
	}

	// GPU implementation of Gauss Jordan Implementation.
	start = clock();
	for (int i = 0; i < iterations; i++) {
	GaussianEliminationGPU(matrix, numberOfRows, numberOfColumns, outputMatrix1, partialPivot );
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;	std::cout << "Device result took " << tgpu << " ms" << std::endl; // Time taken by GPU for computation

	std::cout << "Gauss Jordan Speedup: " << tcpu/tgpu <<  std::endl; // Gives the ratio of CPU computation time to the GPU computation time
	float sum = 0, delta = 0;
	for (int i = 0; i < numberOfRows; i++) {
		int j = numberOfColumns -1;
		delta += (outputMatrix[i][j] - outputMatrix1[i][j]) * (outputMatrix[i][j] - outputMatrix1[i][j]);
		sum += (outputMatrix[i][j] * outputMatrix1[i][j]);
	}
	float L2norm = sqrt(delta / sum);
	std::cout << "Error: " << L2norm << "\n"; // Error that compares CPU and GPU results

	// Free allocated memory
	delete[] matrix; delete[] outputMatrix;delete[] outputMatrix1;

return 1;


}
