#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"
#include <fstream> 

const int Tile_width = 16;


//GPU Kernel to perform Scaling operation in Gauss Jordan Elimination.

__global__ void GaussianEliminationScalingKernel(float* matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float* outputMatrix,float* tempMatrix , bool partialPivot, int iterNo){

	// Calculates the block id and thread id
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	// Calculates the row and column indices of matrices
	int row = blockIdx.y * Tile_width + threadIdx.y;
	int col = blockIdx.x * Tile_width + threadIdx.x;
	
	if(col<numberOfColumns){
		if((row == iterNo)){
			float scale = matrix[row * numberOfColumns + iterNo];
			outputMatrix[row * numberOfColumns + col] = matrix[row * numberOfColumns + col]/scale;
		}
	}
	
}

// GPU Kernel to perform reduction operation in Guass Jordan Elimination.
__global__ void GaussianEliminationReductionKernel(float* matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float* outputMatrix,float* tempMatrix , bool partialPivot,int iterNo)
{
	

	// Calculates the block id and thread id
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	// Calculate the row and column indices of matrices
	int row = blockIdx.y * Tile_width + threadIdx.y;
	int col = blockIdx.x * Tile_width + threadIdx.x;
	if(col<numberOfColumns){
	if((row != iterNo && row< numberOfRows && row>=0) ){
			float temp = matrix[row * numberOfColumns + iterNo];
			
			float temp1 = matrix[( iterNo * numberOfColumns) + col];
			outputMatrix[row * numberOfColumns + col] = matrix[row * numberOfColumns + col] - temp * temp1;
			
		}
	}

	
	
}

bool GaussianEliminationGPU( float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot ){
	
	// Error return value
	cudaError_t status;

	//size of matrix
	int size = numberOfRows * numberOfColumns;

	// Number of bytes in a vector
	int bytes = size * sizeof(float);

	//Pointer to device arrays
	float* matrixid = new float[numberOfRows * numberOfColumns] ;
	float* outputMatrixid = new float[numberOfRows * numberOfColumns];
	int inc=0;
	int row=0;

	for( int i = 0; i < numberOfRows; i++){
		for( int j = 0; j < numberOfColumns; j++){
			matrixid[inc] = matrix[i][j];
			outputMatrixid[inc] = matrix[i][j];
			inc += 1;
		}
	}
	

	float* matrixd;
	float* outputMatrixd;
	float* tempMatrix;



	// Allocate memory on the device to store matrix
	cudaMalloc((void**) &matrixd, bytes);
	cudaMalloc((void**) &outputMatrixd, bytes);
	cudaMalloc((void**) &tempMatrix, bytes);

	// Copy the host input data to the device
	cudaMemcpy(matrixd, matrixid, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(outputMatrixd, outputMatrixid, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(tempMatrix, outputMatrixid, bytes, cudaMemcpyHostToDevice);
	
	// Specify the size of the grid and the size of the block
	dim3 dimBlock(Tile_width,Tile_width); //  is contained in a block
	dim3 dimGrid((int)ceil((float)numberOfRows+1 / (float)Tile_width), (int)ceil((float)numberOfColumns / (float)Tile_width));


	// Gauss Jordan Elimination operation
	for(row=0;row<numberOfRows;row++){
	GaussianEliminationScalingKernel<<<dimGrid, dimBlock>>>(matrixd, numberOfRows, numberOfColumns, outputMatrixd, tempMatrix, partialPivot,row);
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Kernel failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(matrix);
			cudaFree(outputMatrix);
			return false;
	     
		

	}
	cudaThreadSynchronize();

	// Copy the device output data to the device input data
	cudaMemcpy(matrixd, outputMatrixd, bytes, cudaMemcpyDeviceToDevice);
	
	
	GaussianEliminationReductionKernel<<<dimGrid, dimBlock>>>(matrixd, numberOfRows, numberOfColumns, outputMatrixd, tempMatrix, partialPivot,row);
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Kernel failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(matrix);
			cudaFree(outputMatrix);
			return false;
	     
	
	}

	cudaThreadSynchronize();

	// Copy the device output data to the device input data
	cudaMemcpy(matrixd, outputMatrixd, bytes, cudaMemcpyDeviceToDevice);
			
	}

	// Copy the device output data to the host
	cudaMemcpy(outputMatrixid, outputMatrixd, bytes, cudaMemcpyDeviceToHost);
	int inc1=0;
	for( int i = 0; i < numberOfRows; i++){
		for( int j = 0; j < numberOfColumns; j++){
			matrixid[inc] = matrix[i][j];
			outputMatrix[i][j] = outputMatrixid[inc1] ;
			inc1 += 1;
		}
	}

	// Free Device memory
	cudaFree(matrixd);
	cudaFree(outputMatrixd);
	cudaFree(tempMatrix);
	
	return true;

}