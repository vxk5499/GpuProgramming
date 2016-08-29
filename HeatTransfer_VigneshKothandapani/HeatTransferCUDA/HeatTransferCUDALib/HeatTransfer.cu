/************************************************************************/
// The purpose of this file is to provide a GPU implementation of the 
// heat transfer simulation using MATLAB.
//
// Author: Jason Lowden
// Date: October 20, 2013
//
// File: KMeans.h
/************************************************************************/
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include <cuda_texture_types.h>
#include <iostream>

#include "HeatTransfer.h"

texture<float,2> texIn; // Input to texture memory

__global__ void UpdateHeatMapKernel(float * texOut,int size, float heatSpeed)
{
	
	int col = threadIdx.x + blockIdx.x * blockDim.x; // Calculates the current column
	int row = threadIdx.y + blockIdx.y * blockDim.y; // Calculate the current row
	int offset = col + row * size; // indicates the cureent operating element
	if(col > 0 && col < size-1 && row < size-1 && row > 0){
		
	
	float top = tex2D(texIn, col, row-1); // element on top of current element
	float left = tex2D(texIn, col-1, row);  // element on left of current element
	float right = tex2D(texIn, col+1, row); // element on right of current element
	float bottom = tex2D(texIn, col, row+1);// element on bottom of current element
	float current = tex2D(texIn, col, row); // Current element
	float temp =  heatSpeed * ( top + bottom + right + left - (4 * current)); // heat transfeered from other elements
	
	texOut[offset] = current + temp; // New heat

		
	}
}

// Calculates the updated heat map for a given size based on number of iterations
bool UpdateHeatMap(float* dataIn, float* dataOut, int size, float heatSpeed, int numIterations)
{
	cudaError_t status; // to check success of cuda calls

	int bytes = size * size * sizeof(float); // size of input data

	cudaArray_t dataIn_d; // Input data array

	float* texOut; //Output from texture memory

	//Allocation of device data
	cudaMalloc((void**)&texOut, bytes);
			status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Alloc failed: " << cudaGetErrorString(status) <<	std::endl;
			
			cudaFree(texOut);
			return false;     
		}

	//Copying data to device memory
	cudaMemcpy(texOut, dataIn, bytes, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
			std::cout << "Memcopy failed failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(texOut);
			return false;     
		}

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	status = cudaGetLastError();
	if (status != cudaSuccess) {
			std::cout << "Desc failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(texOut);
			return false;     
		}
	unsigned int flags=0;
	// Allocate array in device
	cudaMallocArray(&dataIn_d, &desc,size, size);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
			std::cout << "Array alloc failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(texOut);
			return false;     
		}
	int size1= size * size *sizeof(float);
	//Copy data into array
	cudaMemcpyToArray (dataIn_d, 0, 0, dataIn, size1, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
			std::cout << "memcpy to array failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(texOut);
			return false;     
		}

	//Bind array to texture
	cudaBindTextureToArray (&texIn, dataIn_d, &desc);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
			std::cout << "Cuda Binding failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(texOut);
			return false;     
		}

	dim3 dimBlock(16,16); //  Dimension of block
	dim3 dimGrid((int)ceil((float)size / (float)16), (int)ceil((float)size / (float)16)); // Dynamic allocation for dimension of grid


	for(int i = 0; i < numIterations; i++)
	{
		UpdateHeatMapKernel<<<dimGrid, dimBlock>>>(texOut, size, heatSpeed); // Calls heat map Kernel
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Cuda kernal failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(texOut);
			return false;     
		}
		cudaThreadSynchronize(); // Cuda Synchronisation 
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Sync failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(texOut);
			return false;     
		}
		cudaUnbindTexture (&texIn); // Unbind texture memory
			status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Unbind failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(texOut);
			return false;     
		}
		cudaMemcpyToArray (dataIn_d, 0, 0, texOut, size1, cudaMemcpyDeviceToDevice); // Cuda memcpy to array within device
			status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "memcpy to array failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(texOut);
			return false;     
		}
		cudaBindTextureToArray (&texIn, dataIn_d, &desc); // Bind array to texture memory
			status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Bind failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(texOut);
			return false;     
		}

	}

	cudaMemcpy(dataOut, texOut, bytes, cudaMemcpyDeviceToHost); // Copy results to host
	cudaUnbindTexture (&texIn); // Unbind texture memory
	cudaFree(texOut); // Free cuda memory
	cudaFreeArray(dataIn_d); // Free cuda memory
	





	return true;
}