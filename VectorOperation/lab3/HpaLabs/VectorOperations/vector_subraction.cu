#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

// GPU kernel to perform Vector Addition

__global__ void vector_subractionKernel(float* ad, float* bd, float* cd, int size)
{
	// Retrive thread id within the block
	int th_id = threadIdx.x + blockIdx.x * blockDim.x;

	
		// Perform vector subraction
	while(th_id<size)	{
		cd[th_id] = ad[th_id] - bd[th_id];
		th_id= blockDim.x * gridDim.x;
	}
}

bool subtractVectorGPU( float* a, float* b, float* c, int size )
{
	// Error return value
	cudaError_t status;
	// Number of bytes in a vector
	int bytes = size * sizeof(float);
	// Pointer to the device arrays
	float *ad, *bd, *cd;
	// Allocate memory on the device to store each vector
	cudaMalloc((void**) &ad, bytes);
	cudaMalloc((void**) &bd, bytes);
	cudaMalloc((void**) &cd, bytes);
	// Copy the host input data to the device
	cudaMemcpy(ad, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, bytes, cudaMemcpyHostToDevice);
	// Specify the size of the grid and the size of the block
	dim3 dimBlock(1024); //  is contained in a block
	dim3 dimGrid((size+1023)/1024); // Only using a single grid element
	// Launch the kernel on a size-by-size block of threads
	 vector_subractionKernel<<<dimGrid, dimBlock>>>(ad, bd, cd, size);
	 // Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
	std::cout << "Kernel failed: " << cudaGetErrorString(status) <<	std::endl;
	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);
	return false;
	}
	// Retrieve the result vector
	cudaMemcpy(c, cd, bytes, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);
	// Success
	
	return true;
	

}

