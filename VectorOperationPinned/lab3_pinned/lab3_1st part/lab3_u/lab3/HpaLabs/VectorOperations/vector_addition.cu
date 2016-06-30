#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

// GPU kernel to perform Vector Addition

__global__ void vector_additionKernel(float* ad, float* bd, float* cd, int size)
{
	// Retrive thread id within the block
	int th_id = threadIdx.x + blockIdx.x * blockDim.x;

	
		// Perform vector addition
	while(th_id<size){	
		cd[th_id] = ad[th_id] + bd[th_id];
		th_id= blockDim.x * gridDim.x;
	}
	
}

bool addVectorGPU( float* a, float* b, float* c, int size )
{
	// Error return value
	cudaError_t status;
	// Number of bytes in a vector
	int bytes = size * sizeof(float);
	float *ad, *bd, *cd;
	//Device pointer to pinned memory 
	cudaHostGetDevicePointer( (void**)&ad, a, 0 );	cudaHostGetDevicePointer( (void**)&bd, b, 0 );	cudaHostGetDevicePointer( (void**)&cd, c, 0 );

	
	// Specify the size of the grid and the size of the block
	dim3 dimBlock(1024); //  is contained in a block
	dim3 dimGrid((size+1023)/1024); // Only using a single grid element 
	// Launch the kernel on a size-by-size block of threads
	vector_additionKernel<<<dimGrid, dimBlock>>>(ad, bd, cd, size);
	cudaThreadSynchronize(); // Sync threads
	 // Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
	std::cout << "Kernel failed: " << cudaGetErrorString(status) <<	std::endl;
	return false;
	}
		
	// Success
	
	return true;
	

}

