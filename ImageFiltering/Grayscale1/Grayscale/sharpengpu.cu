#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"
#include <fstream> 
#include <texture_fetch_functions.h>
#include <cuda_texture_types.h>
#include "bitmap.h"
const int Tile_width = 16;

__constant__ double filter_d1[9]; //Constant memory variable

texture<unsigned char,2,cudaReadModeElementType> texIn; // Input to texture memory

__global__ void SharpenGPUKernel(unsigned char* imaged, unsigned char* outputImaged,int width,int height,double* filter){

	int row = blockIdx.y * Tile_width + threadIdx.y;
	int col = blockIdx.x * Tile_width + threadIdx.x;

	if(row < height && col < width){
			//Perform Image convolution 
		double accum = 0;
			for(int fw = 0 ; fw < 3; fw++)
				for(int fh = 0; fh < 3; fh++)
				{
					int ix = ( col - 1 + fw + width)% width;
					int iy = ( row - 1 + fh + height)%height;
					accum = accum + (imaged[iy * width + ix] * filter[fw*3 + fh]);
				}
			unsigned char temp = accum;
			outputImaged[row * width + col] = temp;
	}
}

__global__ void SharpenGPUKernel1(unsigned char* imaged, unsigned char* outputImaged,int width,int height){

	int row = blockIdx.y * Tile_width + threadIdx.y;
	int col = blockIdx.x * Tile_width + threadIdx.x;

	if(row < height && col < width){
			//Perform Image convolution 
		double accum = 0;
			for(int fw = 0 ; fw < 3; fw++)
				for(int fh = 0; fh < 3; fh++)
				{
					int ix = ( col - 1 + fw + width)% width;
					int iy = ( row - 1 + fh + height)%height;
					accum = accum + (imaged[iy * width + ix] * filter_d1[fw*3 + fh]);
				}
			unsigned char temp = accum;
			outputImaged[row * width + col] = temp;
	}
}

__global__ void SharpenGPUKernel2( unsigned char* outputImaged,int width,int height,double* filter){

	int row = blockIdx.y * Tile_width + threadIdx.y;
	int col = blockIdx.x * Tile_width + threadIdx.x;

	if(row < height && col < width){
			//Perform Image convolution 
		double accum = 0;
			for(int fw = 0 ; fw < 3; fw++)
				for(int fh = 0; fh < 3; fh++)
				{
					int ix = ( col - 1 + fw + width)% width;
					int iy = ( row - 1 + fh + height)%height;
					accum = accum + (tex2D(texIn,ix,iy) * filter[fw*3 + fh]);
				}

			unsigned char temp = accum;
			outputImaged[row * width + col] = temp;
	}
}

bool  SharpenGPU( Bitmap* image, Bitmap* outputImage, int choice){
	
	// Error return value
	cudaError_t status;

	

	cudaArray* carray;// Input data array
	cudaChannelFormatDesc channel; //create channel to describe data type
	channel = cudaCreateChannelDesc<unsigned char>(); 
	
	//size of matrix
	int size = image->Height() * image->Width();
	int bytes = size * sizeof(unsigned char);
	int bytes1 = 9 * sizeof(double);

	//Device pointers
	unsigned char *image_d;
	unsigned char *Outputimage_d;
	double *filter_d;

	// The width and height of the input image
	int width = image->Width();
	int height = image->Height();

	cudaMallocArray(&carray,&channel,width,height);

	// Sharpen filter
	double filter[9] =
	{
		0, -1, 0,
	    -1, 5,-1,
        0, -1, 0
	};
	

	//Allocation of device variables
	cudaMalloc((void**)&image_d,bytes);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Alloc failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			return false;
	 	}

	cudaMalloc((void**)&Outputimage_d,bytes);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Alloc failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			return false;
	     
	
	}

	cudaMalloc((void**)&filter_d,bytes1);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Alloc failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			return false;
	     
	
	}

	cudaMemcpyToArray(carray,0,0,image->image,bytes,cudaMemcpyHostToDevice); // Copy required data to Array

	//Set Texture address mode property
	texIn.addressMode[0]=cudaAddressModeWrap;
	texIn.addressMode[1]=cudaAddressModeClamp;
	cudaBindTextureToArray(texIn,carray);

	// Copies the required input from Host to device
	cudaMemcpy(image_d, image->image, bytes, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Copy of image failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			cudaFree(filter_d);
			return false;
	}

	cudaMemcpy(filter_d,filter, bytes1, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Copy of filter failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			cudaFree(filter_d);
			return false;
	}

	cudaMemcpyToSymbol(filter_d1, filter, bytes1, 0, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Copy of constant filter failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			cudaFree(filter_d);
			return false;
	}

	dim3 dimBlock(Tile_width,Tile_width); //  Dimension of block
	dim3 dimGrid((int)ceil((float)width / (float)Tile_width), (int)ceil((float)height / (float)Tile_width)); // Dynamic allocation for dimension of grid
	if (choice == 1)
	SharpenGPUKernel<<<dimGrid, dimBlock>>>(image_d,Outputimage_d,width,height,filter_d); // Kernel call
	else if(choice ==2)
	SharpenGPUKernel1<<<dimGrid, dimBlock>>>(image_d,Outputimage_d,width,height);
	else if(choice == 3)
	SharpenGPUKernel2<<<dimGrid, dimBlock>>>(Outputimage_d,width,height,filter_d);

	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Kernel failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			cudaFree(filter_d);
			return false;
	     
	
	}

	cudaThreadSynchronize(); // Cuda synchronize

	cudaUnbindTexture (&texIn); // Unbind texture memory
			status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Unbind failed: " << cudaGetErrorString(status) <<	std::endl;
			
			return false;     
		}

	cudaMemcpy(outputImage->image, Outputimage_d, bytes, cudaMemcpyDeviceToHost); // Copies the output form host to device.
	//Freeing allocated memory.

	cudaFree(image_d);
	cudaFree(Outputimage_d);
	cudaFree(filter_d);

	return true; 
	
	}



