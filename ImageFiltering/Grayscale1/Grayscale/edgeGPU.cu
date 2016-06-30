#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"
#include <fstream> 
#include "bitmap.h"
#include <math.h> 
const int Tile_width = 16;

__constant__ double filter_d1[9];//Constant memory variable
__constant__ double filter_d2[9];//Constant memory variable


texture<unsigned char,2,cudaReadModeElementType> texIn; // Input to texture memory

__global__ void EdgeSobelGPUKernel(unsigned char* imaged, unsigned char* outputImaged,int width,int height,double* filter1, double* filter2){

	int row = blockIdx.y * Tile_width + threadIdx.y;
	int col = blockIdx.x * Tile_width + threadIdx.x;

	if(row < height && col < width){
			//Perform Image convolution 
		double accum = 0,accum1 = 0,accum2 = 0;
			for(int fw = 0 ; fw < 3; fw++)
				for(int fh = 0; fh < 3; fh++)
				{
					int ix = ( col - 1 + fw + width)% width;
					int iy = ( row - 1 + fh + height)%height;
					accum1 = accum1 + (imaged[iy * width + ix] * filter1[fw*3 + fh]);
					accum2 = accum2 + (imaged[iy * width + ix] * filter2[fw*3 + fh]);

				}
			accum= sqrt(pow(accum1,2)+pow(accum2,2));
			unsigned char temp = accum;
			outputImaged[row * width + col] = temp;
	}
}

__global__ void EdgeSobelGPUKernel1(unsigned char* imaged, unsigned char* outputImaged,int width,int height){

	int row = blockIdx.y * Tile_width + threadIdx.y;
	int col = blockIdx.x * Tile_width + threadIdx.x;

	if(row < height && col < width){
			//Perform Image convolution 
		double accum = 0,accum1 = 0,accum2 = 0;
			for(int fw = 0 ; fw < 3; fw++)
				for(int fh = 0; fh < 3; fh++)
				{
					int ix = ( col - 1 + fw + width)% width;
					int iy = ( row - 1 + fh + height)%height;
					accum1 = accum1 + (imaged[iy * width + ix] * filter_d1[fw*3 + fh]);
					accum2 = accum2 + (imaged[iy * width + ix] * filter_d2[fw*3 + fh]);

				}
			accum= sqrt(pow(accum1,2)+pow(accum2,2));
			unsigned char temp = accum;
			outputImaged[row * width + col] = temp;
	}
}

__global__ void EdgeSobelGPUKernel3(unsigned char* outputImaged,int width,int height,double* filter1, double* filter2){

	int row = blockIdx.y * Tile_width + threadIdx.y;
	int col = blockIdx.x * Tile_width + threadIdx.x;

	if(row < height && col < width){
			//Perform Image convolution 
		double accum = 0,accum1 = 0,accum2 = 0;
			for(int fw = 0 ; fw < 3; fw++)
				for(int fh = 0; fh < 3; fh++)
				{
					int ix = ( col - 1 + fw + width)% width;
					int iy = ( row - 1 + fh + height)%height;
					accum1 = accum1 + (tex2D(texIn,ix,iy) * filter1[fw*3 + fh]);
					accum2 = accum2 + (tex2D(texIn,ix,iy) * filter2[fw*3 + fh]);

				}
			accum= sqrt(pow(accum1,2)+pow(accum2,2));
			unsigned char temp = accum;
			outputImaged[row * width + col] = temp;
	}
}

bool  EdgeSobelGPU( Bitmap* image, Bitmap* outputImage, int choice ){
	
	// Error return value
	cudaError_t status;

	//size of matrix
	int size = image->Height() * image->Width();
	int bytes = size * sizeof(char);
	int bytes1 = 9 * sizeof(double);

	//Device pointers
	unsigned char *image_d;
	unsigned char *Outputimage_d;
	double *filter1_d,*filter2_d;

	// The width and height of the input image
	int width = image->Width();
	int height = image->Height();

	cudaArray* carray;// Input data array
	cudaChannelFormatDesc channel; //create channel to describe data type
	channel = cudaCreateChannelDesc<unsigned char>(); 

	// Sobel filters
		double filter1[3][3] =
	{
	 -1,  0,  1,
	 -2,  0,  2,
	 -1,  0,  1,
	};

		double filter2[3][3] =
	{
	 -1, -2, -1,
	  0,  0,  0,
	  1,  2,  1,
	};
	

	//Allocation of device variables
	cudaMallocArray(&carray,&channel,width,height);
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

	cudaMalloc((void**)&filter1_d,bytes1);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Alloc failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			return false;
	     
	
	}

	cudaMalloc((void**)&filter2_d,bytes1);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Alloc failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			return false;
	     
	
	}

		// Copies the required input from Host to device
	cudaMemcpy(image_d, image->image, bytes, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Copy of image failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			return false;
	}

	cudaMemcpy(filter1_d,filter1, bytes1, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Copy of filter failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			return false;
	}

	cudaMemcpy(filter2_d,filter2, bytes1, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Copy of filter failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			return false;
	}

	cudaMemcpyToSymbol(filter_d1, filter1, bytes1, 0, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Copy of constant filter failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			cudaFree(filter_d1);
			return false;
	}

	cudaMemcpyToSymbol(filter_d2, filter2, bytes1, 0, cudaMemcpyHostToDevice);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Copy of constant filter failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			cudaFree(filter_d2);
			return false;
	}

	
	cudaMemcpyToArray(carray,0,0,image->image,bytes,cudaMemcpyHostToDevice); // Copy required data to Array
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Copy of image failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
			return false;
	}

	//Set Texture address mode property
	texIn.addressMode[0]=cudaAddressModeWrap;
	texIn.addressMode[1]=cudaAddressModeClamp;
	cudaBindTextureToArray(texIn,carray);

	dim3 dimBlock(Tile_width,Tile_width); //  Dimension of block
	dim3 dimGrid((int)ceil((float)width / (float)Tile_width), (int)ceil((float)height / (float)Tile_width)); // Dynamic allocation for dimension of grid
	if (choice == 1)
	EdgeSobelGPUKernel<<<dimGrid, dimBlock>>>(image_d,Outputimage_d,width,height,filter1_d,filter2_d); // Kernel call
	else if(choice == 2)
	EdgeSobelGPUKernel1<<<dimGrid, dimBlock>>>(image_d,Outputimage_d,width,height);
	else if(choice == 3)
	EdgeSobelGPUKernel3<<<dimGrid, dimBlock>>>(Outputimage_d,width,height,filter1_d,filter2_d);
	status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Kernel failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(image_d);
			cudaFree(Outputimage_d);
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
	cudaFree(filter1_d);
	cudaFree(filter2_d);

	return true; 
	
	}



