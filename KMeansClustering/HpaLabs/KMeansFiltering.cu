#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "KMeans.h"
#include <fstream> 

__constant__ Vector2 clusters_d[3];

// Kernel to perform the assignment of data points to the clusters
__global__ void KMeansKernel ( Datapoint* data_d, long n, int k, int* flag )
{					
		
			int pos = 0;
			int th_id = blockIdx.x *blockDim.x + threadIdx.x;
			if(th_id<n){
			int ip = data_d[th_id].cluster;	 // Initial cluster for data point
			float dist1 = clusters_d[0].distSq(data_d[th_id].p); // Distance of the point from the center point of the cluster 1
			float dist2 = clusters_d[1].distSq(data_d[th_id].p); // Distance of the point from the center point of the cluster 2
			float dist3 = clusters_d[2].distSq(data_d[th_id].p); // Distance of the point from the center point of the cluster 3
			
			// Finding the smallest distance
			if(dist1<=dist2 && dist1<=dist3)
				pos = 0;
			
			if(dist2<=dist1 && dist2<=dist3)
				pos = 1;

			if(dist3<=dist2 && dist3<=dist1)
				pos = 2;

		
			// Assignment of the datapoint to the nearest cluster
			switch (pos){
			case 0: data_d[th_id].cluster = 0;
					break;
			case 1: data_d[th_id].cluster = 1;
					break;
			case 2: data_d[th_id].cluster = 2;
					break; 

			}
		

			int fp = data_d[th_id].cluster; // Final cluster for data point
			//Checking whether the data point has chnaged the cluster
			if( ip != fp){
				*flag = 0;}
			}
}
bool KMeansGPU( Datapoint* data, long n, Vector2* clusters, int k )
{
	cudaError_t status; // Check for Error in Cuda

	int size = n * sizeof(Datapoint);
	int size1 = 3 * sizeof(Vector2);
	int size2 = sizeof(int);

	int *flag_d;

	Datapoint* data_d;
	//Allocation of device data
	cudaMalloc((void**) &data_d, size);
			status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Alloc failed: " << cudaGetErrorString(status) <<	std::endl;
			
			cudaFree(data_d);
			return false;     
		}
	//Allocation of device data
	cudaMalloc((void**) &flag_d, size2);

	// Copying of data points to device memory from host
	cudaMemcpy(data_d, data, size, cudaMemcpyHostToDevice);
			status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Alloc1 failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(clusters_d);
			cudaFree(data_d);
			return false;     
		}
			cudaMemcpyToSymbol(clusters_d, clusters, size1, 0, cudaMemcpyHostToDevice);
				status = cudaGetLastError();


	dim3 dimBlock(768);
	dim3 dimGrid((int)ceil((float)n/(float)768));
	int flag = 0;
	while(flag == 0)
	{

		//float pin[3]={0,0,0};
		flag = 1;

		// Copying of data points to device memory from host
		cudaMemcpy(data_d, data, size, cudaMemcpyHostToDevice);
		if (status != cudaSuccess) {
			std::cout << "Memcpy failed: " << cudaGetErrorString(status) <<	std::endl;
			cudaFree(data_d);
			return false;     
		}

		// Copying of flag to device memory from host
		cudaMemcpy(flag_d, &flag, size2, cudaMemcpyHostToDevice);
				status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Memcpy1 failed: " << cudaGetErrorString(status) <<	std::endl;
			
			cudaFree(data_d);
			return false;     
		}

		//Calling of Kernel
		KMeansKernel<<<dimGrid, dimBlock>>>(data_d, n, k,flag_d);
			status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Kernel failed: " << cudaGetErrorString(status) <<	std::endl;
		
			cudaFree(data_d);
			return false;     
		}

		cudaThreadSynchronize();// Thread synchronize

		//Copy of data points to host from device
		cudaMemcpy(data, data_d, size, cudaMemcpyDeviceToHost);
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Memcpy2 failed: " << cudaGetErrorString(status) <<	std::endl;
			
			cudaFree(data_d);
			return false;     
		}

		//Copy of flag to host from device
		cudaMemcpy(&flag, flag_d, size2, cudaMemcpyDeviceToHost);
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Memcpy3 failed: " << cudaGetErrorString(status) <<	std::endl;
			
			cudaFree(data_d);
			return false;     
		}
		
		int count; // To keep track of number of data points in cluster

		//Finding the centre in cluster
		for(int j = 0; j<k ; j++)
		{
			count = 0;
			float X = 0, Y = 0; // Variable to store the sum of values in x and y direction
			for(long i = 0; i < n; i++)
			{
				if(data[i].cluster == j)
				{
					X += data[i].p.x;
					Y += data[i].p.y;
					count += 1;
				}
			}
		clusters[j].x = X / count; // Finding cluster center in x direction
		clusters[j].y = Y / count; // Finding cluster center in y direction
		}

		// Copying of updated cluster values to constant memory
		cudaMemcpyToSymbol(clusters_d, clusters, size1, 0, cudaMemcpyHostToDevice);
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Memcpy4 failed: " << cudaGetErrorString(status) <<	std::endl;
			
			cudaFree(data_d);
			return false;     
		}
		
	}

	cudaFree(data_d);
	cudaFree(flag_d);
	return true;

}