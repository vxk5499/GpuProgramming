/************************************************************************/
// The purpose of this program is to perform K-MeansClustering.
//
// Author: <Your Name>
// Date: April 26, 2012
// Course: 0306-724 - High Performance Architectures
//
// File: main.cpp
/************************************************************************/

#include <cmath> // sqrt()
#include <ctime> // time(), clock()
#include <iostream> // cout, stream
#include <fstream>
#include "KMeans.h"

#define ITERS 2
#define DATA_SIZE (1<<25)

// To reset the cluster data between runs.
void initializeClusters(Vector2* clusters)
{
	clusters[0].x = 0;
	clusters[0].y = 0;

	clusters[1].x = 1;
	clusters[1].y = 0;

	clusters[2].x = -1;
	clusters[2].y = 0;
}

void KMeansCPU(Datapoint* data, long n, Vector2* clusters, int k)
{
	
	int flag = 0;	//Flag to check if the points have changed clusters
	 // Stores the number of points in each cluster
	int iter = 0;
	while(flag == 0)
	{
		float pin[3]={0,0,0};
		flag = 1;
		
		for(long i = 0; i < DATA_SIZE; i++)
		{
			int pos = 0;
			int ip = data[i].cluster;// Initial cluster for data point
			float dist1 = clusters[0].distSq(data[i].p); // Distance of the point from the center point of the cluster 1
			float dist2 = clusters[1].distSq(data[i].p); // Distance of the point from the center point of the cluster 2
			float dist3 = clusters[2].distSq(data[i].p); // Distance of the point from the center point of the cluster 3
			// Finding the smallest distance
			if(dist1<=dist2 && dist1<=dist3)
				pos = 0;
			
			if(dist2<=dist1 && dist2<=dist3)
				pos = 1;

			if(dist3<=dist2 && dist3<=dist1)
				pos = 2;

			
			// Assignment of the datapoint to the nearest cluster
			switch (pos){
			case 0: data[i].cluster = 0;
					break;
			case 1: data[i].cluster = 1;
					break;
			case 2: data[i].cluster = 2;
					break; 

			}
		
			//Count of number of points in particular cluster
			switch (data[i].cluster){
			case 0: pin[0] += 1;
				break;
			case 1: pin[1] += 1;
				break;
			case 2: pin[2] += 1;
				break;
			}
			

			int fp = data[i].cluster;// Final cluster for data point

			//Checking whether the data point has chnaged the cluster
			if( ip != fp)
				flag = 0;

		}

//Finding the centre in cluster
		for(int j = 0; j<k ; j++)
		{
			float X = 0, Y = 0;//Variable to store the sum of values in x and y direction
			for(long i = 0; i < DATA_SIZE; i++)
			{
				if(data[i].cluster == j)
				{
					X += data[i].p.x;
					Y += data[i].p.y;
				}
			}
		clusters[j].x = X / pin[j];// Finding cluster center in x direction
		clusters[j].y = Y / pin[j];// Finding cluster center in y direction
		}
		iter++;
	}

}

/* Entry point for the program. 
   Performs k-means clustering on some sample data. */
int main() 
{
	// The data we want to operate on.
	Datapoint* data = new Datapoint[DATA_SIZE];
	Datapoint* dataCPU = new Datapoint[DATA_SIZE];
    Datapoint* dataGPU = new Datapoint[ DATA_SIZE ];
	Vector2 clusters[3];

	std::cout << "Performing k-means clustering on " << DATA_SIZE << " values." << std::endl;
	
	// Fill up the example data using three gaussian distributed clusters.
	for (long i = 0; i < DATA_SIZE; i++) {
		int cluster = rand()%3;
		float u1 = (float)(rand()+1)/(float)RAND_MAX;
		float u2 = (float)(rand()+1)/(float)RAND_MAX;
		float z1 = sqrt(abs(-2 * log(u1))) * sin(6.283f*u2);
		float z2 = sqrt(abs(-2 * log(u1))) * cos(6.283f*u2);
		data[i].cluster = cluster; // ground truth
		switch (cluster) {
			case 0:
				data[i].p.x = z1;
				data[i].p.y = z2;
				break;
			case 1:
				data[i].p.x = 2 + z1 * 0.5f;
				data[i].p.y = 1 + z2 * 0.5f;
				break;
			case 2:
				data[i].p.x = -2 + z1 * 0.5f;
				data[i].p.y = 1 + z2 * 0.5f;
				break;
		}
	}


	float tcpu, tgpu;
	clock_t start, end;

	// Perform the host computations
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		memcpy(dataCPU, data, sizeof(Datapoint) * DATA_SIZE);
		initializeClusters(clusters);
		KMeansCPU(dataCPU, DATA_SIZE, clusters, 3);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

    long incorrect = 0;
	for (long i = 0; i < DATA_SIZE; i++)
		if (data[i].cluster != dataCPU[i].cluster) incorrect++;

	// Display the results
	std::cout << "Host Result took " << tcpu << " ms (" << (float)incorrect / (float)DATA_SIZE * 100 << "% misclassified)" << std::endl;
	for (int j = 0; j < 3; j++)
		std::cout << "Cluster " << j << ": " << clusters[j].x << ", " << clusters[j].y << std::endl;
    std::cout << std::endl;

	//=================================================================================================
	//Insert your code for the GPU computation here. You should follow the same 
	//format as the code provided for the CPU.
	//=================================================================================================
    
	// Perform the host computations
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		memcpy(dataGPU, data, sizeof(Datapoint) * DATA_SIZE);
		initializeClusters(clusters);
		KMeansGPU(dataGPU, DATA_SIZE, clusters, 3);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

     incorrect = 0;
	for (long i = 0; i < DATA_SIZE; i++)
		if (data[i].cluster != dataGPU[i].cluster) incorrect++;

	// Display the results
	std::cout << "GPU Result took " << tgpu << " ms (" << (float)incorrect / (float)DATA_SIZE * 100 << "% misclassified)" << std::endl;
	for (int j = 0; j < 3; j++)
		std::cout << "Cluster " << j << ": " << clusters[j].x << ", " << clusters[j].y << std::endl;
    std::cout << std::endl;
	//Write the results to a file.
    std::ofstream outfile("results.csv");
	outfile << "x,y,Truth,CPU,GPU" << std::endl;
	for (long i = 0; i < DATA_SIZE; i++) {
		outfile << data[i].p.x << "," << data[i].p.y << "," << data[i].cluster << "," << dataCPU[i].cluster << "," << dataGPU[i].cluster << "\n";
	}
	outfile.close();

	delete[] data;
	delete[] dataCPU;
    delete[] dataGPU;

	// Success
	return 0;
}
