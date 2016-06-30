#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include <cuda_runtime_api.h>
#include "common.h"
// Entry point for the program.
// Function to Check if the Kernel API returns error
bool checkForError(cudaError_t status){
	if (status != cudaSuccess ){
		printf("Kernel Error: %s/n",cudaGetErrorString(status));
	}
	return true;
}

int main()
{

//Keep the error status
int count, driverVersion, runtimeVersion, device;
cudaError_t status;

//Check for errors while calling Kernel API

status = cudaGetDeviceCount(&count);
checkForError( status);
status = cudaRuntimeGetVersion(&runtimeVersion);
checkForError( status);
status = cudaGetDeviceCount(&count);
checkForError( status);


cudaGetDeviceCount(&count);
cudaRuntimeGetVersion(&runtimeVersion);
cudaDriverGetVersion(&driverVersion);

// The following code returns the various properties associated with the GPU device

printf("CUDA Device Capabilities:\n\n");
printf("CUDA Devices Found: %d\n", count);
printf("CUDA Driver: %d\n", driverVersion);
printf("CUDA Runtime: %d\n\n", runtimeVersion);
for (int i = 0; i < count; i++){

	struct cudaDeviceProp prop;
	cudaError_t GetDevicePropertiesStatus;
	status = cudaGetDeviceProperties(&prop, i);
	checkForError( status);									// Checks for errors while calling Kernel API.
	cudaGetDeviceProperties(&prop, i);
	printf("Device %d: %s\n", i, prop.name);
	printf("CUDA Capability %d.%d\n",prop.major, prop.minor);
	printf("Processing:\n\t");
	printf("Multiprocessors: %d\n\t",prop.multiProcessorCount);
	printf("Max Grid Size: %d x %d x %d\n\t", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
	printf("Max Block Size: %d x %d x %d\n\t",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
	printf("Threads per Block: %d\n\t", prop.maxThreadsPerBlock);
	printf("Threads per Multiprocessor: %d\n\t", prop.maxThreadsPerMultiProcessor);
	printf("Warps size: %d\n\t", prop.warpSize);
	printf("Clock rate: %.3f Ghz\n", prop.clockRate/1000000.0);
	printf("Memory:  \n\t");
	printf("Global: %d MB\n\t",prop.totalGlobalMem>>20);
	printf("Constant: %d KB\n\t",prop.totalConstMem>>10);
	printf("Shared/blk: %d KB\n\t",prop.sharedMemPerBlock>>10);
	printf("Registers/blk: %d\n\t",prop.regsPerBlock);
	printf("Maximum Pitch: %d MB\n\t",prop.memPitch>>20);
	printf("Texture Alignment: %d B\n\t",prop.textureAlignment);
	printf("L2 Cache Size: %d B\n\t",prop.l2CacheSize);
	printf("Clock Rate: %d MHz\n",prop.memoryClockRate/1000);
	int d = prop.concurrentKernels;
	if (d==0)
		printf("Concurrent Copy & Execute : No\n",d);
	else
		printf("Concurrent Copy & Execute : Yes\n",d);
	int b = prop.kernelExecTimeoutEnabled;
	if (b==0)
		printf("Kernel Time limit : No\n");
	else
		printf("Kernel Time limit : Yes\n");
	int a = prop.canMapHostMemory;
	if (a==0)
		printf("Supports Page-Locked Memory mapping : No\n");
	else
		printf("Supports Page-Locked Memory mapping : Yes\n");
	int c = prop.computeMode;
	if (c==0)
		printf("Compute Mode : Default \n\n");
	else if (c==1)
		printf("Compute Mode : Exclusive\n\n");
	else if(c==2)
		printf("Compute Mode : Prohibited\n\n");
	else if(c==3)
		printf("Compute Mode : Exclusive Process\n\n");

	}

return 0;
}