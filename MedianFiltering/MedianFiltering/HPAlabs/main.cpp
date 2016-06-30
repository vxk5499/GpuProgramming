#include "Bitmap.h"
#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include <fstream>  
#include <time.h>
#include "MedianFilter.h"

const int iterations= 25;
// The images are compared and no of different pixels is returned.

int CompareBitmaps( Bitmap* inputA, Bitmap* inputB ){
 



		int dif=0;
		for(int i=0; i<inputA->Width(); i++){
			for(int j=0; j<inputA->Height(); j++)
			{
				if(inputA->GetPixel(i,j)!=inputB->GetPixel(i,j))
				{
					dif++;
				
				}
			}
	
		}
	return dif;
	
	}

// Used to find the median of filter values.
unsigned char median(int j, int i, Bitmap* image)
{


	
	unsigned char temp[9]={0,0,0,0,0,0,0,0,0};//Initialize the border pixels to 0.
	
	
	//Replication of border pixels.
	if( (j ==0) || (i == 0) || (j == (image->Width()-1)) || (i == (image->Height()-1)))
	{
		for (int x=0; x<sizeof(temp);x++)
		temp[x] = image->GetPixel(j,i);

	}

	
		
	//Finding the required filter values.
	else{
	temp[0] = image->GetPixel(j-1,i-1);
	temp[1] = image->GetPixel(j-1,i);
	temp[2] = image->GetPixel(j-1,i+1);
	temp[3] = image->GetPixel(j,i-1);
	temp[4] = image->GetPixel(j,i);
	temp[5] = image->GetPixel(j,i+1);
	temp[6] = image->GetPixel(j+1,i-1);
	temp[7] = image->GetPixel(j+1,i);
	temp[8] = image->GetPixel(j+1,i+1);
	}

	// Bubble sort
	for (int k = 0; k < sizeof(temp); k++) {
      for (int l = k+1; l < sizeof(temp); l++) {
         if (temp[k] > temp[l]) {
            unsigned char temp1 = temp[k];
            temp[k] = temp[l];
            temp[l] = temp1;
         }
      }
	
}
	  return temp[4]; //Return the median value.

}

// Performs median filter operation on a image
void MedianFilterCPU( Bitmap* image, Bitmap* outputImage ){
	

	for(int i = 0; i < image->Height(); i++) //indicates current row
	{
		for(int j=0; j<image->Width(); j++){ // indicates current column

			unsigned char new_pixel = median(j,i, image);
			outputImage->SetPixel(j,i,new_pixel);
		
		}

   }
	
}


int main(){


	
	// Timing data
	float tcpu, tgpu;
	clock_t start, end;
	srand (time(NULL));

	Bitmap *inputA = new Bitmap();
	Bitmap *outputA = new Bitmap();
	Bitmap *outputB = new Bitmap();
	Bitmap *outputC = new Bitmap();

		
	inputA->Load("lenna.bmp");
	outputA->Load("lenna.bmp");
	outputB->Load("lenna.bmp");
	outputC->Load("lenna.bmp");
	int totalpixels = inputA->Height() * inputA->Width();
	std::cout<<"Operating on a "<<inputA->Width()<<" x "<<inputA->Height()<<std::endl;

	start = clock();
	for (int i = 0; i < iterations; i++) {	
	MedianFilterCPU( inputA, outputA); 
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per CPU Iteration is: " << tcpu << " ms" << std::endl; // Time taken by CPU for computation
	outputA->Save("lennaCPU.bmp");

	bool success = true;
	success = MedianFilterGPU (inputA, outputB, false);
	if (!success){
		std::cout << "\n * Device error! Add *\n" << std::endl;
		return 1;
	}
	start = clock();
	for (int i = 0; i < iterations; i++) {
	MedianFilterGPU (inputA, outputB, false);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	int result = CompareBitmaps(outputA,outputB);
	std::cout<<"Difference in number of pixels Global memory : "<<result<<std::endl;
	std::cout<<"Error:"<<(float)result/(float)totalpixels<<std::endl;
	outputB->Save("lennaGPU.bmp");

	success = MedianFilterGPU (inputA, outputC, true);
	if (!success){
		std::cout << "\n * Device error! Add *\n" << std::endl;
		return 1;
	}
	start = clock();
	for (int i = 0; i < iterations; i++) {
	MedianFilterGPU (inputA, outputC, true);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;	std::cout << "Average Time per GPU Iteration with Shared memory is" << tgpu << " ms" << std::endl; 
	std::cout << "Shared memory Speedup: " << tcpu/tgpu <<  std::endl;
	result = CompareBitmaps(outputA,outputC);
	std::cout<<"Difference in number of pixels shared memory: "<<result<<std::endl;
	std::cout<<"Error:"<<(float)result/(float)totalpixels<<std::endl;
	outputB->Save("lennaGPUShared.bmp");
	
	

}