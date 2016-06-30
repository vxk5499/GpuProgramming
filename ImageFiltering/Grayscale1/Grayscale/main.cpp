#include "bitmap.h"
#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include <fstream>  
#include <time.h>
#include <math.h> 
#include "common.h"

const int iterations=50 ;

void SharpenCPU( Bitmap* image, Bitmap* outputImage )
{
	//Filter that produces sharpening effect
	double filter[3][3] =
	{
		0, -1, 0,
	    -1, 5,-1,
        0, -1, 0
	};
	
	//Performing image convolution with filter
	for(int j=0; j<image->Width(); j++) //indicates current row
	{
		for(int i = 0; i < image->Height(); i++) // indicates current column
		{ 
			double accum = 0;

		
			for(int fw = 0 ; fw < 3; fw++)
				for(int fh = 0; fh < 3; fh++) 
				{
					int ix = ( j - 1 + fw + image->Width())% image->Width();
					int iy = ( i - 1 + fh + image->Height())%image->Height();
					accum = accum + image->GetPixel(ix, iy) * filter[fw][fh];
				}
			unsigned char temp = accum;
			outputImage->SetPixel(j,i,temp);
		}
	}


}


void EmbossCPU( Bitmap* image, Bitmap* outputImage )
{
	//Filter that produces Emboss effect
	double filter[3][3] =
	{
		-1, -1,  0,
		-1,  0,  1,
		 0,  1,  1
	};
	
	//Performing image convolution with filter
	for(int j=0; j<image->Width(); j++) //indicates current row
	{
		for(int i = 0; i < image->Height(); i++) // indicates current column
		{ 
			double accum = 0;
			for(int fw = 0 ; fw < 3; fw++)
				for(int fh = 0; fh < 3; fh++)
				{
					int ix = ( j - 1 + fw + image->Width())% image->Width();
					int iy = ( i - 1 + fh + image->Height())%image->Height();
					accum = accum + image->GetPixel(ix, iy) * filter[fw][fh];
				}
			unsigned char temp = accum;
			outputImage->SetPixel(j,i,temp);
		}
	}


}


void BlurCPU( Bitmap* image, Bitmap* outputImage )
{
	//Filter that produces blur effect
	double filter[5][5] =
	{
		 0, 0, 1, 0, 0,
		 0, 1, 1, 1, 0,
		 1, 1, 1, 1, 1,
		 0, 1, 1, 1, 0,
		 0, 0, 1, 0, 0,
	};
	
	//Performing image convolution with filter
	for(int j=0; j<image->Width(); j++) //indicates current row
	{
		for(int i = 0; i < image->Height(); i++) // indicates current column
		{ 
			double accum = 0;
			for(int fw = 0 ; fw < 5; fw++)
				for(int fh = 0; fh < 5; fh++)
				{
					int ix = ( j - 2 + fw + image->Width())% image->Width();
					int iy = ( i - 2 + fh + image->Height())%image->Height();
					accum = accum + image->GetPixel(ix, iy) * filter[fw][fh] ;
				}
			accum /= 13;
			unsigned char temp = accum;
			outputImage->SetPixel(j,i,temp);
		}
	}


}

void MotionBlurCPU( Bitmap* image, Bitmap* outputImage )
{

	//Filter that produces motion blur effect
	double filter[9][9] =
	{
		
    1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1,


	};
	
	//Performing image convolution with filter
	for(int j=0; j<image->Width(); j++) //indicates current row
	{
		for(int i = 0; i < image->Height(); i++) // indicates current column
		{ 
			double accum = 0;
			for(int fw = 0 ; fw < 9; fw++)
				for(int fh = 0; fh < 9; fh++)
				{
					int ix = ( j - 4 + fw + image->Width())% image->Width();
					int iy = ( i - 4 + fh + image->Height())%image->Height();
					accum = accum + image->GetPixel(ix, iy) * filter[fw][fh] ;
				}
			accum /= 9;
			unsigned char temp = accum;
			outputImage->SetPixel(j,i,temp);
		}
	}


}

void EdgeSobelCPU( Bitmap* image, Bitmap* outputImage )
{

	//Filter that detects edges
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

	//Performing image convolution with filter
	for(int j=0; j<image->Width(); j++) //indicates current row
	{
		for(int i = 0; i < image->Height(); i++) // indicates current column
		{ 
			double accum = 0,accum1 = 0,accum2 = 0;
			for(int fw = 0 ; fw < 3; fw++)
				for(int fh = 0; fh < 3; fh++)
				{
					int ix = ( j - 1 + fw + image->Width())% image->Width();
					int iy = ( i - 1 + fh + image->Height())%image->Height();
					accum1 = accum1 + image->GetPixel(ix, iy) * filter1[fw][fh] ;
					accum2 = accum2 + image->GetPixel(ix, iy) * filter2[fw][fh] ;
				}
			accum= sqrt(pow(accum1,2)+pow(accum2,2));
			unsigned char temp = accum;
			outputImage->SetPixel(j,i,temp);
		}
	}


}
int main()
{
// Timing variables
	float tcpu, tgpu;
	clock_t start, end;
//Input and output variables
	Bitmap *inputA = new Bitmap();
	Bitmap *outputA = new Bitmap();

	inputA->Load("lenna.bmp");
	outputA->Load("lenna.bmp");

	
	std::cout << "Operating on an image size of 252 x 205"<<std::endl<<"\n";
	start = clock();
	for (int i = 0; i < iterations; i++) {
	EmbossCPU(inputA, outputA);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result for Emboss took " << tcpu << " ms" << std::endl;
	outputA->Save("lennaEmboss.bmp");

	EmbossGPU(inputA, outputA, 1); // GPU warmup
	start = clock();
	for (int i = 0; i < iterations; i++) {
	EmbossGPU(inputA, outputA, 1);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("lennaGPUEmboss.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	EmbossGPU(inputA, outputA, 2);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Constant memory is" << tgpu << " ms" << std::endl;
	std::cout << "Constant memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("lennaGPUCEmboss.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	EmbossGPU(inputA, outputA, 3);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Texture memory is" << tgpu << " ms" << std::endl;
	std::cout << "Texture memory Speedup: " << tcpu/tgpu <<  std::endl<<"\n";
	outputA->Save("lennaGPUTEmboss.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	SharpenCPU(inputA, outputA);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result for Sharpen took " << tcpu << " ms" << std::endl;
	outputA->Save("lennaSharpen.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	SharpenGPU(inputA, outputA, 1);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("lennaSharpenGPU.bmp");

	
	start = clock();
	for (int i = 0; i < iterations; i++) {
	SharpenGPU(inputA, outputA, 2);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Constant memory is" << tgpu << " ms" << std::endl;
	std::cout << "Constant memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("lennaSharpenGPUC.bmp");

	
	start = clock();
	for (int i = 0; i < iterations; i++) {
	SharpenGPU(inputA, outputA, 3);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Texture memory is" << tgpu << " ms" << std::endl;
	std::cout << "Texture memory Speedup: " << tcpu/tgpu <<  std::endl<<"\n";
	outputA->Save("lennaSharpenGPUT.bmp");


	start = clock();
	for (int i = 0; i < iterations; i++) {
	BlurCPU(inputA, outputA);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result for blur took " << tcpu << " ms" << std::endl;
	outputA->Save("lennablur.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	BlurGPU(inputA, outputA,1);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("lennaGPUblur.bmp");

	
	start = clock();
	for (int i = 0; i < iterations; i++) {
	BlurGPU(inputA, outputA,2);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Constant memory is" << tgpu << " ms" << std::endl;
	std::cout << "Constant memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("lennaGPUCblur.bmp");

	
	start = clock();
	for (int i = 0; i < iterations; i++) {
	BlurGPU(inputA, outputA,3);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Texture memory is" << tgpu << " ms" << std::endl;
	std::cout << "Texture memory Speedup: " << tcpu/tgpu <<  std::endl<<"\n";
	outputA->Save("lennaGPUTblur.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	EdgeSobelCPU(inputA, outputA);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result for sobel took " << tcpu << " ms" << std::endl;
	outputA->Save("lennasobel.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	EdgeSobelGPU(inputA, outputA, 1);
	
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("lennasobelGPU.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	EdgeSobelGPU(inputA, outputA, 2);
	
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Constant memory is" << tgpu << " ms" << std::endl;
	std::cout << "Constant memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("lennasobelGPUC.bmp");

		start = clock();
	for (int i = 0; i < iterations; i++) {
	EdgeSobelGPU(inputA, outputA, 3);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Texture memory is" << tgpu << " ms" << std::endl;
	std::cout << "Texture memory Speedup: " << tcpu/tgpu <<  std::endl<<"\n";
	outputA->Save("lennasobelGPUT.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	MotionBlurCPU(inputA, outputA);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result for mblur took " << tcpu << " ms" << std::endl;
	outputA->Save("lenamblur.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	MotionBlurGPU(inputA, outputA,1);
	
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("lenaGPUmblur.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	MotionBlurGPU(inputA, outputA,2);

	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Constant memory is" << tgpu << " ms" << std::endl;
	std::cout << "Constant memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("lenaGPUCmblur.bmp");

		start = clock();
	for (int i = 0; i < iterations; i++) {
	MotionBlurGPU(inputA, outputA,3);

	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Texture memory is" << tgpu << " ms" << std::endl;
	std::cout << "Texture memory Speedup: " << tcpu/tgpu <<  std::endl <<"\n";
	outputA->Save("lenaGPUTmblur.bmp");



	inputA->Load("RIT.bmp");
	outputA->Load("RIT.bmp");
	std::cout << "Operating on an image size of 946 X 532"<<std::endl<<"\n";
	start = clock();
	for (int i = 0; i < iterations; i++) {
	EmbossCPU(inputA, outputA);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result for  Emboss took " << tcpu << " ms" << std::endl;
	outputA->Save("RITEmboss.bmp");

	EmbossGPU(inputA, outputA, 1);
	start = clock();
	for (int i = 0; i < iterations; i++) {
	EmbossGPU(inputA, outputA, 1);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("RITGPUEmboss.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	EmbossGPU(inputA, outputA, 2);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Constant memory is" << tgpu << " ms" << std::endl;
	std::cout << "Constant memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("RITGPUCEmboss.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	EmbossGPU(inputA, outputA, 3);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Texture memory is" << tgpu << " ms" << std::endl;
	std::cout << "Texture memory Speedup: " << tcpu/tgpu <<  std::endl<<"\n";
	outputA->Save("RITGPUTEmboss.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	SharpenCPU(inputA, outputA);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result for Sharpen took " << tcpu << " ms" << std::endl;
	outputA->Save("RITSharpen.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	SharpenGPU(inputA, outputA, 1);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("RITSharpenGPU.bmp");

	
	start = clock();
	for (int i = 0; i < iterations; i++) {
	SharpenGPU(inputA, outputA, 2);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Constant memory is" << tgpu << " ms" << std::endl;
	std::cout << "Constant memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("RITSharpenGPUC.bmp");

	
	start = clock();
	for (int i = 0; i < iterations; i++) {
	SharpenGPU(inputA, outputA, 3);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Texture memory is" << tgpu << " ms" << std::endl;
	std::cout << "Texture memory Speedup: " << tcpu/tgpu <<  std::endl<<"\n";
	outputA->Save("RITSharpenGPUT.bmp");


	start = clock();
	for (int i = 0; i < iterations; i++) {
	BlurCPU(inputA, outputA);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result for  blur took " << tcpu << " ms" << std::endl;
	outputA->Save("RITblur.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	BlurGPU(inputA, outputA,1);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("RITGPUblur.bmp");

	
	start = clock();
	for (int i = 0; i < iterations; i++) {
	BlurGPU(inputA, outputA,2);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Constant memory is" << tgpu << " ms" << std::endl;
	std::cout << "Constant memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("RITGPUCblur.bmp");

	
	start = clock();
	for (int i = 0; i < iterations; i++) {
	BlurGPU(inputA, outputA,3);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Texture memory is" << tgpu << " ms" << std::endl;
	std::cout << "Texture memory Speedup: " << tcpu/tgpu <<  std::endl<<"\n";
	outputA->Save("RITGPUTblur.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	EdgeSobelCPU(inputA, outputA);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result for sobel took " << tcpu << " ms" << std::endl;
	outputA->Save("RITsobel.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	EdgeSobelGPU(inputA, outputA, 1);
	
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("RITsobelGPU.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	EdgeSobelGPU(inputA, outputA, 2);
	
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Constant memory is" << tgpu << " ms" << std::endl;
	std::cout << "Constant memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("RITsobelGPUC.bmp");

		start = clock();
	for (int i = 0; i < iterations; i++) {
	EdgeSobelGPU(inputA, outputA, 3);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Texture memory is" << tgpu << " ms" << std::endl;
	std::cout << "Texture memory Speedup: " << tcpu/tgpu <<  std::endl<<"\n";
	outputA->Save("RITsobelGPUT.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	MotionBlurCPU(inputA, outputA);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Host result for  mblur took " << tcpu << " ms" << std::endl;
	outputA->Save("RITmblur.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	MotionBlurGPU(inputA, outputA,1);
	
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Global memory is" << tgpu << " ms" << std::endl;
	std::cout << "Global memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("RITGPUmblur.bmp");

	start = clock();
	for (int i = 0; i < iterations; i++) {
	MotionBlurGPU(inputA, outputA,2);

	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Constant memory is" << tgpu << " ms" << std::endl;
	std::cout << "Constant memory Speedup: " << tcpu/tgpu <<  std::endl;
	outputA->Save("RITGPUCmblur.bmp");

		start = clock();
	for (int i = 0; i < iterations; i++) {
	MotionBlurGPU(inputA, outputA,3);

	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iterations;
	std::cout << "Average Time per GPU Iteration with Texture memory is" << tgpu << " ms" << std::endl;
	std::cout << "Texture memory Speedup: " << tcpu/tgpu <<  std::endl <<"\n";
	outputA->Save("RITGPUTmblur.bmp");

}