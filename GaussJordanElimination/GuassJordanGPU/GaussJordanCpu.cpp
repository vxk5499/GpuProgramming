#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include "common.h"
#include <time.h>

const int numberOfRows= 3;
const int numberOfColumns= 4;



void swap(float** outputMatrix, int rowNumber, int numberOfColumns)
{
	float *temp= new float[numberOfColumns];
	for (int j=0; j<numberOfColumns; j++)
		temp[j] = outputMatrix [rowNumber] [j];
	for (int a=0; a<numberOfColumns; a++)
		outputMatrix [rowNumber][a]= outputMatrix [rowNumber+1][a];
	for (int a=0; a<numberOfColumns; a++)
		outputMatrix [rowNumber+1][a]= temp[a];
}

void GaussianEliminationCPU( float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot ){

	for(int m=0; m < numberOfRows; m++)
		for(int n=0; n<numberOfColumns; n++){
			outputMatrix[m][n]=matrix[m][n];
		}
	std :: cout<<"The input matrix copied to output matrix:" <<std::endl;		
	for(int i=0; i< numberOfRows; i++){
		std :: cout <<std::endl;
		for(int j=0; j< numberOfColumns; j++){
			std :: cout << outputMatrix[i][j] <<"\t\t";
		}
	}


	for( int i=0; i < numberOfRows; i++)
	{
		// Swap of rows based on certain conditions
		if (i != numberOfRows-1 && outputMatrix[i][i]==0)
			swap( outputMatrix,  i,  numberOfColumns);
		// Scaling of row
		float scale = outputMatrix[i][i];
		for(int j=0; j < numberOfColumns; j++)
		{
			outputMatrix[i][j] = outputMatrix[i][j]/scale;
		}
			std :: cout<<"The output matrix after scaling operation: "<< i <<std::endl;		
	for(int i=0; i< numberOfRows; i++){
		std :: cout <<std::endl;
		for(int j=0; j< numberOfColumns; j++){
			std :: cout << outputMatrix[i][j] <<"\t\t";
		}
	}
		
		// Elementary operation on the matrix
		if (i == 0)
			for(int l=1; l < numberOfRows; l++)
			{
				int temp = outputMatrix[l][i];
			for(int k=0; k < numberOfColumns; k++){


					std :: cout << temp<<std::endl;
					outputMatrix [l][k] =  outputMatrix [l][k]- outputMatrix [i][k] * temp ;
			}
			}
			

			
		else if(i != 0 && i < numberOfRows-1)
		{
			for(int l=0; l < i; l++){
				int temp = outputMatrix[l][i];
				for(int k=0; k < numberOfColumns; k++){

					std :: cout << temp<<std::endl;
					outputMatrix [l][k] = outputMatrix [l][k] - outputMatrix [i][k] * temp  ;
				}
			}
			for(int l=i+1; l < numberOfRows; l++){
				int temp = outputMatrix[l][i];

				for(int k=0; k < numberOfColumns; k++){

					std :: cout << temp<<std::endl;
					outputMatrix [l][k] = outputMatrix [l][k]- outputMatrix [i][k]  * temp  ;
		}
			}
		}

		else if( i==numberOfRows-1){
			for(int l=0; l < numberOfRows-1; l++){
				int temp = outputMatrix[l][i];

				for(int k=0; k < numberOfColumns; k++){

					std :: cout << temp<<std::endl;
					outputMatrix [l][k] = outputMatrix [l][k] - outputMatrix [i][k] * temp  ;
				}

		}}


		std :: cout<<"The output matrix after one row operation: "<< i <<std::endl;		
	for(int i=0; i< numberOfRows; i++){
		std :: cout <<std::endl;
		for(int j=0; j< numberOfColumns; j++){
			std :: cout << outputMatrix[i][j] <<"\t\t";
		}
	}

	
	}

	

}
