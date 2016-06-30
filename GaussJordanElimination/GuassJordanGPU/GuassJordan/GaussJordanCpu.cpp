#include <cstdlib> // malloc(), free()
#include <iostream> // cout, stream
#include "common.h"
#include <time.h>
#include <fstream> 



// Used for swapping one row elements with another
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

// CPU implementation of Guass Jordan Elimination implementation
void GaussianEliminationCPU( float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot ){
	
	// Input matrix copied to the output matrix
	for(int m=0; m < numberOfRows; m++)
		for(int n=0; n<numberOfColumns; n++){
			outputMatrix[m][n]=matrix[m][n];
		}
	

	// Performing scaling and elementary operations to obtain augmented identity matrix
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
			
		// Elementary operations on the matrix
		if (i == 0)
			for(int l=1; l < numberOfRows; l++)
			{
				float temp = outputMatrix[l][i];
			
			for(int k=0; k < numberOfColumns; k++){

			 outputMatrix [l][k] =  outputMatrix [l][k]- outputMatrix [i][k] * temp ;
				}
			}
			

			
		else if(i != 0 && i < numberOfRows-1)
		{
			for(int l=0; l < i; l++){
				float temp = outputMatrix[l][i];
			
				for(int k=0; k < numberOfColumns; k++){

					
					outputMatrix [l][k] = outputMatrix [l][k] - outputMatrix [i][k] * temp  ;
				}
			}

			for(int l=i+1; l < numberOfRows; l++){
				float temp = outputMatrix[l][i];
				

				for(int k=0; k < numberOfColumns; k++){

					
					outputMatrix [l][k] = outputMatrix [l][k]- outputMatrix [i][k]  * temp  ;
				}
			}
		}

		else if( i==numberOfRows-1){
			for(int l=0; l < numberOfRows-1; l++){
				float temp = outputMatrix[l][i];
				

				for(int k=0; k < numberOfColumns; k++){
								
					outputMatrix [l][k] = outputMatrix [l][k] - outputMatrix [i][k] * temp  ;
					}

				}
		}


	}

	

}
