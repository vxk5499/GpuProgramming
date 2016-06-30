#include "common.h"

// Peforms scaling on a vector based on the scaleFactor value
void scaleVectorCPU( float* a, float* c, float scaleFactor, int size )
{
	for(int i=0; i<size; i++){
		c[i] = scaleFactor * a[i] ;
			}
}