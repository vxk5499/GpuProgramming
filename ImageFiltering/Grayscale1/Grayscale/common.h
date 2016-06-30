#include "bitmap.h"


void SharpenCPU( Bitmap* image, Bitmap* outputImage );
void EmbossCPU( Bitmap* image, Bitmap* outputImage );
void BlurCPU( Bitmap* image, Bitmap* outputImage );
void MotionBlurCPU( Bitmap* image, Bitmap* outputImage );
void EdgeSobelCPU( Bitmap* image, Bitmap* outputImage );


bool  SharpenGPU( Bitmap* image, Bitmap* outputImage, int choice );
bool  EmbossGPU( Bitmap* image, Bitmap* outputImage, int choice  );
bool  BlurGPU( Bitmap* image, Bitmap* outputImage, int choice );
bool  MotionBlurGPU( Bitmap* image, Bitmap* outputImage, int choice );
bool  EdgeSobelGPU( Bitmap* image, Bitmap* outputImage, int choice  );

