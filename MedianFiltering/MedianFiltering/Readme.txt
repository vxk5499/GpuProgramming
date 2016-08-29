Median Filtering is used in image processing which can remove sufficient amount of salt and pepper noise from the image and thus produce a better image. We can see the since median filtering is done to each pixel in the image, it is extremely parallel in nature. In this case a 3*3 median filter is applied to the image. This implementation is done in CPU. In GPU, there are two implementations i.e. using global memory and shared memory. The implementation is tested with three different images.

Median Filtering CPU implementation:
In this case first the input image is loaded and an output image equal to the size of input image is created. The boundary pixels is this case are replicated. For every non boundary pixel “a” as shown in the figure 1 eight neighboring values along with the pixel value is passed on to the median filter(Here 3 x 3 filter is used). These pixels are then sorted and the median value is found. This median value is set at the output image in the corresponding position as the current operating pixel in the input image.
1	2	3
4	a	5
6	7	8



Fig 1: 3 X 3 Median filter.

Once all the median values are mapped to the output image, the image is then saved.


Median Filtering GPU implementation:
In this case since the number of pixels in image exceed the block size, tiling is done for operating on all the pixels in the image. The tile width used in this implementation is 16.  The blockdim is tilewidth x tilewidth which results in 256 threads per block or 6 blocks/1536 threads per SM thus resulting in efficient handling of resources. The grid size is dependent on the input image dimensions. In this implementation a 2D block, 2D grid is used. 
Global memory implementation:
In the global memory the global row and column indices are found at the beginning. This implementation will be similar to the CPU implementation except the fact that all the pixels will be processed in parallel. In this case too, boundary pixels are replicated. For each non-boundary pixels the eight neighboring values along with the pixel value is passed on to the median filter along with the pixel value. Similar to the CPU implementation these pixels are sorted and then mapped to output image. 
Shared Memory:
The shared memory has scope within a block i.e. only threads within a particular block can access shared memory as each block has its own shared memory.
In this case a shared memory variable of dimension tile_width+2 x tile_width+2 is created. The extra rows and columns in the shared memory variable as compared to tile size are used to map appropriate neighbors for operating pixels in that shared memory variable for median filtering.
In other words only pixels from row 1 – row 16 and column 1 – column 16 are the actual operating pixels.The border pixel for each shared memory variable are at first initialized to zero. After this mapping is done for the border pixels from global memory with condition based on the threadId of x and y direction in block and the global row and column id. 
After this all the remaining pixels are mapped from the global memory to row 1 – row 16 and column 1 – column 16. Once all the mapping is done the threads are synchronized to prevent conflicts i.e. race condition between the different threads.
After this for each non-boundary elements its required neighboring elements are read form the shared memory followed by sorting to find median values. The mapping of median values to their corresponding position in output images is done. For boundary pixels, they are replicated to output image.

