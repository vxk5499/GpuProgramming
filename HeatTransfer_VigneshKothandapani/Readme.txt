The heat transfer is the exchange of thermal energy between the physical systems by dissipating heat depending on the temperature and pressure. In the algorithm the temperature of the given cell can be found by using the neighboring cells values. The heat speed and the surrounding neighbor’s will have an impact on the final calculated value. The heat transfer algorithm is implemented in GPU and is then compared with the CPU implementation. At last a video file is generated that shows a transfer of heat for a size of 1000 x 1000.

GPU Implementation:
In this a block size of (16,16) is used and a grid size of ((int)ceil((float)size/(float)16, (int)ceil((float)size/(float)16)) which ensures that all SM processes 1536 threads, thus all resources are used efficiently.

In the heat transfer algorithm, at each point, the new heat at that point is calculated by using the simple equation shown below:
Tnew = Tcenter + HeatSpeed * (Tleft + Tright + Ttop + Tbottom - 4 * Tcenter)

•	Since each element calculation is independent of each other, the algorithm is highly parallel in GPU.
•	In this since the output heat map is calculated from an input heat map which remains constant, it is stored in texture memory which provides faster read speeds as compared to global memory. 
•	In this a 2D texture memory is used for storing the input heat map for each iteration. 
•	The input data point are first copied into the allocated device array. 
•	This device array is binded into texture memory which can be accessed in 2D. After each iteration the texture memory is updated with new values for the next iteration. 
