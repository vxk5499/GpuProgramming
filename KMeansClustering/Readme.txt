KMeans clustering is a method of classification of data into clusters. The algorithm consists of two stages, the first one being the assignment of the point to a particular cluster based on distance of point from clusters, the second step being finding the center of each cluster. This algorithm stops when all points do not change its cluster. This algorithm is implemented in CPU and GPU. In the GPU the clusters are stored in the constant memory.

KMeans Clustering algorithm sorts the data points into clusters. Initially guesses are given for the cluster locations. The algorithm keeps iterating till the points do not change its clusters. The algorithm consists of two stages, the first being assignment of points to clusters and the second being update of each cluster center points.
CPU Implementation:
Assignment of data points to clusters:
In this the distance of each data point from the center of different cluster is taken and the smallest distance is noted. The data point is assigned to the nearest cluster.
Update of cluster center points:
In this each cluster is processed one by one. The positions of data points belonging to the particular cluster are added up and the total is divided by the total number of points in that cluster. Thus each cluster updates its center at each iteration
The above two processes continues until the data points do not change its cluster.

GPU Implementation:
In this a block size of (768) is used and a grid size of (int)ceil((float)n/(float)768) which ensures that all SM processes 1536 threads, thus all resources are used efficiently.

In this the Assignment phase is done by the GPU and the update phase is done by the CPU. The assignment phase is similar to CPU except the fact that all the data points would find its clusters in parallel. At each iterations the constant memory updates with new cluster positions.
