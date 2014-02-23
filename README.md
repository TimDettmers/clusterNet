clusterNet
==============

Deep neural network framework for GPU clusters:

- supports NVIDIA GPUDirect RDMA
- easy distributed computation:

	Matrix C = dot(A,B); 	//uses one GPU
	Matrix C = dotPCI(A,B); //uses all GPUs 
	Matrix C = dotMPI(A,B); //uses all GPUs in all nodes
