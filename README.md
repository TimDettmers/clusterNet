clusterNet
==============

Deep neural network framework for GPU clusters:

- supports NVIDIA GPUDirect RDMA
- easy distributed computation:

	Matrix C = dot(A,B); 	//uses one GPU  
	Matrix C = dotPCI(A,B); //uses all GPUs on the board  
	Matrix C = dotMPI(A,B); //uses all GPUs in the network  
- no delay between batches due to asynchronous host to GPU memory copies:  
<code>gpu.init_batch_allocator(X, y, 128);  
	for(int i = 0; i < gpu.m_total_batches; i++)  
		{  
	  	gpu.allocate_next_batch_async(); //loads the next batch while you do computations  
	  	result = gpu.dot(gpu.m_current_batch_X,w1); //do your computations here  
	  	gpu.replace_current_batch_with_next(); //get the next batch which is already loaded  
		}  
</code>
	  
  
