clusterNet
==============

Deep neural network framework for GPU clusters:

- supports NVIDIA GPUDirect RDMA
- easy distributed computation:

	Matrix C = dot(A,B); 	//uses one GPU  
	Matrix C = dotMPI(A,B); //uses all available GPUs on the board or in the network  
- no delay between batches due to asynchronous memory copies to the GPU:  
<code>gpu.init_batch_allocator(X, y, 128);  
	for(int i = 0; i < gpu.m_total_batches; i++)  
		{  
	  	gpu.allocate_next_batch_async(); //loads the next batch while you do computations  
	  	result = gpu.dot(gpu.m_current_batch_X,w1); //do your computations here  
	  	gpu.replace_current_batch_with_next(); //get the next batch which is already loaded  
		}  
</code>  
- distributed weights which are larger than a single GPU memory:  
<code>  
ClusterNet gpus = ClusterNet(argc,argv,12346);  
Matrix *batch = gpus.rand(128,100000);//34 MB  
Matrix *out1 = empty(128,40000);//19 MB  
Matrix *out2 = empty(128,20000);//9 MB  
Matrix *W1 = gpus.distributed_uniformSqrtWeight(100000,40000);//15258 MB  
Matrix *W2 = gpus.distributed_uniformSqrtWeight(40000,20000);//3051 MB  
gpus.tick("Time taken");  
gpus.dotMPI(batch,W1,out1);  
gpus.dotMPI(out1,W2,out2);  
gpus.tock("Time taken");  
>>>Time taken: 117.704285 ms.
</code>
	  
  
