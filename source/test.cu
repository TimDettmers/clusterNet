#include <stdio.h>
#include <cublas_v2.h>
#include <util.cuh>
#include <basicOps.cuh>
#include <mpi.h>
#include <cuda.h>
#include <assert.h>
#include <util.cuh>
#include <clusterNet.h>
#include <time.h>
#include <batchAllocator.h>
#include <DeepNeuralNetwork.h>

using std::cout;
using std::endl;

void run_neural_network()
{
  Matrix *X = read_hdf5("/home/tim/mnist_full_X.hdf5");
  Matrix *y = read_hdf5("/home/tim/mnist_full_y.hdf5");

  ClusterNet gpu = ClusterNet(12345);

  Matrix *w1 = gpu.sparseInitWeight(784,1000);
  Matrix *w2 = gpu.sparseInitWeight(1000,10);
  Matrix *m1 = zeros(784,1000);
  Matrix *m2 = zeros(1000,10);
  Matrix *ms1 = zeros(784,1000);
  Matrix *ms2 = zeros(1000,10);
  Matrix *grad_w1_ms = zeros(784,1000);
  Matrix *grad_w2_ms = zeros(1000,10);
  Matrix *grad_w2 = empty(1000,10);
  Matrix *grad_w1 = empty(784,1000);
  float error = 0;
  float cv_size = 0.1428571f;

  BatchAllocator b = BatchAllocator(X, y,  cv_size, 128, 512);

  clock_t t1,t2;
  t1=clock();
  //code goes here
  int epochs  = 100;
  gpu.tick();
  float learning_rate = 0.003;
  //size_t free = 0;
  //size_t total = 0;
  float momentum = 0.5;
  for(int EPOCH = 0; EPOCH < epochs; EPOCH++)
  {
	  std::cout << "EPOCH: " << EPOCH + 1 << std::endl;
	  //cudaMemGetInfo(&free, &total);
	  //std::cout << free << std::endl;
	  momentum += 0.01;
	  if(momentum > 0.95) momentum = 0.95;
	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {

		  b.allocate_next_batch_async();

		  //nesterov updates
		  scalarMul(m1,momentum,m1);
		  scalarMul(m2,momentum,m2);
		  add(w1,m1,w1);
		  add(w2,m1,w2);

		  Matrix *d0 = gpu.dropout(b.CURRENT_BATCH,0.2);
		  //print_gpu_matrix(w1);
		  Matrix *z1 = gpu.dot(d0, w1);
		  logistic(z1, z1);
		  Matrix *d1 = gpu.dropout(z1,0.6);
		  Matrix *a2 = gpu.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *t = create_t_matrix(b.CURRENT_BATCH_Y,10);

		  //backprop
		  Matrix *e1 = sub(out, t);
		  Matrix *e2 = gpu.dotT(e1, w2);
		  gpu.Tdot(z1,e1,grad_w2);
		  logisticGrad(z1,z1);
		  mul(e2,z1,e2);
		  gpu.Tdot(b.CURRENT_BATCH,e2,grad_w1);

		  RMSprop_with_nesterov_weight_update(ms1,grad_w1,w1,m1,0.9f,learning_rate,b.CURRENT_BATCH->rows);
		  RMSprop_with_nesterov_weight_update(ms2,grad_w2,w2,m2,0.9f,learning_rate,b.CURRENT_BATCH->rows);

		  cudaFree(e1->data);
		  cudaFree(e2->data);
		  cudaFree(z1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(t->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);

		  b.replace_current_batch_with_next();

	  }


	  //Matrix *sum_value = sum(w1);
	  //std::cout << "weight 1 Sum: " << to_host(sum_value)->data[0] << std::endl;

	  error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {
		  b.allocate_next_batch_async();

		  Matrix *a1 = gpu.dot(b.CURRENT_BATCH,w1);

		  logistic(a1, a1);
		  Matrix *a2 = gpu.dot(a1,w2);

		  Matrix *out = softmax(a2);


		  Matrix *result = argmax(out);

		  Matrix *eq = equal(result,b.CURRENT_BATCH_Y);
		  Matrix *sum_mat = sum(eq);
		  float sum_value = to_host(sum_mat)->data[0];

		  //std::cout << "Error count: " << 128.0f - sum_value << std::endl;
		  error += (b.CURRENT_BATCH->rows - sum_value);


		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(sum_mat->data);

		  b.replace_current_batch_with_next();
	  }


	  std::cout << "Train error: " << error/((1.0f - cv_size)*70000.0f)<< std::endl;


	  error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES_CV; i++)
	  {
		  b.allocate_next_cv_batch_async();
		  Matrix *a1 = gpu.dot(b.CURRENT_BATCH_CV,w1);

		  logistic(a1, a1);
		  Matrix *a2 = gpu.dot(a1,w2);

		  Matrix *out = softmax(a2);

		  Matrix *result = argmax(out);

		  Matrix *eq = equal(result,b.CURRENT_BATCH_CV_Y);
		  Matrix *sum_mat = sum(eq);
		  float sum_value = to_host(sum_mat)->data[0];

		  //std::cout << "Error count: " << gpu.m_total_batches_cv - sum_value << std::endl;
		  error += (b.CURRENT_BATCH_CV->rows  - sum_value);


		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(sum_mat->data);

		  b.replace_current_cv_batch_with_next();
	  }

	  std::cout << "Cross validation error: " << error/(cv_size*70000) << std::endl;


  }

  cudaThreadSynchronize();
  t2=clock();
  float diff ((float)t2-(float)t1);
  float mseconds = (diff / CLOCKS_PER_SEC)/1000;
  std::cout<<mseconds<<std::endl;
  gpu.tock();

  b.finish_batch_allocator();

  //gpu.tock("batch replace");
  //gpu.tock("async batch allocate");
  //gpu.tock("feedforward");


  printf("Finished!\n");
}

void MPI_benchmark_P2P(int argc, char *argv[])
{
	char name[100];
    int myrank, length, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Get_processor_name(name, &length);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	int local_rank = myrank % 4;

	int gpus;
	cudaGetDeviceCount(&gpus);
	int mygpu_id;
	int your_gpu_id;
	if(myrank == 0)
	{
		mygpu_id = 0;
		if(gpus > 1)
			your_gpu_id = 1;
		else
			your_gpu_id = 0;

		MPI_Send(&your_gpu_id,1, MPI_INT,1,0,MPI_COMM_WORLD);
	}
	else
	{
		MPI_Recv(&mygpu_id,1,MPI_INT,myrank-1,0,MPI_COMM_WORLD,&status);
		if(gpus > mygpu_id+1)
			your_gpu_id = mygpu_id + 1;
		else
			your_gpu_id = 0;
		if(myrank < size-1)
			MPI_Send(&your_gpu_id,1, MPI_INT,myrank+1,0,MPI_COMM_WORLD);
	}

	cudaSetDevice(mygpu_id);


		int batch_size = 128;
		int inner_dim = 10000;
		int outer_dim = 15000;

		ClusterNet gpu = ClusterNet();
		Matrix *A = gpu.rand(batch_size,inner_dim);
		Matrix *B = gpu.rand(inner_dim,outer_dim);
		Matrix *out = empty(batch_size,outer_dim);
		Matrix *rec = empty(batch_size,outer_dim);

		Matrix *A1 = gpu.rand(batch_size/2,inner_dim);
		Matrix *B1 = gpu.rand(inner_dim,outer_dim);
		Matrix *rec1 = empty(batch_size/2,outer_dim);
		Matrix *out1 = empty(batch_size/2,outer_dim);

		Matrix *A2 = gpu.rand(batch_size,inner_dim);
		Matrix *B2 = gpu.rand(inner_dim,outer_dim/2);
		Matrix *rec2 = empty(batch_size,outer_dim/2);
		Matrix *out2 = empty(batch_size,outer_dim/2);


		gpu.tick("Direct compute");
	    for(int i = 0; i< 100; i++)
	    {
	      gpu.dot(A,B, out);
		//add(A, B, out);
	    }
	    gpu.tock("Direct compute");

		gpu.tick("partial batch direct compute");
	    for(int i = 0; i< 100; i++)
	    {
	      gpu.dot(A1,B1, out1);
		//add(A, B, out);
	    }
	    gpu.tock("partial batch direct compute");

		gpu.tick("partial units direct compute");
	    for(int i = 0; i< 100; i++)
	    {
	      gpu.dot(A2,B2, out2);
		//add(A, B, out);
	    }
	    gpu.tock("partial units direct compute");




		gpu.tick("PCIe transfer");
		for(int i = 0; i< 100; i++)
		{
			if(local_rank == 0 && gpus > 1)
			{
				MPI_Send(out->data, out->size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
			}
			else if(local_rank == 1 && gpus > 1)
			{
				//add(A2,B, out);
				MPI_Recv(rec->data, rec->size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
			}
		}
		gpu.tock("PCIe transfer");


		gpu.tick("PCIe dot");
		for(int i = 0; i< 100; i++)
		{
			if(local_rank == 0 && gpus > 1)
			{
				gpu.dot(A2,B2,out2);
				MPI_Send(out1->data, out1->size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
			}
			else if(local_rank == 1 && gpus > 1)
			{
				gpu.dot(A2,B2,out2);
				MPI_Recv(rec1->data, rec1->size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
				vStack(out2,rec2,rec);
			}
		}
		gpu.tock("PCIe dot");



		gpu.tick("RDMA transfer");
		for(int i = 0; i< 100; i++)
		{
			if(myrank == 0)
			{
				MPI_Send(out->data, out->size, MPI_FLOAT, 3, 100, MPI_COMM_WORLD);
			}
			else if(myrank == 3)
			{
				//add(A2,B, out);
				MPI_Recv(rec->data, rec->size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
			}
		}
		gpu.tock("RDMA transfer");


		gpu.tick("RDMA dot");
		for(int i = 0; i< 100; i++)
		{
			if(myrank == 0)
			{
				gpu.dot(A2,B2,out2);
				MPI_Send(out->data, out->size, MPI_FLOAT, 3, 100, MPI_COMM_WORLD);
			}
			else if(myrank == 3)
			{
				//add(A2,B, out);
				gpu.dot(A2,B2,out2);
				MPI_Recv(rec->data, rec->size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
				vStack(out2,rec2,rec);
			}
		}
		gpu.tock("RDMA dot");








	MPI_Finalize();



}


void MPI_benchmark(int argc, char *argv[])
{
    int myrank;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    ClusterNet gpu = ClusterNet();
    int batch_rows = 128;
    int w_in = 10000;
    int w_out = 8000;

    //dot
    Matrix *B = gpu.rand(w_in,w_out);
    Matrix *A = gpu.rand(batch_rows,w_in);
    assert(test_matrix(A,batch_rows,w_in));
    assert(test_matrix(B,w_in,w_out));
    Matrix *out = empty(batch_rows, w_out);

    Matrix *B1 = gpu.rand(w_in,w_out/2);
    Matrix *B2 = gpu.rand(w_in,w_out/2);
    Matrix *D = empty(batch_rows,w_out/2);
    Matrix *A1 = gpu.rand(batch_rows/2,w_in);
    Matrix *big_out = gpu.rand(batch_rows/2,w_out);
    Matrix *grand_out = empty(batch_rows, w_out);

    Matrix *C = gpu.rand(batch_rows/2,w_in);
    Matrix *C_out = empty(batch_rows/2,w_out);

    Matrix *E = gpu.rand(batch_rows/4,w_in);
    Matrix *E_out = empty(batch_rows/4,w_out);
    Matrix *E_merge = empty(batch_rows/2,w_out);
    Matrix *E_merge2 = empty(batch_rows/2,w_out);

    //add

    /*
    B = gpu.rand(w_in,w_out);
    A = gpu.rand(w_in,w_out);
    out = empty(w_in, w_out);
    A1 = gpu.rand(w_in/2,w_out);
    Matrix *A2 = gpu.rand(w_in/2,w_out);
    D = empty(w_in/2,w_out);
*/

    cudaEvent_t* startstop = tick();
    for(int i = 0; i< 100; i++)
    {
      gpu.dot(A,B, out);
	//add(A, B, out);
    }
    printf("Direct compute:\n");
    tock(startstop);


    out = empty(batch_rows,w_out/2);
    Matrix *out2 = empty(batch_rows,w_out/2);
    startstop = tick();
    for(int i = 0; i< 100; i++)
    {
      gpu.dot(A,B1, out);
      gpu.dot(A,B2, out2);
      vStack(out,out2,grand_out);
    }
    printf("Direct compute x2:\n");
    tock(startstop);

    Matrix *mergemat = empty(batch_rows, w_out);
    out = empty(batch_rows,w_out/2);
    startstop = tick();
    //out = empty(w_in/2,w_out);
    for(int i = 0; i < 100; i++)
    {
	    if(myrank == 0)
	    {
		gpu.dot(A,B1, out);
    		//add(A1, B,out);
		MPI_Send(out->data, out->size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
	    }
	    else
	    {
		gpu.dot(A,B2, out);
		//add(A2,B, out);
	 	MPI_Recv(D->data, D->size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
                vStack(out,D, mergemat);
	    }

    }

    if(myrank == 1)
    {
      printf("GPUDirect RDMA:\n");
      tock(startstop);
    }

    out = empty(batch_rows/2,w_out);
    startstop = tick();
    gpu.tick("aa");
    //out = empty(w_in/2,w_out);
    for(int i = 0; i < 100; i++)
    {
    	gpu.tick("dot");
		gpu.dot(C,B, out);
		gpu.tick("dot");

	    if(myrank == 0)
	    {
    		//add(A1, B,out);
		gpu.tick("send");
		MPI_Send(out->data, out->size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
		gpu.tick("send");
	    }
	    else
	    {
		//add(A2,B, out);
		gpu.tick("receive");
	 	MPI_Recv(C_out->data, C_out->size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
                vStack(out,C_out, grand_out);
                gpu.tick("receive");
	    }

	    if(myrank == 1)
	    {
    		//add(A1, B,out);
		gpu.tick("send");
		MPI_Send(out->data, out->size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD);
		gpu.tick("send");
	    }
	    else
	    {
		//add(A2,B, out);
		gpu.tick("receive");
	 	MPI_Recv(C_out->data, C_out->size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD, &status);
                vStack(out,C_out, grand_out);
                gpu.tick("receive");
	    }

    }

    gpu.tock("dot");

    if(myrank == 1)
    {
      printf("GPUDirect RDMA batch:\n");
      tock(startstop);

      gpu.tock("receive");
      gpu.tock("aa");
    }
    else
    {

        gpu.tock("send");
    }










    MPI_Finalize();

}



void dotMPI_test(int argc, char *argv[])
{
	ClusterNet gpu = ClusterNet(argc, argv, 123465);
	Matrix *A = gpu.rand(128,1000);
	Matrix *B = gpu.rand(1000,400);

	gpu.tick("dot mpi batch");
	for(int i = 0; i < 100; i++)
	{
		gpu.dotMPI_batchSlice(A,B);
	}
	gpu.tock("dot mpi batch");



	gpu.tick("dot mpi unit");
	for(int i = 0; i < 100; i++)
	{
		gpu.dotMPI_unitSlice(A,B);
	}
	gpu.tock("dot mpi unit");

	printf("My rank: %i\n",gpu.m_myrank);
	gpu.benchmark_dot();



	gpu.tick("dot normal");
	for(int i = 0; i < 100; i++)
	{
		gpu.dot(A,B);
	}
	gpu.tock("dot normal");



	gpu.shutdown_MPI();
}



int main(int argc, char *argv[])
{

	//MPI_benchmark_P2P(argc, argv);

//run_neural_network();

/*

	Matrix *X = read_hdf5("/home/tim/mnist_full_X.hdf5");
	Matrix *y = read_hdf5("/home/tim/mnist_full_y.hdf5");
	std::vector<int> layers;
	layers.push_back(1000);
	DeepNeuralNetwork net = DeepNeuralNetwork(X,y,0.20,layers,Classification);
	net.train();
	*/


	dotMPI_test(argc, argv);




}




