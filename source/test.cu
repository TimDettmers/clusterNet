#include <stdio.h>
#include <cublas_v2.h>
#include <util.cuh>
#include <basicOps.cuh>
#include <mpi.h>
#include <cuda.h>
#include <assert.h>
#include <util.cuh>
#include <clusterNet.cuh>
#include <time.h>

void run_neural_network()
{
  Matrix X = read_csv("/home/tim/Downloads/mnist_full_X.csv");
  Matrix y = read_csv("/home/tim/Downloads/mnist_full_y.csv");
  printf("read MNIST\n");

  ClusterNet gpu = ClusterNet();

  Matrix result;
  Matrix w1 = gpu.rand(784,1000);
  Matrix w2 = gpu.rand(1000,10);

  printf("init batch allocator\n");
  gpu.init_batch_allocator(X, y, 124);

  clock_t t1,t2;
  t1=clock();
  //code goes here

  gpu.tick();
  for(int i = 0; i < gpu.m_total_batches; i++)
  {
	  gpu.allocate_next_batch_async();


	  result = gpu.dot(gpu.m_current_batch_X,w1);
	  result = gpuExp(result);
	  result = gpu.dot(result,w2);

	  gpu.replace_current_batch_with_next();

  }
  cudaThreadSynchronize();
  t2=clock();
  float diff ((float)t2-(float)t1);
  float mseconds = (diff / CLOCKS_PER_SEC)/1000;
  std::cout<<mseconds<<std::endl;
  gpu.tock();

  gpu.finish_batch_allocator();
  //gpu.tock("batch replace");
  //gpu.tock("async batch allocate");
  //gpu.tock("feedforward");


  printf("Finished!\n");
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
    Matrix B = gpu.rand(w_in,w_out);
    Matrix A = gpu.rand(batch_rows,w_in);
    assert(test_matrix(A,batch_rows,w_in));
    assert(test_matrix(B,w_in,w_out));
    Matrix out = empty(batch_rows, w_out);

    Matrix B1 = gpu.rand(w_in,w_out/2);
    Matrix B2 = gpu.rand(w_in,w_out/2);
    Matrix D = empty(batch_rows,w_out/2);
    Matrix A1 = gpu.rand(batch_rows/2,w_in);
    Matrix big_out = gpu.rand(batch_rows/2,w_out);
    Matrix grand_out = empty(batch_rows, w_out);

    Matrix C = gpu.rand(batch_rows/2,w_in);
    Matrix C_out = empty(batch_rows/2,w_out);

    Matrix E = gpu.rand(batch_rows/4,w_in);
    Matrix E_out = empty(batch_rows/4,w_out);
    Matrix E_merge = empty(batch_rows/2,w_out);
    Matrix E_merge2 = empty(batch_rows/2,w_out);

    //add

    /*
    B = gpu.rand(w_in,w_out);
    A = gpu.rand(w_in,w_out);
    out = empty(w_in, w_out);
    A1 = gpu.rand(w_in/2,w_out);
    Matrix A2 = gpu.rand(w_in/2,w_out);
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
    Matrix out2 = empty(batch_rows,w_out/2);
    startstop = tick();
    for(int i = 0; i< 100; i++)
    {
      gpu.dot(A,B1, out);
      gpu.dot(A,B2, out2);
      vStack(out,out2,grand_out);
    }
    printf("Direct compute x2:\n");
    tock(startstop);

    Matrix mergemat = empty(batch_rows, w_out);
    out = empty(batch_rows,w_out/2);
    startstop = tick();
    //out = empty(w_in/2,w_out);
    for(int i = 0; i < 100; i++)
    {
	    if(myrank == 0)
	    {
		gpu.dot(A,B1, out);
    		//add(A1, B,out);
		MPI_Send(out.data, out.size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
	    }
	    else
	    {
		gpu.dot(A,B2, out);
		//add(A2,B, out);
	 	MPI_Recv(D.data, D.size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
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
		MPI_Send(out.data, out.size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
		gpu.tick("send");
	    }
	    else
	    {
		//add(A2,B, out);
		gpu.tick("receive");
	 	MPI_Recv(C_out.data, C_out.size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
                vStack(out,C_out, grand_out);
                gpu.tick("receive");
	    }

	    if(myrank == 1)
	    {
    		//add(A1, B,out);
		gpu.tick("send");
		MPI_Send(out.data, out.size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD);
		gpu.tick("send");
	    }
	    else
	    {
		//add(A2,B, out);
		gpu.tick("receive");
	 	MPI_Recv(C_out.data, C_out.size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD, &status);
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
	Matrix A = gpu.rand(128,1000);
	Matrix B = gpu.rand(1000,400);

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

	printf("My rank: %i\n",gpu.m_rank);
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

  //MPI_benchmark(argc, argv);


	run_neural_network();





}



