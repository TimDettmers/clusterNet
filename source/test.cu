#include <stdio.h>
#include <cublas_v2.h>
#include <util.cuh>
#include <basicOps.cuh>
#include <mpi.h>
#include <cuda.h>
#include <assert.h>
#include <util.cuh>
#include <clusterNet.cuh>


void run_neural_network()
{
  Matrix X = read_csv("/home/tim/Downloads/mnist_full_X.csv");
  Matrix y = read_csv("/home/tim/Downloads/mnist_full_y.csv");

  //w1 = gpu.rand(784,1000);
  //w2 = gpu.rand(1000,10);

  printf("Finished!");
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







int main(int argc, char *argv[])
{

  //MPI_benchmark(argc, argv);

	/*
	ClusterNet gpu = ClusterNet();
	Matrix A = gpu.rand(128,10000);
	Matrix B = gpu.rand(10000,8000);

    int m_rank = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
	MPI_Status status;


	for(int i = 0; i < 100; i++)
	{
		if(m_rank == 0)
		{
			Matrix out = gpu.rand(64,8000);
			gpu.tick("MPI test send");
			MPI_Send(out.data, out.size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
			gpu.tick("MPI test send");
		}
		if(m_rank == 1)
		{
			Matrix out_recv = empty(64,8000);
			gpu.tick("MPI test receive");
			MPI_Recv(out_recv.data, out_recv.size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
			gpu.tick("MPI test receive");
		}
	}

	for(int i = 0; i < 100; i++)
	{
		gpu.tick("to beat");
		gpu.dot(A,B);
		gpu.tick("to beat");
	}


	if(m_rank == 0)
	{
		gpu.tock("MPI test send");
	}
	else
	{
		gpu.tock("MPI test receive");
	}

	gpu.tock("to beat");

	MPI_Finalize();

*/



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



