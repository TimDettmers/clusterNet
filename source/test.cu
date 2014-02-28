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

void run_neural_network()
{
  Matrix *X = read_csv("/home/tim/Downloads/mnist_full_X.csv");
  Matrix *y = read_csv("/home/tim/Downloads/mnist_full_y.csv");

  ClusterNet gpu = ClusterNet(12345);

  Matrix *w1 = scalarMul(gpu.rand(784,1000),0.4*sqrt(6.0f/(784.0+1000.0)));
  Matrix *w2 = scalarMul(gpu.rand(1000,10),0.4*sqrt(6.0f/(10.0+1000.0)));
  Matrix *grad1 = empty(1000,10);
  Matrix *grad2 = empty(784,1000);
  float error = 0;

  std::cout << "size: " << X->rows << std::endl;

  gpu.init_batch_allocator(X, y, 0.2, 128, 512);

  clock_t t1,t2;
  t1=clock();
  //code goes here
  int epochs  = 100;
  gpu.tick();
  float learning_rate = 0.1;
  //size_t free = 0;
  //size_t total = 0;

  for(int EPOCH = 1; EPOCH < epochs; EPOCH++)
  {

	  std::cout << "EPOCH: " << EPOCH << std::endl;
	  //cudaMemGetInfo(&free, &total);
	  //std::cout << free << std::endl;

	  for(int i = 0; i < gpu.TOTAL_BATCHES; i++)
	  {
		  gpu.allocate_next_batch_async();

		  //std::cout << "m_current_batch_X.rows: " << gpu.m_current_batch_X->rows << std::endl;
		  //std::cout << "m_current_batch_X.cols: " << gpu.m_current_batch_X->cols << std::endl;
		  Matrix *z1 = gpu.dot(gpu.m_current_batch_X,w1);
		  logistic(z1, z1);
		  Matrix *a2 = gpu.dot(z1,w2);
		  //logistic(a2, a2);
		  Matrix *out = softmax(a2);

		  //std::cout << "batch: " << i << std::endl;
		  //std::cout << "out" << std::endl;
		  //std::cout << "-----------------------" << std::endl;
		  //print_gpu_matrix(out);


		  Matrix *z1_T = T(z1);
		  Matrix *X_T = T(gpu.m_current_batch_X);
		  Matrix *w2_T = T(w2);
		  Matrix *t = create_t_matrix(gpu.m_current_batch_y,10);

		  //backprop
		  Matrix *e1 = sub(out, t);
		  Matrix *e2 = gpu.dot(e1, w2_T);
		  logisticGrad(z1,z1);
		  mul(e2,z1,e2);
		  gpu.dot(z1_T,e1,grad1);
		  gpu.dot(X_T,e2,grad2);


		  scalarMul(grad1,learning_rate/(float)gpu.m_current_batch_X->rows,grad1);
		  scalarMul(grad2,learning_rate/(float)gpu.m_current_batch_X->rows,grad2);
		  sub(w2,grad1,w2);
		  sub(w1,grad2,w1);


		  cudaFree(e1->data);
		  cudaFree(e2->data);
		  cudaFree(z1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(z1_T->data);
		  cudaFree(X_T->data);
		  cudaFree(w2_T->data);
		  cudaFree(t->data);

		  gpu.replace_current_batch_with_next();

	  }


	  //Matrix *sum_value = sum(w1);
	  //std::cout << "weight 1 Sum: " << to_host(sum_value)->data[0] << std::endl;

	  error = 0;
	  for(int i = 0; i < gpu.TOTAL_BATCHES; i++)
	  {
		  gpu.allocate_next_batch_async();

		  Matrix *a1 = gpu.dot(gpu.m_current_batch_X,w1);

		  logistic(a1, a1);
		  Matrix *a2 = gpu.dot(a1,w2);

		  Matrix *out = softmax(a2);


		  Matrix *result = argmax(out);


		  /*
		  std::cout << "y" << std::endl;
		  std::cout << "-----------------------" << std::endl;
		  print_gpu_matrix(gpu.m_current_batch_y);
		  */
		  Matrix *eq = equal(result,gpu.m_current_batch_y);
		  Matrix *sum_mat = sum(eq);
		  float sum_value = to_host(sum_mat)->data[0];

		  //std::cout << "Error count: " << 128.0f - sum_value << std::endl;
		  error += (128.0f - sum_value)/ (128.0f*gpu.TOTAL_BATCHES) ;


		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(sum_mat->data);

		  gpu.replace_current_batch_with_next();
	  }


	  std::cout << "Train error: " << error << std::endl;


	  error = 0;
	  for(int i = 0; i < gpu.TOTAL_BATCHES_CV; i++)
	  {
		  //std::cout << "i: " << i << std::endl;
		  gpu.allocate_next_cv_batch_async();
		  //std::cout << "batch size: " << gpu.m_current_batch_cv_X->rows << std::endl;
		  //std::cout << "batches : " << gpu.m_total_batches_cv << std::endl;

		  Matrix *a1 = gpu.dot(gpu.m_current_batch_cv_X,w1);

		  logistic(a1, a1);
		  Matrix *a2 = gpu.dot(a1,w2);

		  Matrix *out = softmax(a2);

		  Matrix *result = argmax(out);

		  Matrix *eq = equal(result,gpu.m_current_batch_cv_y);
		  Matrix *sum_mat = sum(eq);
		  float sum_value = to_host(sum_mat)->data[0];

		  //std::cout << "Error count: " << gpu.m_total_batches_cv - sum_value << std::endl;
		  error += (512.0f - sum_value)/ (512.0f*gpu.TOTAL_BATCHES_CV) ;


		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(sum_mat->data);

		  gpu.replace_current_cv_batch_with_next();
	  }

	  std::cout << "Cross validation error: " << error << std::endl;



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



