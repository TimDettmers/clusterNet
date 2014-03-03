#include <stdio.h>
#include <util.cuh>
#include <basicOps.cuh>
#include <assert.h>
#include <clusterNet.h>
#include <batchAllocator.h>
#include <string>


void run_miniMNIST_test(int argc, char *argv[])
{

	// Tests RMSprop with weight updates, logistic grad.
	// Additionally tests the interplay between different functions.

	char buff[1024];
	ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
	std::string path = std::string(buff);
	replace(path,"/build/testSuite.out","/tests/");

	Matrix *X = read_hdf5((path + "/mnist_mini_X.hdf5").c_str());
	Matrix *y = read_hdf5((path+ "/mnist_mini_y.hdf5").c_str());

	ClusterNet gpu = ClusterNet(12345);

	Matrix *w1 = scalarMul(gpu.rand(784,1000),0.4*sqrt(6.0f/(784.0+1000.0)));
	Matrix *w2 = scalarMul(gpu.rand(1000,10),0.4*sqrt(6.0f/(10.0+1000.0)));
	Matrix *m1 = zeros(784,1000);
	Matrix *m2 = zeros(1000,10);
	Matrix *ms1 = zeros(784,1000);
	Matrix *ms2 = zeros(1000,10);
	Matrix *grad_w1_ms = zeros(784,1000);
	Matrix *grad_w2_ms = zeros(1000,10);
	Matrix *grad_w2 = empty(1000,10);
	Matrix *grad_w1 = empty(784,1000);
	float cv_error = 0.0f;
	float train_error = 0.0f;

	BatchAllocator b = BatchAllocator(X, y, 0.2, 32, 64);
	int epochs  = 10;
	float learning_rate = 0.003;
	float momentum = 0.5;
	for(int EPOCH = 1; EPOCH < epochs; EPOCH++)
	{
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

		  Matrix *d0 = gpu.dropout(b.m_current_batch_X,0.2);
		  //print_gpu_matrix(w1);
		  Matrix *z1 = gpu.dot(d0, w1);
		  logistic(z1, z1);
		  Matrix *d1 = gpu.dropout(z1,0.6);
		  Matrix *a2 = gpu.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *t = create_t_matrix(b.m_current_batch_y,10);

		  //backprop
		  Matrix *e1 = sub(out, t);
		  Matrix *e2 = gpu.dotT(e1, w2);
		  gpu.Tdot(z1,e1,grad_w2);
		  logisticGrad(z1,z1);
		  mul(e2,z1,e2);
		  gpu.Tdot(b.m_current_batch_X,e2,grad_w1);

		  RMSprop_with_nesterov_weight_update(ms1,grad_w1,w1,m1,0.9f,learning_rate,b.m_current_batch_X->rows);
		  RMSprop_with_nesterov_weight_update(ms2,grad_w2,w2,m2,0.9f,learning_rate,b.m_current_batch_X->rows);

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

	  train_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {
		  b.allocate_next_batch_async();

		  Matrix *a1 = gpu.dot(b.m_current_batch_X,w1);

		  logistic(a1, a1);
		  Matrix *a2 = gpu.dot(a1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b.m_current_batch_y);
		  Matrix *sum_mat = sum(eq);
		  float sum_value = to_host(sum_mat)->data[0];

		  train_error += (b.m_current_batch_X->rows - sum_value)/ (1.0f * b.m_current_batch_X->rows *b.TOTAL_BATCHES) ;

		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(sum_mat->data);

		  b.replace_current_batch_with_next();
	  }


	  //std::cout << "Train error: " << train_error << std::endl;


	  cv_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES_CV; i++)
	  {
		  b.allocate_next_cv_batch_async();

		  Matrix *a1 = gpu.dot(b.m_current_batch_cv_X,w1);
		  logistic(a1, a1);
		  Matrix *a2 = gpu.dot(a1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b.m_current_batch_cv_y);
		  Matrix *sum_mat = sum(eq);
		  float sum_value = to_host(sum_mat)->data[0];

		  cv_error += (b.m_current_batch_cv_X->rows  - sum_value)/ (1.0f * b.m_current_batch_cv_X->rows *b.TOTAL_BATCHES_CV) ;

		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(sum_mat->data);

		  b.replace_current_cv_batch_with_next();
	  }

	  //std::cout << "Cross validation error: " << cv_error << std::endl;


	}

	assert(test_eq(train_error,0.0f,"mini-MNIST train error 10 epochs."));
	ASSERT(cv_error < 0.16, "mini-MNIST train error 10 epochs.");

	b.finish_batch_allocator();


}
