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
#include <WikiMaxoutNet.h>
#include <WikiMaxoutNet_PCIe.h>
#include <WikiMaxoutNet_PCIe2.h>
#include <WikiNetDist.h>
#include <Layer.h>
#include <time.h>

using std::cout;
using std::endl;





void run_neural_network()
{
  Matrix *X = read_hdf5("/home/tim/mnist_full_X.hdf5");
  Matrix *y = read_hdf5("/home/tim/mnist_full_y.hdf5");

  ClusterNet gpu = ClusterNet(12345);

  cout << X->rows << endl;

  int hidden_size = 1024;


  Matrix *w1 = gpu.sparseInitWeight(784,hidden_size);
  Matrix *w2 = gpu.sparseInitWeight(hidden_size,10);
  Matrix *m1 = zeros(784,hidden_size);
  Matrix *m2 = zeros(hidden_size,10);
  Matrix *ms1 = zeros(784,hidden_size);
  Matrix *ms2 = zeros(hidden_size,10);
  Matrix *grad_w1_ms = zeros(784,hidden_size);
  Matrix *grad_w2_ms = zeros(hidden_size,10);
  Matrix *grad_w2 = empty(hidden_size,10);
  Matrix *grad_w1 = empty(784,hidden_size);
  float cv_error = 0;
  float cv_size = 0.1428571f;
  float train_error = 0.0f;

  BatchAllocator b = BatchAllocator();
  b.init(X, y,  cv_size, 128, 512);

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
		  add(w2,m2,w2);

		  Matrix *d0 = gpu.dropout(b.CURRENT_BATCH,0.2);
		  Matrix *z1 = gpu.dot(d0, w1);
		  logistic(z1, z1);
		  Matrix *d1 = gpu.dropout(z1,0.5);
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

		  b.allocate_next_batch_async();

		  RMSprop_with_momentum_weight_update(ms1,grad_w1,w1,m1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_momentum_weight_update(ms2,grad_w2,w2,m2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);

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

	  train_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {

		  b.broadcast_batch_to_processes();

		  //Matrix *d0 = scalarMul(b.CURRENT_BATCH,0.8);
		  Matrix *a1 = gpu.dot(b.CURRENT_BATCH,w1);
		  logistic(a1, a1);
		  //Matrix *d1 = scalarMul(a1,0.5);
		  Matrix *a2 = gpu.dot(a1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b.CURRENT_BATCH_Y);
		  b.allocate_next_batch_async();
		  float sum_value = sum(eq);

		  train_error += (b.CURRENT_BATCH->rows - sum_value)/ (1.0f * b.CURRENT_BATCH->rows *b.TOTAL_BATCHES) ;

		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  //cudaFree(d0->data);
		  //cudaFree(d1->data);

		  b.replace_current_batch_with_next();
	  }


	  std::cout << "Train error: " << train_error << std::endl;


	  cv_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES_CV; i++)
	  {
		  b.broadcast_batch_cv_to_processes();
		  Matrix *d0 = scalarMul(b.CURRENT_BATCH_CV,0.8);
		  Matrix *a1 = gpu.dot(d0,w1);
		  logistic(a1, a1);
		  Matrix *d1 = scalarMul(a1,0.5);
		  Matrix *a2 = gpu.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b.CURRENT_BATCH_CV_Y);
		  b.allocate_next_cv_batch_async();
		  float sum_value = sum(eq);

		  cv_error += (b.CURRENT_BATCH_CV->rows  - sum_value)/ (1.0f * b.CURRENT_BATCH_CV->rows *b.TOTAL_BATCHES_CV) ;

		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);

		  b.replace_current_cv_batch_with_next();
	  }

	  std::cout << "Cross validation error: " << cv_error << std::endl;


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


void run_maxout_network()
{

	cudaSetDevice(0);
    Matrix *X = read_hdf5("/home/tim/mnist_full_X.hdf5");
    Matrix *y = read_hdf5("/home/tim/mnist_full_y.hdf5");

  	ClusterNet gpus = ClusterNet(12345);

  	int hiddenunits = 512;
  	int maxout_Size = 8;
  	int batch_size = 128;

	Matrix *w1 = gpus.uniformSqrtWeight(784,hiddenunits);
	Matrix *w2 = gpus.uniformSqrtWeight(hiddenunits/maxout_Size,10);
	Matrix *b1 = zeros(1,hiddenunits);
	Matrix *b2 = zeros(1,10);
	Matrix *m1 = zeros(784,hiddenunits);
	Matrix *m2 = zeros(hiddenunits/maxout_Size,10);
	Matrix *mb1 = zeros(1,hiddenunits);
	Matrix *mb2 = zeros(1,10);
	Matrix *ms1 = zeros(784,hiddenunits);
	Matrix *ms2 = zeros(hiddenunits/maxout_Size,10);
	Matrix *msb1 = zeros(1,hiddenunits);
	Matrix *msb2 = zeros(1,10);
	Matrix *grad_w1 = zeros(784,hiddenunits);
	Matrix *grad_w2 = zeros(hiddenunits/maxout_Size,10);
	Matrix *grad_b1 = zeros(1,hiddenunits);
	Matrix *grad_b2 = zeros(1,10);


	float cv_error = 0.0f;
	float train_error = 0.0f;

	BatchAllocator b = BatchAllocator();
	b.init(X, y, 0.2, batch_size, 512);
	int epochs  = 1000;
	float learning_rate = 0.001;
	float momentum = 0.5;
	for(int EPOCH = 1; EPOCH < epochs; EPOCH++)
	{
	  cout << "EPOCH: " << EPOCH << endl;
	  //momentum += 0.01;
	  //if(momentum > 0.95) momentum = 0.95;
	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {
		  b.broadcast_batch_to_processes();

		  //nesterov updates
		  scalarMul(m1,momentum,m1);
		  scalarMul(m2,momentum,m2);
		  scalarMul(mb1,momentum,mb1);
		  scalarMul(mb2,momentum,mb2);
		  add(w1,m1,w1);
		  add(w2,m2,w2);
		  add(b1,mb1,b1);
		  add(b2,mb2,b2);


		  //feedforward
		  Matrix *d0 = gpus.dropout(b.CURRENT_BATCH,0.2);
		  Matrix *z1 = gpus.dot(d0, w1);
		  addMatrixVector(z1,b1,z1);
		  Matrix **a_paired = maxout(z1,maxout_Size);
		  Matrix *a1 = a_paired[0];
		  Matrix *a1_idx = a_paired[1];
		  Matrix *d1 = gpus.dropout(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  addMatrixVector(a2,b2,a2);
		  Matrix *out = softmax(a2);
		  Matrix *t = create_t_matrix(b.CURRENT_BATCH_Y,10);

		  b.allocate_next_batch_async();

		  //backprop
		  Matrix *e1 = sub(out, t);
		  Matrix *e2_partial = gpus.dotT(e1, w2);
		  Matrix *e2 = empty(b.CURRENT_BATCH->rows,e2_partial->cols*maxout_Size);
		  Matrix *aB = ones(1,b.CURRENT_BATCH->rows);


		  gpus.Tdot(a1,e1,grad_w2);
		  gpus.dot(aB,e1,grad_b2);
		  expand_to_maxout_grad(e2_partial, a1_idx,e2);
		  gpus.Tdot(b.CURRENT_BATCH,e2,grad_w1);
		  gpus.dot(aB,e2,grad_b1);

		  //weight updates
		  //RMSProp


		  RMSprop_with_momentum_weight_update(ms1,grad_w1,w1,m1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_momentum_weight_update(ms2,grad_w2,w2,m2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);

		  RMSprop_with_momentum_weight_update(msb1,grad_b1,b1,mb1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_momentum_weight_update(msb2,grad_b2,b2,mb2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);



/*
		  scalarMul(grad_w1,learning_rate/(float)b.CURRENT_BATCH->rows,grad_w1);
		  scalarMul(grad_w2,learning_rate/(float)b.CURRENT_BATCH->rows,grad_w2);
		  scalarMul(grad_b1,learning_rate/(float)b.CURRENT_BATCH->rows,grad_b1);
		  scalarMul(grad_b2,learning_rate/(float)b.CURRENT_BATCH->rows,grad_b2);



		  //classical momentum
		  scalarMul(m1,momentum,m1);
		  scalarMul(m2,momentum,m2);
		  scalarMul(mb1,momentum,mb1);
		  scalarMul(mb2,momentum,mb2);
		  sub(m1,grad_w1,m1);
		  sub(m2,grad_w2,m2);
		  sub(mb1,grad_b1,mb1);
		  sub(mb2,grad_b2,mb2);

		  add(w1,m1,w1);
		  add(w2,m2,w2);
		  add(b1,mb1,b1);
		  add(b2,mb2,b2);

		  */



		  /*
		  sub(w1,grad_w1,w1);
		  sub(w2,grad_w2,w2);
		  sub(b1,grad_b1,b1);
		  sub(b2,grad_b2,b2);
		  */



		  cudaFree(e1->data);
		  cudaFree(e2->data);
		  cudaFree(e2_partial->data);
		  cudaFree(z1->data);
		  cudaFree(a1->data);
		  cudaFree(a1_idx->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(t->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);
		  cudaFree(aB->data);
		  free(a_paired);

		  b.replace_current_batch_with_next();

	  }



	  train_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {

		  b.broadcast_batch_to_processes();

		  Matrix *d0 = scalarMul(b.CURRENT_BATCH,0.8);
		  Matrix *z1 = gpus.dot(d0,w1);
		  Matrix **a1_pair = maxout(z1,maxout_Size);
		  Matrix *a1 = a1_pair[0];
		  Matrix *d1 = scalarMul(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b.CURRENT_BATCH_Y);
		  b.allocate_next_batch_async();
		  float sum_value = sum(eq);

		  train_error += (b.CURRENT_BATCH->rows - sum_value)/ (1.0f * b.CURRENT_BATCH->rows *b.TOTAL_BATCHES) ;

		  cudaFree(z1->data);
		  cudaFree(a1->data);
		  cudaFree(a1_pair[1]->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);
		  free(a1_pair);

		  b.replace_current_batch_with_next();
	  }

	  std::cout << "MAXOUT Train error: " << train_error << std::endl;



	  cv_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES_CV; i++)
	  {
		  b.broadcast_batch_cv_to_processes();
		  Matrix *d0 = scalarMul(b.CURRENT_BATCH_CV,0.8);
		  Matrix *z1 = gpus.dot(d0,w1);
		  Matrix **a1_pair = maxout(z1,maxout_Size);
		  Matrix *a1 = a1_pair[0];
		  Matrix *d1 = scalarMul(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b.CURRENT_BATCH_CV_Y);
		  b.allocate_next_batch_async();
		  float sum_value = sum(eq);

		  cv_error += (b.CURRENT_BATCH_CV->rows  - sum_value)/ (1.0f * b.CURRENT_BATCH_CV->rows *b.TOTAL_BATCHES_CV) ;

		  cudaFree(z1->data);
		  cudaFree(a1->data);
		  cudaFree(a1_pair[1]->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);
		  free(a1_pair);

		  b.replace_current_cv_batch_with_next();
	  }

	  std::cout << "MAXOUT Cross validation error: " << cv_error << std::endl;

	}

}


void run_normal_net()
{
	cudaSetDevice(2);
    Matrix *X = read_hdf5("/home/tim/mnist_full_X.hdf5");
    Matrix *y = read_hdf5("/home/tim/mnist_full_y.hdf5");

  	ClusterNet gpus = ClusterNet(12345);

  	int hiddenunits = 1024;
  	int maxout_Size = 1;
  	int batch_size = 128;

	Matrix *w1 = gpus.uniformSqrtWeight(784,hiddenunits);
	Matrix *w2 = gpus.uniformSqrtWeight(hiddenunits/maxout_Size,10);
	Matrix *b1 = zeros(1,hiddenunits);
	Matrix *b2 = zeros(1,10);
	Matrix *m1 = zeros(784,hiddenunits);
	Matrix *m2 = zeros(hiddenunits/maxout_Size,10);
	Matrix *mb1 = zeros(1,hiddenunits);
	Matrix *mb2 = zeros(1,10);
	Matrix *ms1 = zeros(784,hiddenunits);
	Matrix *ms2 = zeros(hiddenunits/maxout_Size,10);
	Matrix *msb1 = zeros(1,hiddenunits);
	Matrix *msb2 = zeros(1,10);
	Matrix *grad_w1 = zeros(784,hiddenunits);
	Matrix *grad_w2 = zeros(hiddenunits/maxout_Size,10);
	Matrix *grad_b1 = zeros(1,hiddenunits);
	Matrix *grad_b2 = zeros(1,10);


	float cv_error = 0.0f;
	float train_error = 0.0f;

	BatchAllocator b = BatchAllocator();
	b.init(X, y, 0.4, batch_size, 512);
	int epochs  = 500;
	float learning_rate = 0.000001;
	float momentum = 0.5;
	for(int EPOCH = 1; EPOCH < epochs; EPOCH++)
	{
	  cout << "EPOCH: " << EPOCH << endl;
	  momentum += 0.01;
	  if(momentum > 0.95) momentum = 0.95;
	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {
		  b.broadcast_batch_to_processes();

		  //nesterov updates
		  scalarMul(m1,momentum,m1);
		  scalarMul(m2,momentum,m2);
		  scalarMul(mb1,momentum,mb1);
		  scalarMul(mb2,momentum,mb2);
		  add(w1,m1,w1);
		  add(w2,m2,w2);
		  add(b1,mb1,b1);
		  add(b2,mb2,b2);







		  //feedforward
		  Matrix *d0 = gpus.dropout(b.CURRENT_BATCH,0.2);
		  Matrix *z1 = gpus.dot(d0, w1);
		  addMatrixVector(z1,b1,z1);
		  Matrix *a1 = logistic(z1);
		  //Matrix *a1 = rectified_linear(z1);
		  Matrix *d1 = gpus.dropout(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  addMatrixVector(a2,b2,a2);
		  Matrix *out = softmax(a2);
		  Matrix *t = create_t_matrix(b.CURRENT_BATCH_Y,10);

		  b.allocate_next_batch_async();

		  //backprop
		  Matrix *e1 = sub(out, t);
		  Matrix *e2 = gpus.dotT(e1, w2);
		  Matrix *aB = ones(1,b.CURRENT_BATCH->rows);


		  gpus.Tdot(a1,e1,grad_w2);
		  gpus.dot(aB,e1,grad_b2);
		  //rectified_linear_derivative(a1,a1);
		  logisticGrad(a1,a1);
		  mul(e2,a1,e2);
		  gpus.Tdot(b.CURRENT_BATCH,e2,grad_w1);
		  gpus.dot(aB,e2,grad_b1);



		  /*
		  //about equal to momentum update + nesterov update -> momentum applyied to gradient+momentum better?
		  RMSprop_with_momentum_weight_update(ms1,grad_w1,w1,m1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_momentum_weight_update(ms2,grad_w2,w2,m2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);

		  RMSprop_with_momentum_weight_update(msb1,grad_b1,b1,mb1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_momentum_weight_update(msb2,grad_b2,b2,mb2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  */

		  /*
		  //slow and generally worse error, but sometimes better results in the end
		  RMSprop_with_momentum_update(ms1,grad_w1,w1,m1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_momentum_update(ms2,grad_w2,w2,m2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);

		  RMSprop_with_momentum_update(msb1,grad_b1,b1,mb1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_momentum_update(msb2,grad_b2,b2,mb2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  */




		  RMSprop_with_nesterov_weight_update(ms1,grad_w1,w1,m1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_nesterov_weight_update(ms2,grad_w2,w2,m2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);

		  RMSprop_with_nesterov_weight_update(msb1,grad_b1,b1,mb1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_nesterov_weight_update(msb2,grad_b2,b2,mb2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);


		  /*
		  //slower but equally good to nesterov momentum
		  RMSprop_with_weight_update(ms1,grad_w1,w1,m1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_weight_update(ms2,grad_w2,w2,m2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);

		  RMSprop_with_weight_update(msb1,grad_b1,b1,mb1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_weight_update(msb2,grad_b2,b2,mb2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  */
		  /*





		  scalarMul(grad_w1,learning_rate/(float)b.CURRENT_BATCH->rows,grad_w1);
		  scalarMul(grad_w2,learning_rate/(float)b.CURRENT_BATCH->rows,grad_w2);
		  scalarMul(grad_b1,learning_rate/(float)b.CURRENT_BATCH->rows,grad_b1);
		  scalarMul(grad_b2,learning_rate/(float)b.CURRENT_BATCH->rows,grad_b2);



		  //classical momentum
		  scalarMul(m1,momentum,m1);
		  scalarMul(m2,momentum,m2);
		  scalarMul(mb1,momentum,mb1);
		  scalarMul(mb2,momentum,mb2);
		  sub(m1,grad_w1,m1);
		  sub(m2,grad_w2,m2);
		  sub(mb1,grad_b1,mb1);
		  sub(mb2,grad_b2,mb2);


		  add(w1,m1,w1);
		  add(w2,m2,w2);
		  add(b1,mb1,b1);
		  add(b2,mb2,b2);
		  */




		  /*
		  sub(w1,grad_w1,w1);
		  sub(w2,grad_w2,w2);
		  sub(b1,grad_b1,b1);
		  sub(b2,grad_b2,b2);
		  */



		  cudaFree(e1->data);
		  cudaFree(e2->data);
		  cudaFree(z1->data);
		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(t->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);
		  cudaFree(aB->data);

		  b.replace_current_batch_with_next();

	  }



	  train_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {

		  b.broadcast_batch_to_processes();

		  Matrix *d0 = scalarMul(b.CURRENT_BATCH,0.8);
		  Matrix *z1 = gpus.dot(d0,w1);
		  Matrix *a1 = logistic(z1);
		  //Matrix *a1 = rectified_linear(z1);
		  Matrix *d1 = scalarMul(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b.CURRENT_BATCH_Y);
		  b.allocate_next_batch_async();
		  float sum_value = sum(eq);

		  train_error += (b.CURRENT_BATCH->rows - sum_value)/ (1.0f * b.CURRENT_BATCH->rows *b.TOTAL_BATCHES) ;

		  cudaFree(z1->data);
		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);

		  b.replace_current_batch_with_next();
	  }

	  std::cout << "MAXOUT Train error: " << train_error << std::endl;



	  cv_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES_CV; i++)
	  {
		  b.broadcast_batch_cv_to_processes();
		  Matrix *d0 = scalarMul(b.CURRENT_BATCH_CV,0.8);
		  Matrix *z1 = gpus.dot(d0,w1);
		  Matrix *a1 = logistic(z1);
		  //Matrix *a1 = rectified_linear(z1);
		  Matrix *d1 = scalarMul(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b.CURRENT_BATCH_CV_Y);
		  b.allocate_next_batch_async();
		  float sum_value = sum(eq);

		  cv_error += (b.CURRENT_BATCH_CV->rows  - sum_value)/ (1.0f * b.CURRENT_BATCH_CV->rows *b.TOTAL_BATCHES_CV) ;

		  cudaFree(z1->data);
		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);

		  b.replace_current_cv_batch_with_next();
	  }

	  std::cout << "MAXOUT Cross validation error: " << cv_error << std::endl;

	}

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

	/*
	ClusterNet gpu = ClusterNet(argc, argv, 123465);
	int inner = 2000;
	int outer = 1200;
	int batch_size = 128;


	int reduced_left = 128;
	int reduced_right = 400;
	Matrix *A = gpu.rand(batch_size,inner);
	Matrix *B = gpu.rand(inner,outer);
	Matrix *A1 = gpu.rand(reduced_left,inner);
	Matrix *B1 = gpu.rand(inner,reduced_right);

	Matrix *out = empty(batch_size,outer);
	Matrix *out1 = empty(reduced_left,reduced_right);
	Matrix *recv1 = empty(reduced_left,reduced_right);
	Matrix *recv2 = empty(reduced_left,reduced_right);
	Matrix *recv3 = empty(reduced_left,reduced_right);
	MPI_Status status;




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

	printf("My rank: %i\n",gpu.MYRANK);
	//gpu.benchmark_dot();



	gpu.tick("dot normal");
	for(int i = 0; i < 100; i++)
	{
		gpu.dot(A,B,out);
	}
	gpu.tock("dot normal");



	//std::vector<MPI_Request> requests;
	MPI_Request *requests = (MPI_Request*)malloc(sizeof(MPI_Request)*gpu.MPI_SIZE-1);
	MPI_Request request_send;
	std::vector<Matrix*> recv_buffer;
	for(int i = 0; i < gpu.MPI_SIZE-1; i++)
	{
		MPI_Request request;
		requests[i] = request;
	}




	int received_count = 0;
	for(int i = 0; i < 100; i++)
	{
		for(int i = 0; i < recv_buffer.size(); i++)
			cudaFree(recv_buffer[i]->data);
		recv_buffer.clear();
		out1 = empty(reduced_left,reduced_right);
		for(int i = 0; i < gpu.MPI_SIZE; i++)
		{
			recv_buffer.push_back(empty(reduced_left,reduced_right));
		}

		gpu.tick("all to all custom");
		//cout << "a1 rows" << A1->rows << endl;
		gpu.dot(A1,B1,out1);
		recv_buffer[gpu.MYRANK]= out1;
		for(int i = 0; i < gpu.MPI_SIZE; i++)
		{
			if(gpu.MYRANK == i) { continue; }
			MPI_Isend(out1->data, out1->size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, &request_send);
		}

		for(int i = 0; i < gpu.MPI_SIZE; i++)
		{
			if(gpu.MYRANK == i) { continue; }
			MPI_Irecv(recv1->data, recv1->size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, &requests[i]);

		}

		for(int i = 0; i < gpu.MPI_SIZE; i++)
		{
			if(gpu.MYRANK == i) { continue; }
			MPI_Wait(&requests[i],MPI_STATUS_IGNORE);
		}



		received_count = 0;
		while(received_count < gpu.MPI_SIZE-1)
		{
			for(int i = 0; i < gpu.MPI_SIZE; i++)
			{
				int received = 0;
				if(gpu.MYRANK == i) { continue; }
				MPI_Test(&requests[i],&received,&status);
				if(received == 1)
				{
					out1 = hStack(out1,recv1);
					received_count++;
				}
			}
		}

		gpu.tick("all to all custom");
	}
	gpu.tock("all to all custom");



	int destination = gpu.MYRANK + 1;
	int source = gpu.MYRANK - 1;
	if(destination == gpu.MPI_SIZE){destination = 0; }
	if(source < 0){ source = gpu.MPI_SIZE - 1;}
	for(int i = 0; i < 100; i++)
	{
		out1 = empty(reduced_left,reduced_right);
		recv1 = empty(reduced_left,reduced_right);
		gpu.tick("chain custom");
		gpu.dot(A1,B1,out1);
		for(int i = 0; i < gpu.MPI_SIZE-1; i++)
		{
			if(i == 0)
				MPI_Isend(out1->data, out1->size, MPI_FLOAT, destination, 100, MPI_COMM_WORLD, &request_send);
			else
				MPI_Isend(recv1->data, recv1->size, MPI_FLOAT, destination, 100, MPI_COMM_WORLD, &request_send);

			MPI_Recv(recv1->data, recv1->size, MPI_FLOAT, source, 100, MPI_COMM_WORLD, &status);

			//MPI_Wait(&requests[i],&status);
			out1 = hStack(out1,recv1);
		}
		gpu.tick("chain custom");
	}
	gpu.tock("chain custom");



	cout << gpu.MYRANK << endl;




	int matrix_idx = gpu.MYRANK;
	Matrix** arrOut = (Matrix**)malloc(sizeof(Matrix*)*gpu.MPI_SIZE);
	for(int i = 0; i < gpu.MPI_SIZE; i++)
		arrOut[i] = empty(reduced_left,reduced_right);

	float **h_arrA = (float**)malloc(sizeof(float*)*gpu.MPI_SIZE);
		for(int i = 0; i < gpu.MPI_SIZE; i++)
			h_arrA[i] = arrOut[i]->data;

	float **d_arrA;
	cudaMalloc((void**) &d_arrA,sizeof(float*)*gpu.MPI_SIZE);
	cudaMemcpy(d_arrA,h_arrA,sizeof(float*)*gpu.MPI_SIZE,cudaMemcpyDefault);

	gpu.tick("chain matrix array");
	for(int i = 0; i < 100; i++)
	{
		gpu.dot(A1,B1,arrOut[gpu.MYRANK]);
		matrix_idx = gpu.MYRANK;
		for(int i = 0; i < gpu.MPI_SIZE-1; i++)
		{
			MPI_Isend(arrOut[matrix_idx]->data, arrOut[matrix_idx]->size, MPI_FLOAT, destination, 100, MPI_COMM_WORLD, &request_send);
			matrix_idx = (matrix_idx - 1) < 0 ? gpu.MPI_SIZE-1 : (matrix_idx - 1);
			MPI_Irecv(arrOut[matrix_idx]->data, arrOut[matrix_idx]->size, MPI_FLOAT, source, 100, MPI_COMM_WORLD,&requests[i]);
		}


		MPI_Waitall(gpu.MPI_SIZE-1,requests,MPI_STATUSES_IGNORE);
		//hStackN(d_arrA,arrOut[0]->size, out,gpu.MPI_SIZE);

	}
	gpu.tock("chain matrix array");


	gpu.shutdown();



*/
}


void async_test(int argc, char *argv[])
{

	ClusterNet gpu = ClusterNet(argc,argv,1324);
	int rows = 512;
	int cols = 128;


	/*

	MPI_Request r = MPI_REQUEST_NULL;
	MPI_Request s = MPI_REQUEST_NULL;



	Matrix *a = gpu.rand(rows,cols);
	Matrix *b = zeros(rows,cols);

	if(gpu.MYRANK == 0)
	{
		MPI_Irecv(b->data,b->size,MPI_FLOAT,1,0,MPI_COMM_WORLD,&r);
		MPI_Isend(a->data,a->size,MPI_FLOAT,1,0,MPI_COMM_WORLD,&s);
	}
	else
	{
		MPI_Irecv(b->data,b->size,MPI_FLOAT,0,0,MPI_COMM_WORLD,&r);
		MPI_Isend(a->data,a->size,MPI_FLOAT,0,0,MPI_COMM_WORLD,&s);
	}

	MPI_Wait(&s,MPI_STATUS_IGNORE);
	MPI_Wait(&r,MPI_STATUS_IGNORE);


	gpu.tick("MPI");
	for(int i = 0; i < 100; i++)
	{
		if(gpu.MYRANK == 0)
		{
			MPI_Irecv(b->data,b->size,MPI_FLOAT,1,0,MPI_COMM_WORLD,&r);
			MPI_Isend(a->data,a->size,MPI_FLOAT,1,0,MPI_COMM_WORLD,&s);
		}
		else
		{
			MPI_Irecv(b->data,b->size,MPI_FLOAT,0,0,MPI_COMM_WORLD,&r);
			MPI_Isend(a->data,a->size,MPI_FLOAT,0,0,MPI_COMM_WORLD,&s);
		}

		MPI_Wait(&s,MPI_STATUS_IGNORE);
		MPI_Wait(&r,MPI_STATUS_IGNORE);
	}

	gpu.tock("MPI");
	*/





	if(gpu.MYRANK == 0)
	{
		cudaSetDevice(0);
		//cudaDeviceEnablePeerAccess(1,0);
		cudaDeviceDisablePeerAccess(1);
		Matrix *A1 = gpu.rand(rows,cols);
		Matrix *A2 = gpu.rand(rows,cols);
		cudaSetDevice(1);
		//cudaDeviceEnablePeerAccess(0,0);
		cudaDeviceDisablePeerAccess(0);
		Matrix *B1 = gpu.rand(rows,cols);
		Matrix *B2 = gpu.rand(rows,cols);

		cudaSetDevice(0);
		cudaStream_t s;
		cudaStreamCreate(&s);
		cudaSetDevice(1);
		cudaStream_t s2;
		cudaStreamCreate(&s2);
		cudaSetDevice(0);

		int access = 0;
		cudaDeviceCanAccessPeer(&access,0,1);
		cout << access << endl;
		cudaDeviceCanAccessPeer(&access,1,0);
		cout << access << endl;

		cudaSetDevice(0);
		gpu.tick("cuda");

		for(int i = 0; i < 100; i++)
		{
			cudaMemcpyPeerAsync(B2->data,1,A2->data,0,A2->bytes,s);
			cudaSetDevice(1);
			cudaMemcpyPeerAsync(A1->data,0,B1->data,1,B1->bytes,s2);

			cudaSetDevice(0);
			cudaStreamSynchronize(s);
			cudaSetDevice(1);
			cudaStreamSynchronize(s2);
			cudaSetDevice(0);
		}
		gpu.tock("cuda");
	}





	MPI_Barrier(MPI_COMM_WORLD);


	gpu.shutdown_MPI();







}

struct arg_struct
{
		ClusterNet *gpus;
		WikiMaxoutNet *net;
		int device;
};

void *run_net(void * args)
{
	struct arg_struct *_args = (struct arg_struct*)args;
	cout << "device: " << _args->device << endl;
	cudaSetDevice(_args->device);
	_args->net->run();

	return 0;
}

void *print_message(void*)
{
    ClusterNet gpu = ClusterNet(124345);
    WikiMaxoutNet net = WikiMaxoutNet(gpu);
    net.run();

    return 0;
}

void bandwidth_test_MPI(int argc, char *argv[])
{
	ClusterNet *gpu = new ClusterNet(argc,argv,1235,true);

	std::vector<MPI_Request*> sends;
	std::vector<MPI_Request*> recvs;
	std::vector<Matrix*> lSync;
	std::vector<Matrix*> lData;

	int packages = 10;
	float time = 0;

	for(int epoch = 1; epoch < 2000; epoch++)
	{
		if(lData.size() > 0)
		{
			for(int i = 0; i < packages; i++)
			{

					cudaFree(lSync[i]->data);
					cudaFree(lData[i]->data);

			}

			lSync.clear();
			lData.clear();
		}

		for(int i = 0; i < packages; i++)
		{
			lSync.push_back(zeros(128*epoch,128*epoch));
			lData.push_back(gpu->rand(128*epoch,128*epoch));
		}

		for(int j = 0; j < packages; j++)
		{


			MPI_Request *send_request = new MPI_Request;
			MPI_Request *recv_request = new MPI_Request;

			sends.push_back(send_request);
			recvs.push_back(recv_request);

			int target = gpu->MYRANK +1 == gpu->MPI_SIZE ? 0 : gpu->MYRANK+1;
			int source = gpu->MYRANK-1 == -1 ? gpu->MPI_SIZE-1 : gpu->MYRANK-1;

			gpu->tick();
			for (int i = 0; i < gpu->MPI_SIZE -1; i++)
			{
				//MPI_Irecv(lSync[j]->data,lSync[j]->size,MPI_FLOAT,source,999,MPI_COMM_WORLD,recv_request);
				//MPI_Isend(lData[j]->data,lData[j]->size,MPI_FLOAT,target,999,MPI_COMM_WORLD,send_request);
				//MPI_Isend(lData[j]->data,lData[j]->size,MPI_FLOAT,target,j,MPI_COMM_WORLD,send_request);
				if(i == gpu->MYRANK)
				{
					MPI_Send(lData[j]->data,lData[j]->size,MPI_FLOAT,target,j,MPI_COMM_WORLD);
					MPI_Recv(lSync[j]->data,lSync[j]->size,MPI_FLOAT,source,j,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				else
				{
					MPI_Recv(lSync[j]->data,lSync[j]->size,MPI_FLOAT,source,j,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Send(lData[j]->data,lData[j]->size,MPI_FLOAT,target,j,MPI_COMM_WORLD);
				}


			}
			gpu->tick();


		}


		/*
		gpu->tick();
		for(int i = 0; i < packages; i++)
		{
			MPI_Wait(sends[i],MPI_STATUS_IGNORE);
			MPI_Wait(recvs[i],MPI_STATUS_IGNORE);
		}
		*/
		time = gpu->tock();


		//for(int i = 0; i < packages; i++)
			//assert(sum(lData[i]) == sum(lSync[i]));

		printdim(lData[0]);
		cout << 10*packages*lData[0]->bytes/1024./1024./1024./time << " GB/s" << endl;
	}

	gpu->shutdown_MPI();

}

void bandwidth_test_peer()
{

	ClusterNet *gpu = new ClusterNet(1235);

	std::vector<Matrix*> lSync0;
	std::vector<Matrix*> lData0;
	std::vector<Matrix*> lSync1;
	std::vector<Matrix*> lData1;

	std::vector<cudaStream_t> s0s;
	std::vector<cudaStream_t> s1s;

	int packages = 1;
	float time = 0;

	cudaSetDevice(0);
	cudaDeviceEnablePeerAccess(1,0);
	cudaSetDevice(1);
	cudaDeviceEnablePeerAccess(0,0);
	for(int i = 0; i < packages; i++)
	{
		cudaStream_t s0;
		cudaStream_t s1;
		cudaSetDevice(0);
		cudaStreamCreate(&s0);
		cudaSetDevice(1);
		cudaStreamCreate(&s1);
		s0s.push_back(s0);
		s1s.push_back(s1);
	}
	cudaSetDevice(0);
	int access = 0;
	cudaDeviceCanAccessPeer(&access,0,1);
	cout << access << endl;
	cudaDeviceCanAccessPeer(&access,1,0);
	cout << access << endl;

	for(int epoch = 1; epoch < 100; epoch++)
	{
		if(lSync0.size() > 0)
		{
			for(int i = 0; i < packages; i++)
			{

					cudaFree(lSync0[i]->data);
					cudaFree(lData0[i]->data);
					cudaFree(lSync1[i]->data);
					cudaFree(lData1[i]->data);

			}

			lSync0.clear();
			lData0.clear();
			lSync1.clear();
			lData1.clear();
		}

		for(int i = 0; i < packages; i++)
		{
			cudaSetDevice(0);
			lSync0.push_back(zeros(128*epoch,128*epoch));
			lData0.push_back(gpu->rand(128*epoch,128*epoch));
			cudaSetDevice(1);
			lSync1.push_back(zeros(128*epoch,128*epoch));
			lData1.push_back(gpu->rand(128*epoch,128*epoch));
		}

		cudaSetDevice(0);
		gpu->tick();
		for(int j = 0; j < packages; j++)
		{
			cudaMemcpyAsync(lSync1[j]->data,lData0[j]->data,lData0[j]->bytes,cudaMemcpyDefault, s0s[j]);
			cudaSetDevice(1);
			cudaMemcpyAsync(lSync0[j]->data,lData1[j]->data,lData1[j]->bytes,cudaMemcpyDefault,s1s[j]);
			cudaSetDevice(0);
		}





		for(int i = 0; i < packages; i++)
		{
			cudaStreamSynchronize(s0s[i]);
			cudaStreamSynchronize(s1s[i]);
		}


		time = gpu->tock()/1000.;



		cout << packages*lData0[0]->bytes/1024./1024./1024./time << " GB/s" << endl;
	}

}

void bandwidth_test_kernel()
{

	ClusterNet *gpu = new ClusterNet(1235);

	std::vector<Matrix*> lSync0;
	std::vector<Matrix*> lData0;
	std::vector<Matrix*> lSync1;
	std::vector<Matrix*> lData1;

	std::vector<cudaStream_t> s0s;
	std::vector<cudaStream_t> s1s;

	int packages = 10;
	float time = 0;

	cudaSetDevice(0);
	cudaDeviceEnablePeerAccess(1,0);
	cudaSetDevice(1);
	cudaDeviceEnablePeerAccess(0,0);
	for(int i = 0; i < packages; i++)
	{
		cudaStream_t s0;
		cudaStream_t s1;
		cudaSetDevice(0);
		cudaStreamCreate(&s0);
		cudaSetDevice(1);
		cudaStreamCreate(&s1);
		s0s.push_back(s0);
		s1s.push_back(s1);
	}
	cudaSetDevice(0);
	int access = 0;
	cudaDeviceCanAccessPeer(&access,0,1);
	cout << access << endl;
	cudaDeviceCanAccessPeer(&access,1,0);
	cout << access << endl;

	for(int epoch = 1; epoch < 1000; epoch++)
	{
		if(lSync0.size() > 0)
		{
			for(int i = 0; i < packages; i++)
			{

					cudaFree(lSync0[i]->data);
					cudaFree(lData0[i]->data);
					cudaFree(lSync1[i]->data);
					cudaFree(lData1[i]->data);

			}

			lSync0.clear();
			lData0.clear();
			lSync1.clear();
			lData1.clear();
		}

		for(int i = 0; i < packages; i++)
		{
			cudaSetDevice(0);
			lSync0.push_back(zeros(128*epoch,128*epoch));
			lData0.push_back(gpu->rand(128*epoch,128*epoch));
			cudaSetDevice(1);
			lSync1.push_back(zeros(128*epoch,128*epoch));
			lData1.push_back(gpu->rand(128*epoch,128*epoch));
		}

		cudaSetDevice(0);
		gpu->tick();

		for(int j = 0; j < packages; j++)
		{
			add(lSync0[j],lData1[j],lSync0[j]);
			cudaSetDevice(1);
			add(lSync1[j],lData0[j],lSync1[j]);
			cudaSetDevice(0);
		}

		cudaDeviceSynchronize();
		cudaSetDevice(1);
		cudaDeviceSynchronize();
		cudaSetDevice(0);
		time = gpu->tock();

		/*
		for(int i = 0; i < packages; i++)
			assert(sum(lData0[i]) == sum(lSync1[i]));

		for(int i = 0; i < packages; i++)
			assert(sum(lData1[i]) == sum(lSync0[i]));
			*/


		printdim(lSync0[0]);
		cout << 1000*2*packages*lData0[0]->bytes/1024./1024./1024./time << " GB/s" << endl;
	}

}


void bandwidth_test_compression(int argc, char *argv[])
{

	ClusterNet *gpu = new ClusterNet(argc,argv,1235,true);
	MPI_Request *send_request = new MPI_Request;
	MPI_Request *recv_request = new MPI_Request;

	Matrix *w_grad_next = empty(1024,1024);
	Matrix *w_next_sync = empty(1024,1024);

	//warmup
	int target = gpu->MYRANK +1 == gpu->MPI_SIZE ? 0 : gpu->MYRANK+1;
	int source = gpu->MYRANK-1 == -1 ? gpu->MPI_SIZE-1 : gpu->MYRANK-1;

	for (int i = 0; i < gpu->MPI_SIZE - 1; i++)
	{
		MPI_Isend(w_grad_next->data,w_grad_next->size,MPI_FLOAT,target,i,MPI_COMM_WORLD, send_request);
		MPI_Irecv(w_next_sync->data,w_grad_next->size,MPI_FLOAT,source,i,MPI_COMM_WORLD,recv_request);
		target = target +1 == gpu->MPI_SIZE ? 0 : target+1;
		source = source-1 == -1 ? gpu->MPI_SIZE-1 : source-1;
	}

	MPI_Wait(recv_request,MPI_STATUS_IGNORE);

	int times = 100;

	gpu->tick();

	for(int i = 0; i < times; i++)
	{
		target = gpu->MYRANK +1 == gpu->MPI_SIZE ? 0 : gpu->MYRANK+1;
		source = gpu->MYRANK-1 == -1 ? gpu->MPI_SIZE-1 : gpu->MYRANK-1;

		for (int i = 0; i < gpu->MPI_SIZE - 1; i++)
		{
			MPI_Isend(w_grad_next->data,w_grad_next->size,MPI_FLOAT,target,i,MPI_COMM_WORLD, send_request);
			MPI_Recv(w_next_sync->data,w_grad_next->size,MPI_FLOAT,source,i,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			target = target +1 == gpu->MPI_SIZE ? 0 : target+1;
			source = source-1 == -1 ? gpu->MPI_SIZE-1 : source-1;
		}

		//MPI_Wait(send_request,MPI_STATUS_IGNORE);
	}


	float sec = gpu->tock()*1000.0;
	float GB = 3*times*w_grad_next->bytes/(1024.0*1024.0*1024.0);
	if(gpu->MYRANK == 0)
	{
		cout << "Size in GB: " << GB << endl;
		cout << "GB/s: " << GB/sec << endl;
	}


	gpu->shutdown_MPI();
}

void simple_bandwidth_test(int argc, char *argv[])
{

		ClusterNet *gpu = new ClusterNet(argc,argv,1235,true);
		MPI_Request *send_request = new MPI_Request;
		MPI_Request *recv_request = new MPI_Request;

		int size = 12000;
		for(int i = 8; i < size; i+=8)
		{
			for(int j = 0; j < 3; j++)
			{
				Matrix *w_grad_next;
				if(j==0) w_grad_next = empty(i,i);
				if(j==1) w_grad_next = empty(i/2,i/2);
				if(j==2) w_grad_next = empty(i/4,i/8);
				Matrix *w_next_sync;
				if(j==0) w_next_sync = empty(i,i);
				if(j==1) w_next_sync = empty(i/2,i/2);
				if(j==2) w_next_sync = empty(i/4,i/8);

				if(gpu->MYRANK == 0)
					MPI_Send(w_grad_next->data,w_grad_next->size,MPI_FLOAT,1,999,MPI_COMM_WORLD);
				if(gpu->MYRANK == 1)
					MPI_Recv(w_next_sync->data,w_next_sync->size,MPI_FLOAT,0,999,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				MPI_Barrier(MPI_COMM_WORLD);
				int times = 100;
				gpu->tick();
				for(int k = 0; k < times; k++)
				{
					if(gpu->MYRANK == 0)
						MPI_Send(w_grad_next->data,w_grad_next->size,MPI_FLOAT,1,999,MPI_COMM_WORLD);
					if(gpu->MYRANK == 1)
						MPI_Recv(w_next_sync->data,w_next_sync->size,MPI_FLOAT,0,999,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				}

				float quant = 9.5e-08f;
				float dequant = 2.0e-08f;
				float compreess = 1.e-07f;
				float decompress = 5.0e-08f;

				float added_penalty = 0.0f;
				if(j == 1)added_penalty = compreess + decompress;
				if(j == 2)added_penalty = quant + dequant;

				//cout << 100*(added_penalty)*w_grad_next->size << endl;
				float sec = gpu->tock() + (100*(added_penalty)*(i*i));
				float GB = times*w_grad_next->bytes/(1024.0*1024.0*1024.0);
				if(gpu->MYRANK == 0)
				{
					cout << "Size: " << w_grad_next->rows << "x" << w_grad_next->cols << " GB/s: " << GB/(sec/1000) << " " << sec*(j == 2 ? 2.0 : 1.0) << "ms"<< endl;

				}

				cudaFree(w_grad_next->data);
				cudaFree(w_next_sync->data);
			}
		}


		gpu->shutdown_MPI();

}


void model_parallelism_test(int argc, char *argv[])
{

		ClusterNet *GPU = new ClusterNet(argc,argv,1235,true);
		MPI_Request *send_request = new MPI_Request;
		MPI_Request *recv_request = new MPI_Request;

		std::vector<MPI_Request *> send_requests;
		std::vector<MPI_Request *> recv_requests;

		for(int i = 0; i < GPU->MPI_SIZE-1; i++)
		{
			send_requests.push_back(new MPI_Request);
			recv_requests.push_back(new MPI_Request);
		}

		float max_value = 1.0f;



		for(int round = 128; round <= 8192; round+=128)
		{
			int batch_size = 256;
			int inner = round;
			int outer = round;

			Matrix *A = GPU->rand(batch_size,inner);
			Matrix *B = GPU->distributed_uniformSqrtWeight(inner,outer);
			Matrix *B_normal = GPU->uniformSqrtWeight(inner,outer);
			Matrix *out = zeros(batch_size,outer);
			Matrix *out_stacked = zeros(batch_size,outer);


			int col_split_size = (B->isDistributed == 1 ? B->cols_distributed : B->cols) / GPU->MPI_SIZE;
			int remainder = (B->isDistributed == 1 ? B->cols_distributed : B->cols) - (col_split_size*GPU->MPI_SIZE);

			if(GPU->MYRANK == 0)
			cout << batch_size << "x" << inner << " DOT " << inner << "x" << outer << endl;

			Matrix** arrOut = (Matrix**) malloc(sizeof(Matrix*) * GPU->MPI_SIZE);
			Matrix** arrOut8 = (Matrix**) malloc(sizeof(Matrix*) * GPU->MPI_SIZE);

			for (int i = 0; i < GPU->MPI_SIZE; i++)
			{
				if (i == GPU->MPI_SIZE - 1)
				{
					arrOut[i] = empty(A->rows, col_split_size + remainder);
					arrOut8[i] = empty_char(A->rows, col_split_size + remainder);
				}
				else
				{
					arrOut[i] = empty(A->rows, col_split_size);
					arrOut8[i] = empty_char(A->rows, col_split_size);
				}
			}

			float **h_arrA = (float**) malloc(sizeof(float*) * GPU->MPI_SIZE);
			unsigned char **h_arrA8 = (unsigned char**) malloc(sizeof(unsigned char*) * GPU->MPI_SIZE);
			for (int i = 0; i < GPU->MPI_SIZE; i++)
			{
				h_arrA[i] = arrOut[i]->data;
				h_arrA8[i] = arrOut8[i]->char_data;
			}

			float **d_arrA;
			cudaMalloc((void**) &d_arrA, sizeof(float*) * GPU->MPI_SIZE);
			cudaMemcpy(d_arrA, h_arrA, sizeof(float*) * GPU->MPI_SIZE,cudaMemcpyDefault);

			unsigned char **d_arrA8;
			cudaMalloc((unsigned char**) &d_arrA8, sizeof(unsigned char*) * GPU->MPI_SIZE);
			cudaMemcpy(d_arrA8, h_arrA8, sizeof(unsigned char*) * GPU->MPI_SIZE,cudaMemcpyDefault);

			for(int epoch = 0; epoch < 2; epoch++)
			for(int type = 0; type < 3; type++)
			{
				std::string text = "";
				if(type == 0)
					text = "DOT";
				else if(type == 1)
					text = "DOT32BIT";
				else if(type == 2)
					text = "DOT8BIT";

				if(GPU->MYRANK == 0 && epoch == 1){ GPU->tick(text); }
				for(int i = 0; i < 100; i++)
				{
					if(type == 0)
					{
						GPU->dot(A,B_normal,out);
						continue;
					}
					GPU->dot(A,B,arrOut[GPU->MYRANK]);

					int target = GPU->MYRANK +1 == GPU->MPI_SIZE ? 0 : GPU->MYRANK+1;
					int source = GPU->MYRANK-1 == -1 ? GPU->MPI_SIZE-1 : GPU->MYRANK-1;

					if(type == 2)
					{
						GPU->compression_8bit(arrOut[GPU->MYRANK],max_value,arrOut8[GPU->MYRANK]);
						for (int i = 0; i < GPU->MPI_SIZE - 1; i++)
						{
							MPI_Isend(arrOut8[GPU->MYRANK]->char_data,arrOut8[GPU->MYRANK]->size,MPI_CHAR,target,i,MPI_COMM_WORLD, send_requests[i]);
							MPI_Irecv(arrOut8[source]->char_data,arrOut8[source]->size,MPI_CHAR,source,i,MPI_COMM_WORLD,recv_requests[i]);
							target = target +1 == GPU->MPI_SIZE ? 0 : target+1;
							source = source-1 == -1 ? GPU->MPI_SIZE-1 : source-1;
						}
					}


					if(type == 1)
					{
						for (int i = 0; i < GPU->MPI_SIZE - 1; i++)
						{
							MPI_Isend(arrOut[GPU->MYRANK]->data,arrOut[GPU->MYRANK]->size,MPI_FLOAT,target,i,MPI_COMM_WORLD, send_requests[i]);
							MPI_Irecv(arrOut[source]->data,arrOut[source]->size,MPI_FLOAT,source,i,MPI_COMM_WORLD,recv_requests[i]);
							target = target +1 == GPU->MPI_SIZE ? 0 : target+1;
							source = source-1 == -1 ? GPU->MPI_SIZE-1 : source-1;
						}
					}
					//MPI_Wait(next->send_request,MPI_STATUS_IGNORE);
					for(int i = 0; i < GPU->MPI_SIZE-1; i++)
						MPI_Wait(recv_requests[i],MPI_STATUS_IGNORE);

					if(type == 2)
					{
						for (int i = 0; i < GPU->MPI_SIZE; i++)
						{
							if(i == GPU->MYRANK){continue;}
							GPU->decompression_8bit(arrOut8[i],max_value,arrOut[i]);
						}
					}

					hStackN(d_arrA,	arrOut[0]->size, out_stacked, GPU->MPI_SIZE);


				}
				if(GPU->MYRANK == 0 && epoch == 1){ GPU->tock(text); }


				/*
				MPI_Barrier(MPI_COMM_WORLD);
				if(type == 0)
					printsum(out);
				else if(type == 1)
					printsum(out_stacked);
				else if(type == 2)
					printsum(out_stacked);

				MPI_Barrier(MPI_COMM_WORLD);
				if(type == 0)
					printmat(out,0,4,0,4);
				else if(type == 1)
					printmat(out_stacked,0,4,0,4);
				else if(type == 2)
					printmat(out_stacked,0,4,0,4);

				*/


				if(type == 0)
				{
					abs(out,out);
					max_value = max(out);
				}

			}

			cudaFree(A->data);
			cudaFree(B->data);
			cudaFree(out->data);
			cudaFree(out_stacked->data);
			cudaFree(d_arrA8);
			cudaFree(d_arrA);
			for(int i = 0; i < GPU->MPI_SIZE; i++)
			{
				cudaFree(arrOut[i]->data);
				cudaFree(arrOut8[i]->char_data);
			}

			size_t total, free;
			cudaMemGetInfo(&free, &total);



		}

		GPU->shutdown_MPI();

}

void simple_bandwidth_test_CPU(int argc, char *argv[])
{

		ClusterNet *gpu = new ClusterNet(argc,argv,1235,true);
		MPI_Request *send_request = new MPI_Request;
		MPI_Request *recv_request = new MPI_Request;

		size_t size = 1024*1024*1024;
		float *data = (float*)malloc(sizeof(float)*size);
		float *data_sync = (float*)malloc(sizeof(float)*size);


		int times = 10;
		gpu->tick();
		for(int i = 0; i < times; i++)
		{
			if(gpu->MYRANK == 0)
			{
				MPI_Send(data,size,MPI_FLOAT,1,999,MPI_COMM_WORLD);
				//MPI_Recv(w_next_sync->data,w_next_sync->size,MPI_FLOAT,0,999,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			if(gpu->MYRANK == 1)
			{
				MPI_Recv(data_sync,size,MPI_FLOAT,0,999,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//MPI_Send(w_grad_next->data,w_grad_next->size,MPI_FLOAT,1,999,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}


		double sec = gpu->tock()*1000.0;
		double GB = times*size*4.0/(1024.0*1024.0*1024.0);
		if(gpu->MYRANK == 0)
		{
			cout << "Size in GB: " << GB << endl;
			cout << "GB/s: " << GB/sec << endl;
		}


		gpu->shutdown_MPI();

}

void compression_test(int argc, char *argv[])
{

	ClusterNet *gpu = new ClusterNet();
	Matrix *A = scalarMul(gpu->randn(5120,5120),1.0f/10.0f);
	Matrix *out = empty_char(5120,5120);





	gpu->tick();
	for(int i = 0; i < 10000; i++)
		gpu->compression_8bit(A, 0.1f,out);
	gpu->tock();

	gpu->tick();
	for(int i = 0; i < 10000; i++)
		gpu->decompression_8bit(out, 0.1f,A);
	gpu->tock();
}


int main(int argc, char *argv[])
{
	
	//bandwidth_test_peer();

	//bandwidth_test_MPI(argc,argv);

	//bandwidth_test_kernel();

	//compression_test(argc,argv);

	//simple_bandwidth_test(argc,argv);
	//simple_bandwidth_test_CPU(argc,argv);
	//model_parallelism_test(argc,argv);



	//ClusterNet *gpu = new ClusterNet(234);


	/*
	Matrix *rdm = gpu->rand_numbers(10,10);

	printmat(rdm);
	*/


	/*
	ClusterNet *gpu = new ClusterNet(234);
	int out_rows = 128;
	int out_cols = 800;
	int inner = 784;


	Matrix *A = gpu->rand(out_rows,inner);
	Matrix *B = gpu->rand(inner,out_cols);
	Matrix *out1 = zeros(out_rows,out_cols);

	Matrix *charA = empty_char(out_rows,inner);
	Matrix *charB = empty_char(inner,out_cols);
	Matrix *out2 = empty(out_rows,out_cols);
	Matrix *out3 = empty(out_rows,out_cols);

	gpu->tick();
	for(int i = 0; i < 100; i++)
		gpu->dot(A,B,out3);
	gpu->tock();

	float maxA = max(abs(A));
	float maxB = max(abs(B));
	gpu->compression_8bit(A,maxA,charA);
	gpu->compression_8bit(B,maxB,charB);


	//printmat(A);
	//printmat(gpu->decompression_8bit(charA,maxA));
	//printmat(B);
	//printmat(gpu->decompression_8bit(charB,maxB));
	//cout << sum(gpuSqrt(square(sub(B,gpu->decompression_8bit(charB,maxB)))))/(float)B->size << endl;
	//cout << sum(gpuSqrt(square(sub(A,gpu->decompression_8bit(charA,maxA)))))/(float)B->size << endl;
	//gpu->compression_8bit(A,maxA,charA);

	//printmat(out1);
	//printmat(out1,60,65,70,80);
	gpu->tick();
	for(int i = 0; i < 100; i++)
	{
		fill_matrix(out1,0.0f);
		gpu->dot8bit(charA,charB,maxA,maxB,out1);
	}
	gpu->tock();

	gpu->tick();
	for(int i = 0; i < 100; i++)
		gpu->dot8bit_shared(charA,charB,maxA,maxB,out2);
	gpu->tock();
	//printmat(gpu->decompression_8bit(charB,maxB));
	//printmat(out1,60,65,70,80);
	//printmat(out2,60,65,70,80);
	//printmat(out1);
	//printmat(out2);

	//printsum(out1);
	//printsum(out2);
	cout << sum(gpuSqrt(square(sub(out1,out2))))/(float)out1->size << endl;
	cout << sum(gpuSqrt(square(sub(out1,out3))))/(float)out1->size << endl;
	cout << sum(gpuSqrt(square(sub(out2,out3))))/(float)out1->size << endl;


	//cout << "max A " << maxA <<endl;
	//cout << "max B " << maxB <<endl;

	*/



	//ClusterNet *gpu = new ClusterNet(argc,argv,123635,false);
	ClusterNet *gpu = new ClusterNet(argc,argv);
	/*

	Matrix *A = gpu->distributed_uniformSqrtWeight(6,4);
	Matrix *B = gpu->rand(4,6);
	Matrix *A2 = empty(6,2);

	A2->data = A->data;
	printmat(A);

	Matrix *C = gpu->dotMPI(B,A);
	Matrix *C2 = gpu->dot(B,A2);

	printmat(C);
	printmat(C2);

	gpu->shutdown_MPI();
	*/












	//Matrix *X = read_hdf5("/home/tim/data/mnist/X.hdf5");
	//Matrix *y = read_hdf5("/home/tim/data/mnist/y.hdf5");

	Matrix *X = gpu->distribute_file("/home/tim/data/mnist/X.hdf5");
	Matrix *y = gpu->distribute_file("/home/tim/data/mnist/y.hdf5");


	//Matrix *X = gpu->distribute_rows_hdf5_file("/home/tim/data/mnist/X.hdf5");
	//Matrix *y = gpu->distribute_rows_hdf5_file("/home/tim/data/mnist/y.hdf5");
	//Matrix *y = gpu->distribute_rows_hdf5_file("/home/tim/data/mnist/y_15000.hdf5");


	printdim(X);
	printdim(y);
	BatchAllocator b = BatchAllocator();
	//16384
	int batch_size_per_GPU = 128;

	b.init(X,y,(1.0-0.85715),batch_size_per_GPU,128,gpu, Single_GPU);


	Layer *l0 = new Layer(X->cols,batch_size_per_GPU,Input,gpu);
	//l0->PARALLELISM = DataParallelism;
	l0->PARALLELISM = ModelParallelism;
	Layer *l1 = new Layer(1024, Logistic, l0);
	//l1->PARALLELISM = DataParallelism;
	l1->PARALLELISM = ModelParallelism;
	Layer *l2 = new Layer(1024, Logistic, l1);
	//l2->PARALLELISM = DataParallelism;
	l2->PARALLELISM = ModelParallelism;
	Layer *l3 = new Layer(10, Softmax, l2);
	//l3->PARALLELISM = DataParallelism;
	l3->PARALLELISM = ModelParallelism;


	l0->DROPOUT = 0.2f;
	l0->set_hidden_dropout(0.5f);

	cout << gpu->MYRANK << endl;

	float decay = 0.98f;
	gpu->tick("pass");
	b.SKIP_LAST_BATCH = true;
	int epochs = 60;
	for(int epoch = 0; epoch < epochs; epoch++)
	{
		gpu->tick("epoch");
		if(gpu->MYRANK == 0)
			cout << "EPOCH: " << epoch + 1 << endl;
		b.propagate_through_layers(l0,Training,epoch);
		b.propagate_through_layers(l0,Trainerror,epoch);
		b.propagate_through_layers(l0,CVerror,epoch);


		l0->learning_rate_decay(decay);

		if(epoch == 50)
		{
			l0->dropout_decay();
			decay = 0.85f;
		}

		//cout << l1->MAX_GRAD_VALUE << endl;
		gpu->tock("epoch");
	}
	gpu->tock("pass");



	gpu->shutdown_MPI();




	if(gpu->MYRANK == 0)
	{
		int n1 = l3->Train_errors[0].size();
		int n2 = l3->CV_errors[0].size();
		cout << n1 << endl;
		cout << n2 << endl;
		Matrix *train = empty_cpu(epochs,n1);
		Matrix *cv = empty_cpu(epochs,n2);

		for(int i = 0; i < epochs; i++)
		{
			for(int j = 0; j < n1; j++)
				train->data[j + (i*n1)] = l3->Train_errors[i][j];
			for(int j = 0; j < n2; j++)
				cv->data[j + (i*n2)] = l3->CV_errors[i][j];
		}

		write_hdf5("/home/tim/data/mnist/results/32bit/train_error_model.hdf5" ,train);
		write_hdf5("/home/tim/data/mnist/results/32bit/cv_error_model.hdf5",cv);
	}





	/*

	cudaSetDevice(0);

	Matrix *X = read_hdf5("/home/tim/data/mnist/X.hdf5");
	Matrix *y = read_hdf5("/home/tim/data/mnist/y.hdf5");



	ClusterNet gpu = ClusterNet(1235);


	BatchAllocator b = BatchAllocator();

	std::vector<int> layers;
	layers.push_back(1200);
	layers.push_back(1200);
	std::vector<float> dropout;
	dropout.push_back(0.2f);
	dropout.push_back(0.5f);
	dropout.push_back(0.5f);
	BatchAllocator allocator = BatchAllocator();
	allocator.init(X,y,(1.0-0.8571429),128,256,gpu, Single_GPU);
	DeepNeuralNetwork net = DeepNeuralNetwork(layers,Classification, gpu, allocator, 10);
	net.EPOCHS = 500;
	net.TRANSITION_EPOCH = 75;
	net.LEARNING_RATE = 0.003;
	net.UPDATE_TYPE = RMSProp;
	net.DROPOUT = dropout;
	//net.MAIN_UNIT = Double_Rectified_Linear;
	net.train();

	*/


	//cudaSetDevice(1);
	//ClusterNet *gpus = new ClusterNet(123635);
	//WikiMaxoutNet_PCIe net = WikiMaxoutNet_PCIe(gpus);
	//net.run();


	/*
	cudaSetDevice(0);
	struct arg_struct *args0 = (arg_struct*)malloc(sizeof(arg_struct));
	ClusterNet *gpus0 = new ClusterNet(23452345);
	WikiMaxoutNet *net0 = new WikiMaxoutNet(gpus0[0]);
	args0->gpus = gpus0;
	args0->net = net0;
	args0->device = 0;

	net0->run();

	pthread_t t0;
	pthread_create(&t0, NULL, &run_net, args0);

	cudaSetDevice(1);
	struct arg_struct *args1 = (arg_struct*)malloc(sizeof(arg_struct));
	ClusterNet *gpus1 = new ClusterNet(23452345);
	WikiMaxoutNet *net1 = new WikiMaxoutNet(gpus1[0]);
	args1->gpus = gpus1;
	args1->net = net1;
	args1->device = 1;

	pthread_t t1;
	//pthread_create(&t1, NULL, &run_net, args1);

	cudaSetDevice(2);
	struct arg_struct *args2 = (arg_struct*)malloc(sizeof(arg_struct));
	ClusterNet *gpus2 = new ClusterNet(23452345);
	WikiMaxoutNet *net2 = new WikiMaxoutNet(gpus2[0]);
	args2->gpus = gpus2;
	args2->net = net2;
	args2->device = 2;

	pthread_t t2;
	//pthread_create(&t2, NULL, &run_net, args2);


	cout << "rolfen kek!" << endl;

	void* result0;
	void* result1;
	void* result2;
	pthread_join(t0,&result0);
	//pthread_join(t1,&result1);
	//pthread_join(t2,&result2);
	*/



}







