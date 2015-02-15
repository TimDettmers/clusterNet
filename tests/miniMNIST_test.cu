#include <stdio.h>
#include <util.cuh>
#include <basicOps.cuh>
#include <assert.h>
#include <clusterNet.h>
#include <batchAllocator.h>
#include <string>
#include <DeepNeuralNetwork.h>

using std::cout;
using std::endl;

void run_miniMNIST_test(ClusterNet gpus)
{

	// Tests RMSprop with weight updates, logistic grad.
	// Additionally tests the interplay between different functions.

	char buff[1024] = {0};
	ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
	std::string path = std::string(buff);
	replace(path,"/build/testSuite.out","/tests/");

	//Matrix *X = read_hdf5((path + "/mnist_mini_X.hdf5").c_str());
	//Matrix *y = read_hdf5((path + "/mnist_mini_y.hdf5").c_str());

	Matrix *X = read_hdf5("/home/tim/data/mnist/X.hdf5");
	Matrix *y = read_hdf5("/home/tim/data/mnist/y.hdf5");


	Matrix *w1 = gpus.uniformSqrtWeight(784,1000);
	Matrix *w2 = gpus.uniformSqrtWeight(1000,10);
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

	Matrix *z = zeros(10,1000);

	BatchAllocator b = BatchAllocator();
	b.init(X, y, 0.2, 128, 64);
	int epochs  = 20;
	float learning_rate = 0.003;
	float momentum = 0.5;
	for(int EPOCH = 1; EPOCH < epochs; EPOCH++)
	{
	  momentum += 0.01;
	  if(momentum > 0.95) momentum = 0.95;

	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {
		  b.broadcast_batch_to_processes();


		  //nesterov updates
		  scalarMul(m1,momentum,m1);
		  scalarMul(m2,momentum,m2);
		  add(w1,m1,w1);
		  add(w2,m2,w2);

		  //feedforward
		  Matrix *d0 = gpus.dropout(b.CURRENT_BATCH,0.2);
		  //print_gpus_matrix(w1);
		  Matrix *z1 = gpus.dot(d0, w1);
		  //logistic(z1, z1);
		  rectified_linear(z1,z1);

		  add_to_z(z,z1,y,10,z);

		  b.allocate_next_batch_async();

		  b.replace_current_batch_with_next();

	  }

	  scalarMul(z,1.0/(b.TOTAL_BATCHES));



	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {
		  b.broadcast_batch_to_processes();


		  //nesterov updates
		  scalarMul(m1,momentum,m1);
		  scalarMul(m2,momentum,m2);
		  add(w1,m1,w1);
		  add(w2,m2,w2);

		  //feedforward
		  Matrix *d0 = gpus.dropout(b.CURRENT_BATCH,0.2);
		  //print_gpus_matrix(w1);
		  Matrix *z1 = gpus.dot(d0, w1);
		  //logistic(z1, z1);
		  rectified_linear(z1,z1);
		  Matrix *d1 = gpus.dropout(z1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *t = create_t_matrix(b.CURRENT_BATCH_Y,10);

		  b.allocate_next_batch_async();

		  //backprop
		  Matrix *e1 = sub(out, t);
		  Matrix *e2 = gpus.dotT(e1, w2);
		  gpus.Tdot(z1,e1,grad_w2);

		  gpus.Tdot(b.CURRENT_BATCH,e2,grad_w1);

		  //weight updates
		  RMSprop_with_nesterov_weight_update(ms1,grad_w1,w1,m1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_nesterov_weight_update(ms2,grad_w2,w2,m2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);

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

		  b.broadcast_batch_to_processes();

		  Matrix *d0 = scalarMul(b.CURRENT_BATCH,0.8);
		  Matrix *a1 = gpus.dot(d0,w1);
		  logistic(a1, a1);
		  Matrix *d1 = scalarMul(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b.CURRENT_BATCH_Y);
		  b.allocate_next_batch_async();
		  float sum_value = sum(eq);

		  train_error += (b.CURRENT_BATCH->rows - sum_value)/ (1.0f * b.CURRENT_BATCH->rows *b.TOTAL_BATCHES) ;

		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);

		  b.replace_current_batch_with_next();
	  }

	  std::cout << "Train error: " << train_error << std::endl;

	  cv_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES_CV; i++)
	  {
		  b.broadcast_batch_cv_to_processes();
		  Matrix *d0 = scalarMul(b.CURRENT_BATCH_CV,0.8);
		  Matrix *a1 = gpus.dot(d0,w1);
		  logistic(a1, a1);
		  Matrix *d1 = scalarMul(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b.CURRENT_BATCH_CV_Y);
		  b.allocate_next_cv_batch_async();
		  float sum_value = sum(eq);

		  cv_error += (b.CURRENT_BATCH_CV->rows  - sum_value)/ (1.0f * b.CURRENT_BATCH_CV->rows *b.TOTAL_BATCHES_CV) ;

		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);

		  b.replace_current_cv_batch_with_next();
	  }

	  std::cout << "Cross validation error: " << cv_error << std::endl;

	}
	/*
	ASSERT(train_error < 0.03f,"mini-MNIST train error 17 epochs < 0.03.");
	ASSERT(cv_error < 0.22f, "mini-MNIST train error 17 epochs < 0.22.");

	b.finish_batch_allocator();


	Matrix *w1_dist = gpus.distributed_uniformSqrtWeight(784,1000);
	Matrix *w2_dist = gpus.distributed_uniformSqrtWeight(1000,10);
	Matrix *m1_dist = gpus.distributed_zeros(784,1000);
	Matrix *m2_dist = gpus.distributed_zeros(1000,10);
	Matrix *ms1_dist = gpus.distributed_zeros(784,1000);
	Matrix *ms2_dist = gpus.distributed_zeros(1000,10);
	Matrix *grad_w1_ms_dist = gpus.distributed_zeros(784,1000);
	Matrix *grad_w2_ms_dist = gpus.distributed_zeros(1000,10);
	Matrix *grad_w1_dist = gpus.distributed_zeros(784,1000);
	Matrix *grad_w2_dist = gpus.distributed_zeros(1000,10);

	BatchAllocator b_dist = BatchAllocator();
	b_dist.init(X, y, 0.2, 32, 64, gpus, Distributed_weights);
	for(int EPOCH = 1; EPOCH < epochs; EPOCH++)
	{
	  momentum += 0.01;
	  if(momentum > 0.95) momentum = 0.95;
	  for(int i = 0; i < b_dist.TOTAL_BATCHES; i++)
	  {

		  b_dist.broadcast_batch_to_processes();

		  //nesterov updates
		  scalarMul(m1_dist,momentum,m1_dist);
		  scalarMul(m2_dist,momentum,m2_dist);
		  add(w1_dist,m1_dist,w1_dist);
		  add(w2_dist,m2_dist,w2_dist);

		  Matrix *d0 = gpus.dropout(b_dist.CURRENT_BATCH,0.2);
		  //print_gpus_matrix(w1);
		  Matrix *z1 = gpus.dot(d0, w1_dist);
		  logistic(z1, z1);
		  Matrix *d1 = gpus.dropout(z1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2_dist);
		  Matrix *out = softmax(a2);
		  Matrix *t = create_t_matrix(b_dist.CURRENT_BATCH_Y,10);

		  b_dist.allocate_next_batch_async();

		  //backprop
		  Matrix *e1 = sub(out, t);
		  Matrix *e2 = gpus.dotT(e1, w2_dist);
		  gpus.Tdot(z1,e1,grad_w2_dist);
		  logisticGrad(z1,z1);
		  mul(e2,z1,e2);
		  gpus.Tdot(b_dist.CURRENT_BATCH,e2,grad_w1_dist);

		  RMSprop_with_nesterov_weight_update(ms1_dist,grad_w1_dist,w1_dist,m1_dist,0.9f,learning_rate,b_dist.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_nesterov_weight_update(ms2_dist,grad_w2_dist,w2_dist,m2_dist,0.9f,learning_rate,b_dist.CURRENT_BATCH->rows, momentum);

		  cudaFree(e1->data);
		  cudaFree(e2->data);
		  cudaFree(z1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(t->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);

		  b_dist.replace_current_batch_with_next();

	  }

	  train_error = 0;
	  for(int i = 0; i < b_dist.TOTAL_BATCHES; i++)
	  {
		  b_dist.broadcast_batch_to_processes ();

		  Matrix *d0 = scalarMul(b_dist.CURRENT_BATCH,0.8);
		  Matrix *a1 = gpus.dot(d0,w1);

		  logistic(a1, a1);
		  Matrix *d1 = scalarMul(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b_dist.CURRENT_BATCH_Y);
		  float sum_value = sum(eq);

		  b_dist.allocate_next_batch_async();

		  train_error += (b_dist.CURRENT_BATCH->rows - sum_value)/ (1.0f * b_dist.CURRENT_BATCH->rows *b_dist.TOTAL_BATCHES) ;

		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(out->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);

		  b_dist.replace_current_batch_with_next();
	  }

	  //std::cout << "Train error: " << train_error << std::endl;

	  cv_error = 0;
	  for(int i = 0; i < b_dist.TOTAL_BATCHES_CV; i++)
	  {
		  b_dist.broadcast_batch_cv_to_processes();

		  Matrix *d0 = scalarMul(b_dist.CURRENT_BATCH_CV,0.8);
		  Matrix *a1 = gpus.dot(d0,w1);
		  logistic(a1, a1);
		  Matrix *d1 = scalarMul(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *result = argmax(out);
		  Matrix *eq = equal(result,b_dist.CURRENT_BATCH_CV_Y);
		  float sum_value = sum(eq);

		  b_dist.allocate_next_cv_batch_async();

		  cv_error += (b_dist.CURRENT_BATCH_CV->rows  - sum_value)/ (1.0f * b_dist.CURRENT_BATCH_CV->rows *b_dist.TOTAL_BATCHES_CV) ;

		  cudaFree(a1->data);
		  cudaFree(a2->data);
		  cudaFree(d0->data);
		  cudaFree(d1->data);
		  cudaFree(out->data);
		  cudaFree(result->data);
		  cudaFree(eq->data);

		  b_dist.replace_current_cv_batch_with_next();
	  }

	  //std::cout << "Cross validation error: " << cv_error << std::endl;

	}


	ASSERT(train_error < 0.03f,"mini-MNIST train error 17 epochs < 0.03.");
	ASSERT(cv_error < 0.22f, "mini-MNIST train error 17 epochs < 0.22.");



	b_dist.finish_batch_allocator();


	// Maxout test

	// Tests RMSprop with weight updates, logistic grad.
	// Additionally tests the interplay between different functions.

	w1 = gpus.uniformSqrtWeight(784,1024);
	w2 = gpus.uniformSqrtWeight(128,10);
	m1 = zeros(784,1024);
	m2 = zeros(128,10);
	ms1 = zeros(784,1024);
	ms2 = zeros(128,10);
	grad_w1_ms = zeros(784,1024);
	grad_w2_ms = zeros(128,10);
	grad_w1 = empty(784,1024);
	grad_w2 = empty(128,10);
	cv_error = 0.0f;
	train_error = 0.0f;

	b = BatchAllocator();
	b.init(X, y, 0.2, 32, 64);
	epochs  = 17;
	learning_rate = 0.01;
	momentum = 0.5;
	for(int EPOCH = 1; EPOCH < epochs; EPOCH++)
	{
	  momentum += 0.01;
	  if(momentum > 0.95) momentum = 0.95;
	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {
		  b.broadcast_batch_to_processes();

		  //nesterov updates
		  scalarMul(m1,momentum,m1);
		  scalarMul(m2,momentum,m2);
		  add(w1,m1,w1);
		  add(w2,m2,w2);

		  //feedforward
		  Matrix *d0 = gpus.dropout(b.CURRENT_BATCH,0.2);
		  //print_gpus_matrix(w1);
		  Matrix *z1 = gpus.dot(d0, w1);
		  Matrix **a_paired = maxout(z1,8);
		  Matrix *a1 = a_paired[0];
		  Matrix *a1_idx = a_paired[1];
		  Matrix *d1 = gpus.dropout(a1,0.5);
		  Matrix *a2 = gpus.dot(d1,w2);
		  Matrix *out = softmax(a2);
		  Matrix *t = create_t_matrix(b.CURRENT_BATCH_Y,10);

		  b.allocate_next_batch_async();

		  //backprop
		  Matrix *e1 = sub(out, t);
		  Matrix *e2_partial = gpus.dotT(e1, w2);
		  Matrix *e2 = empty(b.CURRENT_BATCH->rows,e2_partial->cols*8);

		  gpus.Tdot(a1,e1,grad_w2);
		  expand_to_maxout_grad(e2_partial, a1_idx,e2);
		  gpus.Tdot(b.CURRENT_BATCH,e2,grad_w1);

		  //weight updates
		  RMSprop_with_nesterov_weight_update(ms1,grad_w1,w1,m1,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);
		  RMSprop_with_nesterov_weight_update(ms2,grad_w2,w2,m2,0.9f,learning_rate,b.CURRENT_BATCH->rows, momentum);

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
		  free(a_paired);

		  b.replace_current_batch_with_next();

	  }



	  train_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES; i++)
	  {

		  b.broadcast_batch_to_processes();

		  Matrix *d0 = scalarMul(b.CURRENT_BATCH,0.8);
		  Matrix *z1 = gpus.dot(d0,w1);
		  Matrix **a1_pair = maxout(z1,8);
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
		  cudaFree(d1->data);
		  cudaFree(d0->data);
		  free(a1_pair);

		  b.replace_current_batch_with_next();
	  }

	  //std::cout << "MAXOUT Train error: " << train_error << std::endl;



	  cv_error = 0;
	  for(int i = 0; i < b.TOTAL_BATCHES_CV; i++)
	  {
		  b.broadcast_batch_cv_to_processes();
		  Matrix *d0 = scalarMul(b.CURRENT_BATCH_CV,0.8);
		  Matrix *z1 = gpus.dot(d0,w1);
		  Matrix **a1_pair = maxout(z1,8);
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

	  //std::cout << "MAXOUT Cross validation error: " << cv_error << std::endl;

	}


	ASSERT(train_error < 0.02f,"mini-MNIST train error 17 epochs < 0.02.");
	ASSERT(cv_error < 0.22f, "mini-MNIST train error 17 epochs < 0.22.");
	*/
	std::vector<int> layers;
	layers.push_back(768);
	layers.push_back(512);


	BatchAllocator allocator = BatchAllocator();
	allocator.init(X,y,0.2,128,256,gpus, Distributed_weights);
	DeepNeuralNetwork net = DeepNeuralNetwork(layers,Classification, gpus, allocator, 10);
	net.EPOCHS = 1000;
	//net.LEARNING_RATE = 0.001;
	net.LEARNING_RATE = 0.001;
	net.train();

	if(gpus.MYRANK == 0)
	{
		cout << endl;
		cout << "Train error should be: 0.0025" << endl;
		cout << "Cross validation error should be: 0.13" << endl;
	}

	allocator = BatchAllocator();
	Matrix *t = to_host(create_t_matrix(to_gpu(y),10));
	allocator.init(X,t,0.2,128,256,gpus, Distributed_weights);
	net = DeepNeuralNetwork(layers,Regression, gpus, allocator, 10);
	net.EPOCHS = 100;
	net.PRINT_MISSCLASSIFICATION = true;
	net.OUTPUT_IS_PROBABILITY = true;
	net.LEARNING_RATE = 0.0003;
	net.train();

	if(gpus.MYRANK == 0)
	{
		cout << endl;
		cout << "Train error should be about: 0.05" << endl;
		cout << "Cross validation error should be about: 0.25" << endl;
	}


	/*
	if(gpus.MYGPUID == 0)
	{
		X = read_sparse_hdf5((path + "crowdflower_X_test.hdf5").c_str());
		y = read_sparse_hdf5((path + "crowdflower_y_test.hdf5").c_str());
	}
	else
	{
		X = empty_pinned_sparse(1,1,1);
		y = empty_pinned_sparse(1,1,1);
	}

	b = BatchAllocator();
	b.init(X,y,0.2,128,512,gpus, Distributed_weights_sparse);
	layers.clear();
	layers.push_back(400);
	layers.push_back(400);

	net = DeepNeuralNetwork(layers,Regression,gpus,b,24);
	net.EPOCHS = 4;
	net.TRANSITION_EPOCH = 4;
	net.LEARNING_RATE = 0.0001;
	net.OUTPUT_IS_PROBABILITY = true;
	net.train();
	*/


}
