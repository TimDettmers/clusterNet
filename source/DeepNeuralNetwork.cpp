#include <DeepNeuralNetwork.h>
#include <batchAllocator.h>
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <clusterNet.h>
#include <basicOps.cuh>

DeepNeuralNetwork::DeepNeuralNetwork(Matrix *X, Matrix *y, float cv_size, std::vector<int> lLayerSizes)
{
	m_BA = BatchAllocator(X,y,cv_size,128,512);
	m_gpus = ClusterNet(12345);

	LEARNING_RATE = 0.003;
	MOMENTUM = 0.5;

	lLayers = lLayerSizes;

	lDropout.push_back(0.2f);
	for(int i = 0;i < lLayers.size(); i++)
	{
		lDropout.push_back(0.5f);
	}
}

void DeepNeuralNetwork::init_weights()
{
	for(int i = 0;i < lLayers.size(); i++)
	{

	}
}

void DeepNeuralNetwork::train()
{
	  ClusterNet gpu = ClusterNet(12345);

	  Matrix *w1 = scalarMul(gpu.rand(m_BA.m_current_batch_X->cols,lLayers[0]),0.4*sqrt(6.0f/(784.0+lLayers[0])));
	  Matrix *w2 = scalarMul(gpu.rand(lLayers[0],10),0.4*sqrt(6.0f/(10.0+lLayers[0])));
	  Matrix *m1 = zeros(m_BA.m_current_batch_X->cols,lLayers[0]);
	  Matrix *m2 = zeros(lLayers[0],10);
	  Matrix *ms1 = zeros(m_BA.m_current_batch_X->cols,lLayers[0]);
	  Matrix *ms2 = zeros(m_BA.m_current_batch_X->cols,10);
	  Matrix *grad_w1_ms = zeros(m_BA.m_current_batch_X->cols,lLayers[0]);
	  Matrix *grad_w2_ms = zeros(lLayers[0],10);
	  Matrix *grad_w2 = empty(lLayers[0],10);
	  Matrix *grad_w1 = empty(m_BA.m_current_batch_X->cols,lLayers[0]);
	  float error = 0;
	  int epochs = 10;

	 for(int EPOCH = 0; EPOCH < epochs; EPOCH++)
	  {
		  std::cout << "EPOCH: " << EPOCH + 1 << std::endl;
		  //cudaMemGetInfo(&free, &total);
		  //std::cout << free << std::endl;
		  MOMENTUM += 0.01;
		  if(MOMENTUM > 0.95) MOMENTUM = 0.95;
		  for(int i = 0; i < m_BA.TOTAL_BATCHES; i++)
		  {
			  m_BA.allocate_next_batch_async();

			  //nesterov updates
			  scalarMul(m1,MOMENTUM,m1);
			  scalarMul(m2,MOMENTUM,m2);
			  add(w1,m1,w1);
			  add(w2,m1,w2);

			  Matrix *d0 = m_gpus.dropout(m_BA.m_current_batch_X,0.2);
			  //print_gpu_matrix(w1);
			  Matrix *z1 = m_gpus.dot(d0, w1);
			  logistic(z1, z1);
			  Matrix *d1 = m_gpus.dropout(z1,0.6);
			  Matrix *a2 = m_gpus.dot(d1,w2);
			  Matrix *out = softmax(a2);
			  Matrix *t = create_t_matrix(m_BA.m_current_batch_y,10);

			  //backprop
			  Matrix *e1 = sub(out, t);
			  Matrix *e2 = m_gpus.dotT(e1, w2);
			  m_gpus.Tdot(z1,e1,grad_w2);
			  logisticGrad(z1,z1);
			  mul(e2,z1,e2);
			  m_gpus.Tdot(m_BA.m_current_batch_X,e2,grad_w1);

			  RMSprop_with_nesterov_weight_update(ms1,grad_w1,w1,m1,0.9f,LEARNING_RATE,m_BA.m_current_batch_X->rows);
			  RMSprop_with_nesterov_weight_update(ms2,grad_w2,w2,m2,0.9f,LEARNING_RATE,m_BA.m_current_batch_X->rows);

			  cudaFree(e1->data);
			  cudaFree(e2->data);
			  cudaFree(z1->data);
			  cudaFree(a2->data);
			  cudaFree(out->data);
			  cudaFree(t->data);
			  cudaFree(d0->data);
			  cudaFree(d1->data);

			  m_BA.replace_current_batch_with_next();

		  }


		  //Matrix *sum_value = sum(w1);
		  //std::cout << "weight 1 Sum: " << to_host(sum_value)->data[0] << std::endl;

		  error = 0;
		  for(int i = 0; i < m_BA.TOTAL_BATCHES; i++)
		  {
			  m_BA.allocate_next_batch_async();

			  Matrix *a1 = m_gpus.dot(m_BA.m_current_batch_X,w1);

			  logistic(a1, a1);
			  Matrix *a2 = m_gpus.dot(a1,w2);

			  Matrix *out = softmax(a2);


			  Matrix *result = argmax(out);

			  Matrix *eq = equal(result,m_BA.m_current_batch_y);
			  Matrix *sum_mat = sum(eq);
			  float sum_value = to_host(sum_mat)->data[0];

			  //std::cout << "Error count: " << 128.0f - sum_value << std::endl;
			  error += (m_BA.m_current_batch_X->rows - sum_value);


			  cudaFree(a1->data);
			  cudaFree(a2->data);
			  cudaFree(out->data);
			  cudaFree(result->data);
			  cudaFree(eq->data);
			  cudaFree(sum_mat->data);

			  m_BA.replace_current_batch_with_next();
		  }


		  std::cout << "Train error: " << error/((1.0f - 0.15)*70000.0f)<< std::endl;


		  error = 0;
		  for(int i = 0; i < m_BA.TOTAL_BATCHES_CV; i++)
		  {
			  m_BA.allocate_next_cv_batch_async();
			  Matrix *a1 = m_gpus.dot(m_BA.m_current_batch_cv_X,w1);

			  logistic(a1, a1);
			  Matrix *a2 = m_gpus.dot(a1,w2);

			  Matrix *out = softmax(a2);

			  Matrix *result = argmax(out);

			  Matrix *eq = equal(result,m_BA.m_current_batch_cv_y);
			  Matrix *sum_mat = sum(eq);
			  float sum_value = to_host(sum_mat)->data[0];

			  //std::cout << "Error count: " << m_gpus.m_total_batches_cv - sum_value << std::endl;
			  error += (m_BA.m_current_batch_cv_X->rows  - sum_value);


			  cudaFree(a1->data);
			  cudaFree(a2->data);
			  cudaFree(out->data);
			  cudaFree(result->data);
			  cudaFree(eq->data);
			  cudaFree(sum_mat->data);

			  m_BA.replace_current_cv_batch_with_next();
		  }

		  std::cout << "Cross validation error: " << error/(0.15*70000) << std::endl;
	  }

}
