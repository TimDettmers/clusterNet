#include <DeepNeuralNetwork.h>
#include <batchAllocator.h>
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <clusterNet.h>
#include <basicOps.cuh>
#include <util.cuh>

using std::cout;
using std::endl;

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
	W.push_back(m_gpus.uniformSqrtWeight(m_BA.CURRENT_BATCH->cols,lLayers[0]));
	M.push_back(zeros(m_BA.CURRENT_BATCH->cols,lLayers[0]));
	MS.push_back(zeros(m_BA.CURRENT_BATCH->cols,lLayers[0]));
	GRAD.push_back(zeros(m_BA.CURRENT_BATCH->cols,lLayers[0]));
	for(int i = 0;i < (lLayers.size()-1); i++)
	{
		W.push_back(m_gpus.uniformSqrtWeight(lLayers[i],lLayers[i+1]));
		M.push_back(zeros(lLayers[i],lLayers[i+1]));
		MS.push_back(zeros(lLayers[i],lLayers[i+1]));
		GRAD.push_back(zeros(lLayers[i],lLayers[i+1]));
	}
	W.push_back(m_gpus.uniformSqrtWeight(lLayers.back(),10));
	M.push_back(zeros(lLayers.back(),10));
	MS.push_back(zeros(lLayers.back(),10));
	GRAD.push_back(zeros(lLayers.back(),10));

	cout << W.size() << endl;
	cout << lDropout.size() << endl;
}

void DeepNeuralNetwork::train()
{
	  ClusterNet gpu = ClusterNet(12345);

	  init_weights();
	  float error = 0;
	  int epochs = 100;

	 size_t free, total;
	 gpu.tick();
	 for(int EPOCH = 0; EPOCH < epochs; EPOCH++)
	  {
		  std::cout << "EPOCH: " << EPOCH + 1 << std::endl;
		  cudaMemGetInfo(&free, &total);
		  std::cout << free << std::endl;
		  MOMENTUM += 0.01;
		  if(MOMENTUM > 0.95) MOMENTUM = 0.95;
		  for(int i = 0; i < m_BA.TOTAL_BATCHES; i++)
		  {
			  m_BA.allocate_next_batch_async();

			  //nesterov updates
			  for(int i = 0;i < M.size(); i++)
			  {
				  scalarMul(M[i],MOMENTUM,M[i]);
				  add(W[i],M[i],W[i]);
			  }

			  Z.push_back(m_BA.CURRENT_BATCH);
			  for(int i = 0; i < W.size(); i++)
			  {
				  D.push_back(m_gpus.dropout(Z.back(),lDropout[i]));
				  Z.push_back(m_gpus.dot(D.back(), W[i]));
				  if(i == W.size() - 1)
					  softmax(Z.back(),Z.back());
				  else
					  logistic(Z.back(),Z.back());
			  }
			  Matrix *t = create_t_matrix(m_BA.m_current_batch_y,10);

			  //backprop

			  E.push_back(sub(Z.back(), t));
			  for(int i = W.size()-1; i > 0; i--)
			  {
				  m_gpus.Tdot(Z[i],E.back(),GRAD[i]);
				  logisticGrad(Z[i],Z[i]);
				  E.push_back(m_gpus.dotT(E.back(), W[i]));
				  mul(E.back(),Z[i],E.back());
			  }
			  m_gpus.Tdot(Z[0],E.back(),GRAD[0]);

			  for(int i = 0;i < GRAD.size(); i++)
			  {
				  RMSprop_with_nesterov_weight_update(MS[i],GRAD[i],W[i],M[i],0.9f,LEARNING_RATE,m_BA.CURRENT_BATCH->rows);
			  }

			  for(int i = 0; i < D.size(); i++)
			  		cudaFree(D[i]->data);
			  D.clear();
			  for(int i = 1; i < Z.size(); i++)
			  		cudaFree(Z[i]->data);
  			  Z.clear();
  			  for(int i = 1; i < E.size(); i++)
  					cudaFree(E[i]->data);
  			  E.clear();
			  cudaFree(t->data);

			  m_BA.replace_current_batch_with_next();




		  }


		  //Matrix *sum_value = sum(w1);
		  //std::cout << "weight 1 Sum: " << to_host(sum_value)->data[0] << std::endl;

		  error = 0;
		  for(int i = 0; i < m_BA.TOTAL_BATCHES; i++)
		  {
			  m_BA.allocate_next_batch_async();

			  Z.push_back(m_BA.CURRENT_BATCH);
			  for(int i = 0; i < W.size(); i++)
			  {
				  Z.push_back(m_gpus.dot(Z.back(), W[i]));
				  if(i == W.size() - 1)
					  softmax(Z.back(),Z.back());
				  else
					  logistic(Z.back(),Z.back());
			  }
			  Matrix *result = argmax(Z.back());

			  Matrix *eq = equal(result,m_BA.m_current_batch_y);
			  Matrix *sum_mat = sum(eq);
			  float sum_value = to_host(sum_mat)->data[0];

			  //std::cout << "Error count: " << 128.0f - sum_value << std::endl;
			  error += (m_BA.CURRENT_BATCH->rows - sum_value);
			  for(int i = 1; i < Z.size(); i++)
				  cudaFree(Z[i]->data);
			  Z.clear();

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
			  Z.push_back(m_BA.CURRENT_BATCH_CV);
			  for(int i = 0; i < W.size(); i++)
			  {
				  Z.push_back(m_gpus.dot(Z.back(), W[i]));
				  if(i == W.size() - 1)
					  softmax(Z.back(),Z.back());
				  else
					  logistic(Z.back(),Z.back());
			  }

			  Matrix *result = argmax(Z.back());

			  Matrix *eq = equal(result,m_BA.m_current_batch_cv_y);
			  Matrix *sum_mat = sum(eq);
			  float sum_value = to_host(sum_mat)->data[0];

			  //std::cout << "Error count: " << m_gpus.m_total_batches_cv - sum_value << std::endl;
			  error += (m_BA.CURRENT_BATCH_CV->rows  - sum_value);



			  for(int i = 1; i < Z.size(); i++)
				  cudaFree(Z[i]->data);
			  Z.clear();
			  cudaFree(result->data);
			  cudaFree(eq->data);
			  cudaFree(sum_mat->data);

			  m_BA.replace_current_cv_batch_with_next();
		  }

		  std::cout << "Cross validation error: " << error/(0.15*70000) << std::endl;
	  }
	 gpu.tock();

}
