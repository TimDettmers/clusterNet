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

DeepNeuralNetwork::DeepNeuralNetwork(std::vector<int> lLayerSizes, Networktype_t net_type, ClusterNet gpus, BatchAllocator allocator)
{
	m_gpus = gpus;
	m_BA = allocator;

	EPOCHS = 100;
	LEARNING_RATE = 0.003;
	MOMENTUM = 0.5;

	init_network_layout(lLayerSizes,net_type);
	init_weights();

}

void DeepNeuralNetwork::init_network_layout(std::vector<int> lLayerSizes, Networktype_t net_type)
{
	lLayers = lLayerSizes;
	if(net_type == Classification){ m_costFunction = Root_Squared_Error;}
		if(net_type == Regression){ m_costFunction = Cross_Entropy; }

		lDropout.push_back(0.2f);
		for(int i = 0;i < lLayers.size(); i++)
		{
			if(net_type == Classification){ lUnits.push_back(Logistic); }
			if(net_type == Regression){ lUnits.push_back(Rectified_Linear); }
			lDropout.push_back(0.5f);
		}
		if(net_type == Classification){ lUnits.push_back(Softmax); }
		if(net_type == Regression){ lUnits.push_back(Linear); }
}

void DeepNeuralNetwork::init_weights()
{
	if(m_BA.BATCH_METHOD == Distributed_weights)
	{
		W.push_back(m_gpus.distributed_uniformSqrtWeight(m_BA.CURRENT_BATCH->cols,lLayers[0]));
		M.push_back(m_gpus.distributed_zeros(m_BA.CURRENT_BATCH->cols,lLayers[0]));
		MS.push_back(m_gpus.distributed_zeros(m_BA.CURRENT_BATCH->cols,lLayers[0]));
		GRAD.push_back(m_gpus.distributed_zeros(m_BA.CURRENT_BATCH->cols,lLayers[0]));
		for(int i = 0;i < (lLayers.size()-1); i++)
		{
			W.push_back(m_gpus.distributed_uniformSqrtWeight(lLayers[i],lLayers[i+1]));
			M.push_back(m_gpus.distributed_zeros(lLayers[i],lLayers[i+1]));
			MS.push_back(m_gpus.distributed_zeros(lLayers[i],lLayers[i+1]));
			GRAD.push_back(m_gpus.distributed_zeros(lLayers[i],lLayers[i+1]));
		}
		W.push_back(m_gpus.distributed_uniformSqrtWeight(lLayers.back(),10));
		M.push_back(m_gpus.distributed_zeros(lLayers.back(),10));
		MS.push_back(m_gpus.distributed_zeros(lLayers.back(),10));
		GRAD.push_back(m_gpus.distributed_zeros(lLayers.back(),10));
	}
	else
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
	}
}

void DeepNeuralNetwork::train()
{
	for(int EPOCH = 0; EPOCH < EPOCHS; EPOCH++)
	{
		if(m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0))
			std::cout << "EPOCH: " << EPOCH + 1 << std::endl;
		MOMENTUM += 0.01;
		if(MOMENTUM > 0.95) MOMENTUM = 0.95;
		for(int i = 0; i < m_BA.TOTAL_BATCHES; i++)
		{
		  m_BA.broadcast_batch_to_PCI2();

		  nesterov_updates();
		  feedforward(Dropout);
		  m_BA.allocate_next_batch_async2();
		  backprop();

		  weight_updates();
		  free_variables();

		  m_BA.replace_current_batch_with_next2();
		}

		train_error();
		cross_validation_error();
	}

	m_BA.finish_batch_allocator();
}

void DeepNeuralNetwork::backprop()
{
	  //backprop
	  Matrix *t = create_t_matrix(m_BA.CURRENT_BATCH_Y,10);
	  E.push_back(sub(Z.back(), t));
	  for(int i = W.size()-1; i > 0; i--)
	  {
		  m_gpus.Tdot(Z[i],E.back(),GRAD[i]);
		  derivative_function(i, Z[i]);
		  E.push_back(m_gpus.dotT(E.back(), W[i]));
		  mul(E.back(),Z[i],E.back());
	  }
	  m_gpus.Tdot(Z[0],E.back(),GRAD[0]);
	  cudaFree(t->data);
}

void DeepNeuralNetwork::free_variables()
{
	  for(int i = 0; i < D.size(); i++)
			cudaFree(D[i]->data);
	  D.clear();
	  for(int i = 1; i < Z.size(); i++)
			cudaFree(Z[i]->data);
	  Z.clear();
	  for(int i = 0; i < E.size(); i++)
			cudaFree(E[i]->data);
	  E.clear();

}

void DeepNeuralNetwork::weight_updates()
{
	  for(int i = 0;i < GRAD.size(); i++)
		  RMSprop_with_nesterov_weight_update(MS[i],GRAD[i],W[i],M[i],0.9f,LEARNING_RATE,m_BA.CURRENT_BATCH->rows);
}


void DeepNeuralNetwork::feedforward(FeedForward_t ff)
{
	if(ff == Dropout)
	{
		Z.push_back(m_BA.CURRENT_BATCH);
		for(int i = 0; i < W.size(); i++)
		{
		  D.push_back(m_gpus.dropout(Z.back(),lDropout[i]));
		  Z.push_back(m_gpus.dot(D.back(), W[i]));
		  activation_function(i, Z.back());
		}
	}
	else
	{
		if(ff == Train_error){ Z.push_back(m_BA.CURRENT_BATCH);}
		else{ Z.push_back(m_BA.CURRENT_BATCH_CV);}

		for(int i = 0; i < W.size(); i++)
		{
			Z.push_back(m_gpus.dot(Z.back(), W[i]));
			activation_function(i, Z.back());
		}
	}

}

int DeepNeuralNetwork::get_classification_errors(Batchtype_t batch_t)
{
	  Matrix *result = argmax(Z.back());
	  Matrix *eq;
	  if(batch_t == Train){	eq = equal(result,m_BA.CURRENT_BATCH_Y);}
	  else{	eq = equal(result,m_BA.CURRENT_BATCH_CV_Y);}

	  float sum_value = sum(eq);
	  int errors = (Z.back()->rows  - sum_value);

	  cudaFree(result->data);
	  cudaFree(eq->data);

	  return errors;
}

void DeepNeuralNetwork::activation_function(int layer, Matrix * A)
{
	switch(lUnits[layer])
	{
		case Logistic:
			logistic(A,A);
			break;
		case Rectified_Linear:
			rectified_linear(A,A);
			break;
		case Softmax:
			softmax(A,A);
			break;
		case Linear:
			break;
	}
}

void DeepNeuralNetwork::derivative_function(int layer, Matrix * A)
{
	switch(lUnits[layer-1])
	{
		case Logistic:
			logisticGrad(A,A);
			break;
		case Rectified_Linear:
			rectified_linear_derivative(A,A);
			break;
		default:
			throw "Unknown unit";
			break;
	}
}

void DeepNeuralNetwork::nesterov_updates()
{
	//nesterov updates
	for(int i = 0;i < M.size(); i++)
	{
		scalarMul(M[i],MOMENTUM,M[i]);
		add(W[i],M[i],W[i]);
	}
}

void DeepNeuralNetwork::train_error()
{
	  int errors = 0;
	  for(int i = 0; i < m_BA.TOTAL_BATCHES; i++)
	  {

		  m_BA.broadcast_batch_to_PCI2();
		  feedforward(Train_error);
		  m_BA.allocate_next_batch_async2();

		  errors += get_classification_errors(Train);

		  free_variables();

		  m_BA.replace_current_batch_with_next2();
	  }

	  if(m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0))
		  std::cout << "Train error: " << errors/(float)m_BA.TRAIN_SET_SIZE << std::endl;
}


void DeepNeuralNetwork::cross_validation_error()
{
	  int errors = 0;
	  for(int i = 0; i < m_BA.TOTAL_BATCHES_CV; i++)
	  {
		  m_BA.broadcast_cv_batch_to_PCI2();
		  feedforward(CV_error);
		  m_BA.allocate_next_cv_batch_async2();
		  errors += get_classification_errors(CV);

		  free_variables();

		  m_BA.replace_current_cv_batch_with_next2();
	  }

	  if(m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0))
		  std::cout << "Cross validation error: " << errors/(float)m_BA.CV_SET_SIZE << std::endl;
}
