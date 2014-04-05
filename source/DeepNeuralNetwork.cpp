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

DeepNeuralNetwork::DeepNeuralNetwork(std::vector<int> lLayerSizes, Networktype_t net_type, ClusterNet gpus, BatchAllocator allocator, int categories)
{
	m_gpus = gpus;
	m_BA = allocator;

	EPOCHS = 100;
	TRANSITION_EPOCH = 75;
	LEARNING_RATE = 0.003;
	MOMENTUM = 0.5;
	OUTPUT_IS_PROBABILITY = false;
	m_output_dim = categories;
	m_net_type = net_type;
	m_update_type = RMSProp;

	init_network_layout(lLayerSizes);
	init_weights();

}

void DeepNeuralNetwork::init_network_layout(std::vector<int> lLayerSizes)
{
	m_lLayers = lLayerSizes;
	if(m_net_type == Classification){ m_costFunction = Root_Squared_Error;}
	if(m_net_type == Regression){ m_costFunction = Cross_Entropy; }

	lDropout.push_back(0.2f);
	for(int i = 0;i < m_lLayers.size(); i++)
	{
		if(m_net_type == Classification){ lUnits.push_back(Logistic); }
		if(m_net_type == Regression){ lUnits.push_back(Rectified_Linear); }
		lDropout.push_back(0.5f);
	}
	if(m_net_type == Classification){ lUnits.push_back(Softmax); }
	if(m_net_type == Regression){ lUnits.push_back(Linear); }
}

void DeepNeuralNetwork::init_weights()
{

	int output_size = m_output_dim;
	if(m_net_type == Regression)
		output_size = m_BA.CURRENT_BATCH_Y->cols;

	if(m_BA.BATCH_METHOD == Distributed_weights)
	{
		W.push_back(m_gpus.distributed_uniformSqrtWeight(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		M.push_back(m_gpus.distributed_zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		MS.push_back(m_gpus.distributed_zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		GRAD.push_back(m_gpus.distributed_zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		for(int i = 0;i < (m_lLayers.size()-1); i++)
		{
			W.push_back(m_gpus.distributed_uniformSqrtWeight(m_lLayers[i],m_lLayers[i+1]));
			M.push_back(m_gpus.distributed_zeros(m_lLayers[i],m_lLayers[i+1]));
			MS.push_back(m_gpus.distributed_zeros(m_lLayers[i],m_lLayers[i+1]));
			GRAD.push_back(m_gpus.distributed_zeros(m_lLayers[i],m_lLayers[i+1]));
		}
		W.push_back(m_gpus.distributed_uniformSqrtWeight(m_lLayers.back(), output_size));
		M.push_back(m_gpus.distributed_zeros(m_lLayers.back(),output_size));
		MS.push_back(m_gpus.distributed_zeros(m_lLayers.back(),output_size));
		GRAD.push_back(m_gpus.distributed_zeros(m_lLayers.back(),output_size));
	}
	else
	{
		W.push_back(m_gpus.uniformSqrtWeight(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		M.push_back(zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		MS.push_back(zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		GRAD.push_back(zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		for(int i = 0;i < (m_lLayers.size()-1); i++)
		{
			W.push_back(m_gpus.uniformSqrtWeight(m_lLayers[i],m_lLayers[i+1]));
			M.push_back(zeros(m_lLayers[i],m_lLayers[i+1]));
			MS.push_back(zeros(m_lLayers[i],m_lLayers[i+1]));
			GRAD.push_back(zeros(m_lLayers[i],m_lLayers[i+1]));
		}
		W.push_back(m_gpus.uniformSqrtWeight(m_lLayers.back(),output_size));
		M.push_back(zeros(m_lLayers.back(),output_size));
		MS.push_back(zeros(m_lLayers.back(),output_size));
		GRAD.push_back(zeros(m_lLayers.back(),output_size));
	}
}

void DeepNeuralNetwork::train()
{
	if(OUTPUT_IS_PROBABILITY)
		lUnits.back() = Double_Rectified_Linear;

	float original_learning_rate = LEARNING_RATE;
	for(int EPOCH = 0; EPOCH < EPOCHS; EPOCH++)
	{
		if(m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0))
			std::cout << "EPOCH: " << EPOCH + 1 << std::endl;
		MOMENTUM += 0.01;
		if(MOMENTUM > 0.95) MOMENTUM = 0.95;

		if(EPOCH >= TRANSITION_EPOCH)
			LEARNING_RATE = original_learning_rate / (EPOCH - TRANSITION_EPOCH + 1.0f);

		if(EPOCH == TRANSITION_EPOCH)
		{
			m_update_type = NoMomentum;
			for(int i = 0; i < lDropout.size(); i++)
				lDropout[i] = lDropout[i] / 2.0;
		}






		for(int i = 0; i < m_BA.TOTAL_BATCHES; i++)
		{
		  nesterov_updates();
		  m_BA.broadcast_batch_to_processes();
		  feedforward(Dropout);
		  m_BA.allocate_next_batch_async();
		  backprop();

		  weight_updates();
		  free_variables();

		  m_BA.replace_current_batch_with_next();
		}

		train_error();
		cross_validation_error();
	}

	m_BA.finish_batch_allocator();
}

void DeepNeuralNetwork::backprop()
{
	  //backprop
	  if(m_net_type == Classification)
	  {
		  Matrix *t = create_t_matrix(m_BA.CURRENT_BATCH_Y,m_output_dim);
	  	  E.push_back(sub(Z.back(), t));
	  	  cudaFree(t->data);
	  }
	  else
	  {
		  E.push_back(sub(Z.back(), m_BA.CURRENT_BATCH_Y));
	  }
	  for(int i = W.size()-1; i > 0; i--)
	  {
		  m_gpus.Tdot(Z[i],E.back(),GRAD[i]);
		  derivative_function(i, Z[i]);
		  E.push_back(m_gpus.dotT(E.back(), W[i]));
		  mul(E.back(),Z[i],E.back());
	  }
	  m_gpus.Tdot(Z[0],E.back(),GRAD[0]);

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
	  {
		  if(m_update_type == RMSProp)
			  RMSprop_with_nesterov_weight_update(MS[i],GRAD[i],W[i],M[i],0.9f,LEARNING_RATE,m_BA.CURRENT_BATCH->rows);
		  else if(m_update_type == NoMomentum)
		  {
			 scalarMul(GRAD[i],LEARNING_RATE/(float)m_BA.CURRENT_BATCH->rows,GRAD[i]);
			 sub(W[i],GRAD[i],W[i]);
		  }
	  }
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

float DeepNeuralNetwork::get_errors(Batchtype_t batch_t)
{
	float errors = 0;
	if(m_net_type == Classification)
	{
		Matrix *result = argmax(Z.back());
		Matrix *eq;
		if(batch_t == Train){	eq = equal(result,m_BA.CURRENT_BATCH_Y);}
		else{	eq = equal(result,m_BA.CURRENT_BATCH_CV_Y);}

		float sum_value = sum(eq);
		errors = (Z.back()->rows  - sum_value);
		cudaFree(result->data);
		cudaFree(eq->data);
	}
	else
	{
		Matrix *sqrErr = squared_error(Z.back(),batch_t == Train ? m_BA.CURRENT_BATCH_Y : m_BA.CURRENT_BATCH_CV_Y);
		errors = sum(sqrErr);
		errors /=  m_BA.CURRENT_BATCH_Y->cols;
		errors = sqrt(errors);
		cudaFree(sqrErr->data);
	}


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
		case Double_Rectified_Linear:
			doubleRectifiedLinear(A,A);
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
		case Double_Rectified_Linear:
			double_rectified_linear_derivative(A,A);
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
	  float errors = 0;
	  for(int i = 0; i < m_BA.TOTAL_BATCHES; i++)
	  {

		  //m_BA.slice_batch();
		  m_BA.broadcast_batch_to_processes();
		  feedforward(Train_error);
		  m_BA.allocate_next_batch_async();

		  errors += get_errors(Train);

		  free_variables();

		  m_BA.replace_current_batch_with_next();
	  }

	  if(m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0))
		  std::cout << "Train error: " << errors/m_BA.TRAIN_SET_SIZE << std::endl;
}


void DeepNeuralNetwork::cross_validation_error()
{
	  float errors = 0;
	  for(int i = 0; i < m_BA.TOTAL_BATCHES_CV; i++)
	  {
		  m_BA.broadcast_batch_cv_to_processes();
		  feedforward(CV_error);
		  m_BA.allocate_next_cv_batch_async();
		  errors += get_errors(CV);

		  free_variables();

		  m_BA.replace_current_cv_batch_with_next();
	  }

	  if(m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0))
		  std::cout << "Cross validation error: " << errors/m_BA.CV_SET_SIZE << std::endl;
}

Matrix* DeepNeuralNetwork::predict(Matrix *X)
{
	int batch_size = 128;
	int rows = X->rows;
	int cols = X->cols;

	if(m_gpus.MYGPUID == 0)
		for(int i = 1; i < m_gpus.PCIe_RANKS.size();i++)
		{
			MPI_Send(&rows,1,MPI_INT,m_gpus.PCIe_RANKS[i],999,MPI_COMM_WORLD);
			MPI_Send(&cols,1,MPI_INT,m_gpus.PCIe_RANKS[i],999,MPI_COMM_WORLD);
		}

	else
	{
		MPI_Recv(&rows,1,MPI_INT,m_gpus.PCIe_RANKS[0],999,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&cols,1,MPI_INT,m_gpus.PCIe_RANKS[0],999,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	Matrix *batch = empty(batch_size,cols);
	Matrix *off_batch = empty(rows % batch_size,cols);
	Matrix *buffer = empty_cpu(batch_size,cols);
	Matrix *off_buffer = empty_cpu(rows % batch_size,cols);

	Matrix *out;

	int full_batches = (rows / batch_size)-1;
	for(int i = 0; i < (rows/batch_size) + 1; i++)
	{
		if(m_gpus.MYGPUID == 0)
		{
			if(X->isSparse == 0)
			{
				if(i < full_batches)
					cudaMemcpy(&batch->data[0],&X->data[(i*X->cols)*batch_size],batch->bytes,cudaMemcpyDefault);
				else
					cudaMemcpy(&off_batch->data[0],&X->data[(i*X->cols)*batch_size],off_batch->bytes,cudaMemcpyDefault);
			}
			else
			{
				if(i  < full_batches)
				{
					slice_sparse_to_dense(X,buffer,i*batch_size,batch_size);
					cudaMemcpy(&batch->data[0],&buffer->data[0],buffer->bytes,cudaMemcpyDefault);

				}
				else
				{
					slice_sparse_to_dense(X,off_buffer,i*batch_size,X->rows % batch_size);
					cudaMemcpy(&off_batch->data[0],&off_buffer->data[0],off_buffer->bytes,cudaMemcpyDefault);
				}
			}

			if(i  < full_batches)
				for(int i = 1; i < m_gpus.PCIe_RANKS.size();i++)
				{
					MPI_Send(batch->data,batch->size,MPI_FLOAT,m_gpus.PCIe_RANKS[i],999,MPI_COMM_WORLD);
				}
			else
				for(int i = 1; i < m_gpus.PCIe_RANKS.size();i++)
					MPI_Send(off_batch->data,off_batch->size,MPI_FLOAT,m_gpus.PCIe_RANKS[i],999,MPI_COMM_WORLD);


		}
		else
		{

			if(i  < full_batches)
			{
				MPI_Recv(batch->data,batch->size,MPI_FLOAT,m_gpus.PCIe_RANKS[0],999,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			else
				MPI_Recv(off_batch->data,off_batch->size,MPI_FLOAT,m_gpus.PCIe_RANKS[0],999,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		}




		if(i  < full_batches)
		{
			to_col_major(batch,batch);
			Z.push_back(batch);
		}
		else
		{
			to_col_major(off_batch,off_batch);
			Z.push_back(off_batch);
		}

		//feed forward
		for(int j = 0; j < W.size(); j++)
		{
			Z.push_back(m_gpus.dot(Z.back(), W[j]));
			activation_function(j, Z.back());
		}

		if(m_gpus.MYGPUID == 0)
		{
			if(i == 0)
				out = empty_cpu(X->rows,Z.back()->cols);

			Matrix *host = to_host(Z.back());
			for(int k = 0; k < host->size; k++)
				out->data[(i*batch_size*host->cols) + k] = host->data[k];

			free(host->data);

		}



		free_variables();



	}

	cudaFree(batch->data);
	cudaFree(off_batch->data);

	return out;


}
