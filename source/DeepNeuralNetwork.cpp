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
	PRINT_MISSCLASSIFICATION = net_type == Classification ? true : false;
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

	if(m_BA.BATCH_METHOD == Distributed_weights || m_BA.BATCH_METHOD == Distributed_weights_sparse)
	{
		W.push_back(m_gpus.distributed_uniformSqrtWeight(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		B.push_back(zeros(1,m_lLayers[0]));
		M.push_back(m_gpus.distributed_zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		B_M.push_back(zeros(1,m_lLayers[0]));
		MS.push_back(m_gpus.distributed_zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		B_MS.push_back(zeros(1,m_lLayers[0]));
		GRAD.push_back(m_gpus.distributed_zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		B_GRAD.push_back(zeros(1,m_lLayers[0]));
		for(int i = 0;i < (m_lLayers.size()-1); i++)
		{
			W.push_back(m_gpus.distributed_uniformSqrtWeight(m_lLayers[i],m_lLayers[i+1]));
			B.push_back(zeros(1,m_lLayers[i+1]));
			M.push_back(m_gpus.distributed_zeros(m_lLayers[i],m_lLayers[i+1]));
			B_M.push_back(zeros(1,m_lLayers[i+1]));
			MS.push_back(m_gpus.distributed_zeros(m_lLayers[i],m_lLayers[i+1]));
			B_MS.push_back(zeros(1,m_lLayers[i+1]));
			GRAD.push_back(m_gpus.distributed_zeros(m_lLayers[i],m_lLayers[i+1]));
			B_GRAD.push_back(zeros(1,m_lLayers[i+1]));
		}
		W.push_back(m_gpus.distributed_uniformSqrtWeight(m_lLayers.back(), output_size));
		B.push_back(zeros(1, output_size));
		M.push_back(m_gpus.distributed_zeros(m_lLayers.back(),output_size));
		B_M.push_back(zeros(1, output_size));
		MS.push_back(m_gpus.distributed_zeros(m_lLayers.back(),output_size));
		B_MS.push_back(zeros(1, output_size));
		GRAD.push_back(m_gpus.distributed_zeros(m_lLayers.back(),output_size));
		B_GRAD.push_back(zeros(1, output_size));
	}
	else
	{
		W.push_back(m_gpus.uniformSqrtWeight(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		B.push_back(zeros(1,m_lLayers[0]));
		M.push_back(zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		B_M.push_back(zeros(1,m_lLayers[0]));
		MS.push_back(zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		B_MS.push_back(zeros(1,m_lLayers[0]));
		GRAD.push_back(zeros(m_BA.CURRENT_BATCH->cols,m_lLayers[0]));
		B_GRAD.push_back(zeros(1,m_lLayers[0]));
		for(int i = 0;i < (m_lLayers.size()-1); i++)
		{
			W.push_back(m_gpus.uniformSqrtWeight(m_lLayers[i],m_lLayers[i+1]));
			B.push_back(zeros(1,m_lLayers[i+1]));
			M.push_back(zeros(m_lLayers[i],m_lLayers[i+1]));
			B_M.push_back(zeros(1,m_lLayers[i+1]));
			MS.push_back(zeros(m_lLayers[i],m_lLayers[i+1]));
			B_MS.push_back(zeros(1,m_lLayers[i+1]));
			GRAD.push_back(zeros(m_lLayers[i],m_lLayers[i+1]));
			B_GRAD.push_back(zeros(1,m_lLayers[i+1]));
		}
		W.push_back(m_gpus.uniformSqrtWeight(m_lLayers.back(),output_size));
		B.push_back(zeros(1,output_size));
		M.push_back(zeros(m_lLayers.back(),output_size));
		B_M.push_back(zeros(1,output_size));
		MS.push_back(zeros(m_lLayers.back(),output_size));
		B_MS.push_back(zeros(1,output_size));
		GRAD.push_back(zeros(m_lLayers.back(),output_size));
		B_GRAD.push_back(zeros(1,output_size));
	}

}

void DeepNeuralNetwork::train()
{
	//if(OUTPUT_IS_PROBABILITY)
		//lUnits.back() = Double_Rectified_Linear;


	float original_learning_rate = LEARNING_RATE;
	for(int EPOCH = 0; EPOCH < EPOCHS; EPOCH++)
	{
		if(m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0))
			std::cout << "EPOCH: " << EPOCH + 1 << std::endl;
		MOMENTUM += 0.01;
		if(MOMENTUM > 0.95) MOMENTUM = 0.95;

		if(EPOCH > TRANSITION_EPOCH)
			LEARNING_RATE = original_learning_rate / (EPOCH - TRANSITION_EPOCH);

		if(EPOCH == TRANSITION_EPOCH)
		{
			cout << "Transition point reached: Halving dropout!" << endl;
			m_update_type = NoMomentum;
			for(int i = 0; i < lDropout.size(); i++)
				lDropout[i] = lDropout[i] / 2.0;
		}

		for(int i = 0; i < m_BA.TOTAL_BATCHES; i++)
		{
		  nesterov_updates();
		  m_BA.broadcast_batch_to_processes();
		  cout << "pre feed forward" << endl;
		  MPI_Barrier(MPI_COMM_WORLD);
		  feedforward(Dropout);

		  //if(i == 0)
			//  printmat(Z.back(),1,100);
		  cout << "pre backprop" << endl;
		  MPI_Barrier(MPI_COMM_WORLD);
		  if(m_BA.CURRENT_BATCH->isSparse == 0)
			  m_BA.allocate_next_batch_async();
		  backprop();
		  cout << "post backprop" << endl;
		  MPI_Barrier(MPI_COMM_WORLD);

		  weight_updates();
		  free_variables();

		  if(m_BA.CURRENT_BATCH->isSparse == 1)
			  m_BA.allocate_next_batch_async();
		  m_BA.replace_current_batch_with_next();
		}

		cout << "pre train and cv calc" << endl;
		MPI_Barrier(MPI_COMM_WORLD);

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
		  Matrix *bias_activation = ones(1,E.back()->rows);
		  m_gpus.Tdot(Z[i],E.back(),GRAD[i]);
		  m_gpus.dot(bias_activation,E.back(),B_GRAD[i]);
		  cudaFree(bias_activation->data);
		  derivative_function(i, Z[i]);
		  E.push_back(m_gpus.dotT(E.back(), W[i]));
		  cout << "sum Z[i]: " << sum(Z[i]) << endl;
		  mul(E.back(),Z[i],E.back());
	  }
	  Matrix *bias_activation = ones(1,E.back()->rows);
	  cout << "sum pre: " << sum(bias_activation) << endl;
	  cout << m_BA.CURRENT_BATCH->cols << " vs " << GRAD[0]->rows << endl;
	  cout << E.back()->cols << " vs " << GRAD[0]->cols << endl;
	  m_gpus.Tdot(m_BA.CURRENT_BATCH,E.back(),GRAD[0]);
	  size_t free, total;
	  cudaMemGetInfo(&free, &total);
	  cout << "memory: " << free << endl;
	  m_gpus.dot(bias_activation,E.back(),B_GRAD[0]);
	  cudaFree(bias_activation->data);

}

void DeepNeuralNetwork::free_variables()
{
	  for(int i = 0; i < D.size(); i++)
	  {
		  if(D[i]->isSparse == 0)
			cudaFree(D[i]->data);
		  else
		  {
			  cudaFree(D[i]->data);
			  cudaFree(D[i]->idx_cols);
			  cudaFree(D[i]->ptr_rows);
		  }
	  }
	  D.clear();

	  for(int i = 1; i < Z.size(); i++)
	  {
		  if(Z[i]->isSparse == 0)
			cudaFree(Z[i]->data);
		  else
		  {
			  cudaFree(Z[i]->data);
			  cudaFree(Z[i]->idx_cols);
			  cudaFree(Z[i]->ptr_rows);
		  }
	  }
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
		  {
			  RMSprop_with_nesterov_weight_update(MS[i],GRAD[i],W[i],M[i],0.9f,LEARNING_RATE,m_BA.CURRENT_BATCH->rows);
			  RMSprop_with_nesterov_weight_update(B_MS[i],B_GRAD[i],B[i],B_M[i],0.9f,LEARNING_RATE,m_BA.CURRENT_BATCH->rows);
		  }
		  else if(m_update_type == NoMomentum)
		  {
			 scalarMul(GRAD[i],LEARNING_RATE/(float)m_BA.CURRENT_BATCH->rows,GRAD[i]);
			 sub(W[i],GRAD[i],W[i]);
		  }
	  }
}


void DeepNeuralNetwork::feedforward(FeedForward_t ff)
{
	//scale up the weights
	if(ff == Dropout)
	{
		Z.push_back(m_BA.CURRENT_BATCH);


		Matrix *m2 = m_gpus.rand(Z.back()->cols, 100);
		Matrix *out2 = empty(m_BA.CURRENT_BATCH->rows,100);
		cout << "DOT TEST NORMAL" << endl;
		m_gpus.dot(Z.back(),m2,out2);
		cout << "POST DOT TEST NORMAL" << endl;


		Matrix *rogue_copy = m_gpus.rand(Z.back()->rows,Z.back()->cols);
		Matrix *m1 = m_gpus.rand(Z.back()->rows, 100);
		Matrix *out = empty(m_BA.CURRENT_BATCH->cols,100);
		cout << "DOT TEST" << endl;
		printf("%ix%i vs %ix%i\n",out->rows,out->cols,Z.back()->cols,m1->cols);
		printf("%ix%i\n",Z.back()->rows,m1->rows);
		m_gpus.Tdot(rogue_copy,m1,out);
		cout << "POST rogue dot TEST" << endl;

		size_t free, total;
		cudaMemGetInfo(&free, &total);
		cout << "free mem post dot test: " << free << endl;

		m_gpus.Tdot(Z.back(),m1,out);
		cout << "POST DOT TEST" << endl;

		cudaMemGetInfo(&free, &total);
		cout << "free mem post dot test: " << free << endl;

		for(int i = 0; i < W.size(); i++)
		{
		  //D.push_back(Z.back());
		  D.push_back(m_gpus.dropout(Z.back(),lDropout[i]));
		  Z.push_back(m_gpus.dot(D.back(), W[i]));
		  addMatrixVector(Z.back(),B[i],Z.back());
		  activation_function(i, Z.back());
		}
/*
		cout << "DOT TEST2" << endl;
		m_gpus.Tdot(Z.back(),m1,out);
		cout << "POST DOT TEST2" << endl;
		*/

	}
	else
	{
		//TODO: Correct input dropout rescaling
		if(ff == Train_error){ Z.push_back(m_BA.CURRENT_BATCH);}
		else{ Z.push_back(m_BA.CURRENT_BATCH_CV);}

		for(int i = 0; i < W.size(); i++)
		{
			Z.push_back(m_gpus.dot(Z.back(), W[i]));
			addMatrixVector(Z.back(),B[i],Z.back());
			activation_function(i, Z.back());
			if(i < W.size() -1)
				scalarMul(Z.back(), 1.0f-lDropout[i+1], Z.back());
		}
	}

	if(OUTPUT_IS_PROBABILITY)
		doubleRectifiedLinear(Z.back(),Z.back());

}

float DeepNeuralNetwork::get_errors(Batchtype_t batch_t)
{
	float errors = 0;
	if(m_net_type == Classification || PRINT_MISSCLASSIFICATION)
	{

		Matrix *result = argmax(Z.back());
		Matrix *eq;
		if(m_net_type == Classification)
		{
			if(batch_t == Train){	eq = equal(result,m_BA.CURRENT_BATCH_Y);}
			else{	eq = equal(result,m_BA.CURRENT_BATCH_CV_Y);}
		}
		else
		{
			Matrix *argmax_regression_batch;
			if(batch_t == Train){argmax_regression_batch = argmax(m_BA.CURRENT_BATCH_Y); eq = equal(result,argmax_regression_batch);}
			else{argmax_regression_batch = argmax(m_BA.CURRENT_BATCH_CV_Y);	eq = equal(result,argmax_regression_batch);}
		}

		float sum_value = sum(eq);
		missclassification_error += (Z.back()->rows  - sum_value);
		cudaFree(result->data);
		cudaFree(eq->data);
	}

	if(m_net_type == Regression)
	{
		//Matrix *sqrErr = squared_error(Z.back(),batch_t == Train ? m_BA.CURRENT_BATCH_Y : m_BA.CURRENT_BATCH_CV_Y);
		Matrix *sqrErr = sub(Z.back(), batch_t == Train ? m_BA.CURRENT_BATCH_Y : m_BA.CURRENT_BATCH_CV_Y);
		square(sqrErr,sqrErr);


		errors = sum(sqrErr);
		errors /=  m_BA.CURRENT_BATCH_Y->cols;
		errors *= batch_t == Train ? m_BA.CURRENT_BATCH_Y->rows : m_BA.CURRENT_BATCH_CV_Y->rows;
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
		scalarMul(B_M[i],MOMENTUM,B_M[i]);
		add(B[i],B_M[i],B[i]);
	}
}

void DeepNeuralNetwork::train_error()
{
	  float errors = 0;
	  missclassification_error = 0.0f;
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



	  if((m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0)) && m_net_type != Classification)
		  std::cout << "Train error: " << errors/m_BA.TRAIN_SET_ROWS << std::endl;
	  if((m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0)) &&
			  PRINT_MISSCLASSIFICATION)
		  std::cout << "Train classification error: " << missclassification_error/m_BA.TRAIN_SET_ROWS << std::endl;
}


void DeepNeuralNetwork::cross_validation_error()
{
	  float errors = 0;
	  missclassification_error = 0.0f;
	  for(int i = 0; i < m_BA.TOTAL_BATCHES_CV; i++)
	  {
		  m_BA.broadcast_batch_cv_to_processes();
		  feedforward(CV_error);
		  m_BA.allocate_next_cv_batch_async();
		  errors += get_errors(CV);
		  free_variables();

		  m_BA.replace_current_cv_batch_with_next();
	  }


	 // cout << "Number of errors: " << missclassification_error << endl;

	  if((m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0)) && m_net_type != Classification)
		  std::cout << "Cross validation error: " << errors/m_BA.CV_SET_ROWS << std::endl;
	  if((m_BA.BATCH_METHOD == Single_GPU || (m_BA.BATCH_METHOD != Single_GPU && m_gpus.MYRANK == 0)) &&
			  PRINT_MISSCLASSIFICATION)
		  std::cout << "Cross validation classification error: " << missclassification_error/m_BA.CV_SET_ROWS << std::endl;
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
			addMatrixVector(Z.back(),B[j],Z.back());
			activation_function(j, Z.back());
			if(j < W.size() -1)
				scalarMul(Z.back(),1.0f-lDropout[i+1],Z.back());
		}

		if(OUTPUT_IS_PROBABILITY)
			doubleRectifiedLinear(Z.back(),Z.back());

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
