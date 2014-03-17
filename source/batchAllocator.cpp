#include <cublas_v2.h>
#include <batchAllocator.h>
#include <basicOps.cuh>
#include <util.cuh>
#include <cstdlib>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cuda_device_runtime_api.h>

using std::cout;
using std::endl;

void BatchAllocator::init(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int batch_size_cv)
{ m_mygpuID = 0; init(X,y,cross_validation_size,batch_size,batch_size_cv,Single_GPU); }
void BatchAllocator::init(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int cv_batch_size, ClusterNet cluster, BatchAllocationMethod_t batchmethod)
{
	m_cluster = cluster;

	m_mygpuID = m_cluster.MYGPUID;
	m_myrank = m_cluster.MYRANK;
	if(m_cluster.MYGPUID != 0)
	{
		cudaFree(X->data);
		cudaFree(y->data);
		X = zeros(1,1);
		y = zeros(1,1);
	}

	//for(int i = 0; i < m_cluster.PCIe_RANKS.size(); i++)
	//	cout << "myrank: " << m_myrank << " pcie rank: " << m_cluster.PCIe_RANKS[i] << endl;

	//cout << "myrank: " << m_myrank << endl;;
	init(X,y,cross_validation_size,batch_size,cv_batch_size, batchmethod);

}
void BatchAllocator::init(std::string path_X, std::string path_y, float cross_validation_size, int batch_size, int cv_batch_size, ClusterNet cluster, BatchAllocationMethod_t batchmethod)
{
	m_cluster = cluster;
	Matrix *X;
	Matrix *y;
	m_mygpuID = m_cluster.MYGPUID;
	m_myrank = m_cluster.MYRANK;
	if(m_cluster.MYGPUID == 0)
	{
		if(path_X.find("cvs") != std::string::npos)
		{
			X = read_csv(path_X.c_str());
			y = read_csv(path_y.c_str());
		}
		else if(path_X.find("hdf5") != std::string::npos)
		{
			X = read_hdf5(path_X.c_str());
			y = read_hdf5(path_y.c_str());
		}
		else
		{
			cout << "Only the cvs and hdf5 formats are supported!" << endl;
			throw "Only the cvs and hdf5 formats are supported!";
		}

	}
	else
	{
		X = zeros(1,1);
		y = zeros(1,1);
	}


	init(X,y,cross_validation_size,batch_size,cv_batch_size, batchmethod);

}

void BatchAllocator::init(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int batch_size_cv, BatchAllocationMethod_t batchmethod)
{
	m_full_X = X;
	m_full_y = y;
	BATCH_METHOD = batchmethod;

	if(BATCH_METHOD != Single_GPU)
		MPI_get_dataset_dimensions(X,y);
	else
	{
		m_Rows = X->rows;
		m_Cols_X = X->cols;
		m_Cols_y = y->cols;
	}


	BATCH_SIZE = batch_size;
	if(batchmethod == Batch_split)
	{
		//cout << "Batch size will be set to 64 for each GPU to run in the batch split mode." << endl;
		BATCH_SIZE = 64;
	}
	BATCH_SIZE_CV = batch_size_cv;
	TRAIN_SET_SIZE = ceil(m_Rows * (1.0f-cross_validation_size));
	CV_SET_SIZE = m_Rows - TRAIN_SET_SIZE;
	TOTAL_BATCHES = ceil(TRAIN_SET_SIZE /(BATCH_SIZE*1.0f));
	TOTAL_BATCHES_CV = ceil((m_Rows - TRAIN_SET_SIZE)/(BATCH_SIZE_CV*1.0f));
	TOTAL_ITERATIONS = BATCH_METHOD == Batch_split ? TOTAL_BATCHES/m_cluster.MPI_SIZE : TOTAL_BATCHES;
	TOTAL_ITERATIONS_CV = BATCH_METHOD == Batch_split ? TOTAL_BATCHES_CV/m_cluster.MPI_SIZE : TOTAL_BATCHES_CV;

	if(BATCH_SIZE_CV > (m_Rows*cross_validation_size))
	{
		std::cout << "ERROR: Cross validation batch size must be smaller than the cross validation set." << std::endl;
		throw "Cross validation batch size must be smaller than the cross validation set.";
	}

	if(BATCH_SIZE > TRAIN_SET_SIZE)
	{
		std::cout << "ERROR: Batch size must be smaller than the training set." << std::endl;
		throw "ERROR: Batch size must be smaller than the training set.";
	}


	if(m_mygpuID == 0)
	{
		m_Rows = m_full_X->rows;
		m_Cols_X = m_full_X->cols;
		m_Cols_y = m_full_y->cols;

		cudaStreamCreate(&m_streamNext_batch_X);
		cudaStreamCreate(&m_streamNext_batch_y);
		cudaStreamCreate(&m_streamNext_batch_cv_X);
		cudaStreamCreate(&m_streamNext_batch_cv_y);
	}

	CURRENT_BATCH = empty(BATCH_SIZE,m_Cols_X);
	m_next_matrices_X.push_back(empty(BATCH_SIZE,m_Cols_X));
	CURRENT_BATCH_Y = empty(BATCH_SIZE,m_Cols_y);
	m_next_matrices_y.push_back(empty(BATCH_SIZE,m_Cols_y));

	CURRENT_BATCH_CV = empty(BATCH_SIZE_CV,m_Cols_X);
	m_next_matrices_cv_X.push_back(empty(BATCH_SIZE_CV,m_Cols_X));
	CURRENT_BATCH_CV_Y = empty(BATCH_SIZE_CV,m_Cols_y);
	m_next_matrices_cv_y.push_back(empty(BATCH_SIZE_CV,m_Cols_y));


	if(m_mygpuID == 0)
	{
		int pci_count = 1;
		m_myrank = 0;
		if(batchmethod == Batch_split){ pci_count = m_cluster.PCIe_RANKS.size(); m_myrank = m_cluster.MYRANK; }
		if(batchmethod == Distributed_weights){ m_myrank = m_cluster.MYRANK; }

		for(int i = 1; i < pci_count; i++)
		{
			m_next_matrices_X.push_back(empty(BATCH_SIZE,m_Cols_X));
			m_next_matrices_y.push_back(empty(BATCH_SIZE,m_Cols_y));
			m_next_matrices_cv_X.push_back(empty(BATCH_SIZE_CV,m_Cols_X));
			m_next_matrices_cv_y.push_back(empty(BATCH_SIZE_CV,m_Cols_y));
		}

		if(batchmethod == Batch_split)
		{
			//there needs to be a offset for each master gpu
			//this offset is simply the rank of the
			m_next_batch_number = m_myrank;
			m_next_batch_number_cv = m_myrank;
		}
		else
		{
			m_next_batch_number = 0;
			m_next_batch_number_cv = 0;
		}

		for(int i = 0; i < pci_count; i++)
		{
			cudaMemcpy(&m_next_matrices_X[i]->data[0],&m_full_X->data[(m_full_X->cols * (m_next_batch_number+i) * BATCH_SIZE)],
					m_next_matrices_X[i]->bytes, cudaMemcpyHostToDevice);
			cudaMemcpy(&m_next_matrices_y[i]->data[0],&m_full_y->data[(m_full_y->cols * (m_next_batch_number+i) * BATCH_SIZE)],
					m_next_matrices_y[i]->bytes, cudaMemcpyHostToDevice);
			cudaMemcpy(&m_next_matrices_cv_X[i]->data[0],&m_full_X->data[(TRAIN_SET_SIZE * m_full_X->cols)  + ((m_next_batch_number_cv+i) * BATCH_SIZE_CV * m_full_X->cols)],
					m_next_matrices_cv_X[i]->bytes, cudaMemcpyHostToDevice);
			cudaMemcpy(&m_next_matrices_cv_y[i]->data[0],&m_full_y->data[(TRAIN_SET_SIZE * m_full_y->cols)  + ((m_next_batch_number_cv+i) * BATCH_SIZE_CV * m_full_y->cols)],
					m_next_matrices_cv_y[i]->bytes, cudaMemcpyHostToDevice);

			//overwrite and send all batches away
			//but keep the last one

			to_col_major(m_next_matrices_X[i], CURRENT_BATCH);
			to_col_major(m_next_matrices_y[i], CURRENT_BATCH_Y);
			to_col_major(m_next_matrices_cv_X[i], CURRENT_BATCH_CV);
			to_col_major(m_next_matrices_cv_y[i], CURRENT_BATCH_CV_Y);

			if(batchmethod == Batch_split && i < pci_count -1)
			{
				MPI_Send(CURRENT_BATCH->data,CURRENT_BATCH->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],999,MPI_COMM_WORLD);
				MPI_Send(CURRENT_BATCH_Y->data,CURRENT_BATCH_Y->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],998,MPI_COMM_WORLD);
				MPI_Send(CURRENT_BATCH_CV->data,CURRENT_BATCH_CV->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],997,MPI_COMM_WORLD);
				MPI_Send(CURRENT_BATCH_CV_Y->data,CURRENT_BATCH_CV_Y->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],996,MPI_COMM_WORLD);
			}
		}

		if(batchmethod == Distributed_weights)
		{
			for(int i = 0; i < m_cluster.PCIe_RANKS.size()-1; i++)
			{
				MPI_Send(CURRENT_BATCH->data,CURRENT_BATCH->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],999,MPI_COMM_WORLD);
				MPI_Send(CURRENT_BATCH_Y->data,CURRENT_BATCH_Y->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],998,MPI_COMM_WORLD);
				MPI_Send(CURRENT_BATCH_CV->data,CURRENT_BATCH_CV->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],997,MPI_COMM_WORLD);
				MPI_Send(CURRENT_BATCH_CV_Y->data,CURRENT_BATCH_CV_Y->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],996,MPI_COMM_WORLD);
			}
		}
	}
	else
	{
		m_myrank = m_cluster.MYRANK;
		MPI_Recv(CURRENT_BATCH->data,CURRENT_BATCH->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],999,MPI_COMM_WORLD,&m_status);
		MPI_Recv(CURRENT_BATCH_Y->data,CURRENT_BATCH_Y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_status);
		MPI_Recv(CURRENT_BATCH_CV->data,CURRENT_BATCH_CV->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],997,MPI_COMM_WORLD,&m_status);
		MPI_Recv(CURRENT_BATCH_CV_Y->data,CURRENT_BATCH_CV_Y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],996,MPI_COMM_WORLD,&m_status);
	}

	if(batchmethod == Batch_split )
	{
		m_next_batch_number += m_cluster.MPI_SIZE;
		m_next_batch_number_cv += m_cluster.MPI_SIZE;
	}
	else
	{
		m_next_batch_number += 1;
		m_next_batch_number_cv += 1;
	}
}

void BatchAllocator::MPI_get_dataset_dimensions(Matrix *X, Matrix *y)
{
	if(m_cluster.MYGPUID == 0)
	{
		m_Cols_X = X->cols;
		m_Cols_y = y->cols;
		m_Rows = X->rows;

		for(int i = 1; i < m_cluster.PCIe_RANKS.size(); i++)
		{
			MPI_Send(&m_Cols_X,1, MPI_INT,m_cluster.PCIe_RANKS[i],999,MPI_COMM_WORLD);
			MPI_Send(&m_Cols_y,1, MPI_INT,m_cluster.PCIe_RANKS[i],998,MPI_COMM_WORLD);
			MPI_Send(&m_Rows,1, MPI_INT,m_cluster.PCIe_RANKS[i],997,MPI_COMM_WORLD);
		}
	}
	else
	{
		MPI_Recv(&m_Cols_X,1,MPI_INT,m_cluster.PCIe_RANKS[0],999,MPI_COMM_WORLD,&m_status);
		MPI_Recv(&m_Cols_y,1,MPI_INT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_status);
		MPI_Recv(&m_Rows,1,MPI_INT,m_cluster.PCIe_RANKS[0],997,MPI_COMM_WORLD,&m_status);
	}
}

void BatchAllocator::broadcast_batch_to_PCI()
{
	if(m_mygpuID == 0)
	{
		cudaStreamSynchronize(m_streamNext_batch_X);
		cudaStreamSynchronize(m_streamNext_batch_y);
		for(int i = 1; i < m_cluster.PCIe_RANKS.size(); i++)
		{
			MPI_Isend(m_next_matrices_X[0]->data,m_next_matrices_X[0]->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],999,MPI_COMM_WORLD,&m_request_send_X);
			MPI_Isend(m_next_matrices_y[0]->data,m_next_matrices_y[0]->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],998,MPI_COMM_WORLD,&m_request_send_y);
		}
	}
	else
	{
		MPI_Recv(m_next_matrices_X[0]->data,m_next_matrices_X[0]->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],999,MPI_COMM_WORLD,&m_status);
		MPI_Recv(m_next_matrices_y[0]->data,m_next_matrices_y[0]->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_status);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void BatchAllocator::broadcast_cv_batch_to_PCI()
{
	if(m_mygpuID == 0)
	{
		cudaStreamSynchronize(m_streamNext_batch_cv_X);
		cudaStreamSynchronize(m_streamNext_batch_cv_y);
		for(int i = 1; i < m_cluster.PCIe_RANKS.size(); i++)
		{
			MPI_Isend(m_next_matrices_cv_X[0]->data,m_next_matrices_cv_X[0]->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],999,MPI_COMM_WORLD,&m_request_send_cv_X);
			MPI_Isend(m_next_matrices_cv_y[0]->data,m_next_matrices_cv_y[0]->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],998,MPI_COMM_WORLD,&m_request_send_cv_y);
		}
	}
	else
	{
		MPI_Recv(m_next_matrices_cv_X[0]->data,m_next_matrices_cv_X[0]->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],999,MPI_COMM_WORLD,&m_status);
		MPI_Recv(m_next_matrices_cv_y[0]->data,m_next_matrices_cv_y[0]->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_status);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void BatchAllocator::allocate_next_batch_async()
{
	int pci_count = 1;
	if(BATCH_METHOD == Batch_split){ pci_count = m_cluster.PCIe_RANKS.size();  }
	for(int i = 0; i < pci_count; i++)
	{
		int copy_range_bytes_X = m_next_matrices_X[i]->bytes;
		int copy_range_bytes_y = m_next_matrices_y[i]->bytes;
		if((BATCH_SIZE * (m_next_batch_number + 1 + i)) > TRAIN_SET_SIZE)
		{
			//the next batch is smaller than the given standard batch size
			int partial_batch_size = TRAIN_SET_SIZE % BATCH_SIZE;
			copy_range_bytes_X = partial_batch_size*m_Cols_X*sizeof(float);
			copy_range_bytes_y = partial_batch_size*m_Cols_y*sizeof(float);
			cudaFree(m_next_matrices_X[i]->data);
			cudaFree(m_next_matrices_y[i]->data);
			m_next_matrices_X[i] = empty(partial_batch_size, m_Cols_X);
			m_next_matrices_y[i] = empty(partial_batch_size, m_Cols_y);
		}

		if(!(BATCH_METHOD == Distributed_weights && m_mygpuID != 0))
		{
			cudaMemcpyAsync(m_next_matrices_X[i]->data,&m_full_X->data[(m_full_X->cols * (m_next_batch_number+i) * BATCH_SIZE)],
							copy_range_bytes_X, cudaMemcpyHostToDevice,m_streamNext_batch_X);
			cudaMemcpyAsync(m_next_matrices_y[i]->data,&m_full_y->data[(m_full_y->cols * (m_next_batch_number+i) * BATCH_SIZE)],
							copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_y);
		}
	}
}

void BatchAllocator::allocate_next_cv_batch_async()
{
	int pci_count = 1;
	if(BATCH_METHOD == Batch_split){ pci_count = m_cluster.PCIe_RANKS.size();  }
	for(int i = 0; i < pci_count; i++)
	{
		int copy_range_bytes_X = m_next_matrices_cv_X[i]->bytes;
		int copy_range_bytes_y = m_next_matrices_cv_y[i]->bytes;

		if((BATCH_SIZE_CV * (m_next_batch_number_cv + 1 + i)) > (m_Rows - TRAIN_SET_SIZE))
		{
			//the next batch is smaller than the given standard batch size
			int partial_batch_size = (m_Rows - TRAIN_SET_SIZE) % BATCH_SIZE_CV;
			copy_range_bytes_X = partial_batch_size*m_Cols_X*sizeof(float);
			copy_range_bytes_y = partial_batch_size*m_Cols_y*sizeof(float);
			cudaFree(m_next_matrices_cv_X[i]->data);
			cudaFree(m_next_matrices_cv_y[i]->data);
			m_next_matrices_cv_X[i] = empty(partial_batch_size, m_Cols_X);
			m_next_matrices_cv_y[i] = empty(partial_batch_size, m_Cols_y);
		}

		if(!(BATCH_METHOD == Distributed_weights && m_mygpuID != 0))
		{
			cudaMemcpyAsync(m_next_matrices_cv_X[i]->data,&m_full_X->data[(TRAIN_SET_SIZE * m_full_X->cols)  + ((m_next_batch_number_cv + i) * BATCH_SIZE_CV * m_full_X->cols)],
							copy_range_bytes_X, cudaMemcpyHostToDevice,m_streamNext_batch_cv_X);
			cudaMemcpyAsync(m_next_matrices_cv_y[i]->data,&m_full_y->data[(TRAIN_SET_SIZE * m_full_y->cols)  + ((m_next_batch_number_cv + i) * BATCH_SIZE_CV * m_full_y->cols)],
							copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_cv_y);
		}
	}
}

void BatchAllocator::replace_current_batch_with_next()
{
	//cout << "enter batcho" << endl;
	/*
	if(m_mygpuID != 0)
	{
		cout << "wait it!" << endl;
		cout << "myrank: " << m_myrank << endl;
		MPI_Wait(&m_request_X,&m_status);
		MPI_Wait(&m_request_y,&m_status);
	}
	*/

	//cout << "post wait it" << endl;
	if(m_next_matrices_X[0]->rows != CURRENT_BATCH->rows)
	{
		cudaFree(CURRENT_BATCH->data);
		cudaFree(CURRENT_BATCH_Y->data);
		CURRENT_BATCH = empty(m_next_matrices_X[0]->rows,m_next_matrices_X[0]->cols);
		CURRENT_BATCH_Y = empty(m_next_matrices_y[0]->rows,m_next_matrices_y[0]->cols);
	}

	//cout << "post batch quier" << endl;

	if(BATCH_METHOD == Single_GPU)
		cudaStreamSynchronize(m_streamNext_batch_X);
	to_col_major(m_next_matrices_X[0], CURRENT_BATCH);
	if(BATCH_METHOD == Single_GPU)
		cudaStreamSynchronize(m_streamNext_batch_y);
	to_col_major(m_next_matrices_y[0], CURRENT_BATCH_Y);


	//cout << "post tranpose" << endl;

	if(BATCH_METHOD == Batch_split )
		m_next_batch_number += m_cluster.MPI_SIZE;
	else
		m_next_batch_number += 1;

	if(m_next_batch_number >= TOTAL_BATCHES)
	{
		//reset to the intial state
		if(BATCH_METHOD == Batch_split)
			m_next_batch_number = m_myrank;
		else
			m_next_batch_number = 0;
		if(CURRENT_BATCH->rows != BATCH_SIZE)
		{
			cudaFree(m_next_matrices_X[0]->data);
			cudaFree(m_next_matrices_y[0]->data);
			m_next_matrices_X[0] = empty(BATCH_SIZE,m_Cols_X);
			m_next_matrices_y[0] = empty(BATCH_SIZE,m_Cols_y);
		}
	}
}

void BatchAllocator::replace_current_cv_batch_with_next()
{
	/*
	if(m_mygpuID != 0)
	{
		MPI_Wait(&m_request_cv_X,&m_status);
		MPI_Wait(&m_request_cv_y,&m_status);
	}
	*/

	if(m_next_matrices_cv_X[0]->rows != CURRENT_BATCH_CV->rows)
	{
		cudaFree(CURRENT_BATCH_CV->data);
		cudaFree(CURRENT_BATCH_CV_Y->data);
		CURRENT_BATCH_CV = empty(m_next_matrices_cv_X[0]->rows,m_next_matrices_cv_X[0]->cols);
		CURRENT_BATCH_CV_Y = empty(m_next_matrices_cv_y[0]->rows,m_next_matrices_cv_y[0]->cols);
	}

	if(BATCH_METHOD == Single_GPU)
		cudaStreamSynchronize(m_streamNext_batch_cv_X);
	to_col_major(m_next_matrices_cv_X[0],CURRENT_BATCH_CV);
	if(BATCH_METHOD == Single_GPU)
		cudaStreamSynchronize(m_streamNext_batch_cv_y);
	to_col_major(m_next_matrices_cv_y[0],CURRENT_BATCH_CV_Y);

	if(BATCH_METHOD == Batch_split )
		m_next_batch_number_cv += m_cluster.MPI_SIZE;
	else
		m_next_batch_number_cv += 1;

	if(m_next_batch_number_cv >= TOTAL_BATCHES_CV)
	{
		//reset to the intial state

		if(BATCH_METHOD == Batch_split)
			m_next_batch_number_cv = m_myrank;
		else
			m_next_batch_number_cv = 0;
		if(CURRENT_BATCH_CV->rows != BATCH_SIZE_CV)
		{
			cudaFree(m_next_matrices_cv_X[0]->data);
			cudaFree(m_next_matrices_cv_y[0]->data);
			m_next_matrices_cv_X[0] = empty(BATCH_SIZE_CV,m_Cols_X);
			m_next_matrices_cv_y[0] = empty(BATCH_SIZE_CV,m_Cols_y);
		}
	}
}

void BatchAllocator::finish_batch_allocator()
{
	cudaStreamDestroy(m_streamNext_batch_X);
	cudaStreamDestroy(m_streamNext_batch_y);
	cudaStreamDestroy(m_streamNext_batch_cv_X);
	cudaStreamDestroy(m_streamNext_batch_cv_y);
}


void BatchAllocator::average_weight(Matrix *W)
{
	//first compute the average of PCIe matrices
	//then pass them to the master gpu neighbors to compute the network average
	//then pass the overall results back to the PCIe GPUs

	//calc PCIe average

	if(m_cluster.MYRANK == 0)
		m_cluster.tick("PCIe");
	Matrix *recv_buffer = empty(W->rows,W->cols);
	int destination = m_myrank + 1;
	int source = m_myrank - 1;
	if(destination > m_cluster.PCIe_RANKS.back()){ destination =  m_cluster.PCIe_RANKS[0]; }
	if(source < m_cluster.PCIe_RANKS[0]){ source = m_cluster.PCIe_RANKS.back(); }
	if(m_cluster.PCIe_RANKS.size() > 1)
		for(int i = 0; i < m_cluster.PCIe_RANKS.size()-1;i++)
		{
			if(i == 0)
				MPI_Isend(W->data,W->size,MPI_FLOAT,destination,997,MPI_COMM_WORLD,&m_request_send_W);
			else
				MPI_Isend(recv_buffer->data,recv_buffer->size,MPI_FLOAT,destination,997,MPI_COMM_WORLD,&m_request_send_W);
			MPI_Recv(recv_buffer->data,recv_buffer->size,MPI_FLOAT,source,997,MPI_COMM_WORLD,&m_status);

			//MPI_Wait(&m_request_W,&m_status);
			add(W,recv_buffer,W);

		}


	if(m_cluster.MYRANK == 0)
			m_cluster.tick("PCIe");
	//cout << "post PCIe average" << endl;

	if(m_cluster.MYRANK == 0)
			m_cluster.tick("MPI");
	//calc network average
	if(m_mygpuID == 0)
	{
		//determine source and destination
		for(int i = 0; i < m_cluster.MASTER_GPU_RANKS.size(); i++)
		{
			if(m_cluster.MASTER_GPU_RANKS[i] == m_myrank)
			{
				if((i + 1) < m_cluster.MASTER_GPU_RANKS.size()){ destination = m_cluster.MASTER_GPU_RANKS[i+1];}
				else{ destination = m_cluster.MASTER_GPU_RANKS[0]; }

				if((i - 1) < 0){ source = m_cluster.MASTER_GPU_RANKS.back(); }
				else{ source = m_cluster.MASTER_GPU_RANKS[i-1]; }
			}
		}

		for(int i = 0; i < m_cluster.MASTER_GPU_RANKS.size()-1; i++)
		{
			if(i == 0)
				MPI_Isend(W->data,W->size,MPI_FLOAT,destination,999,MPI_COMM_WORLD,&m_request_send_W);
			else
				MPI_Isend(recv_buffer->data,recv_buffer->size,MPI_FLOAT,destination,999,MPI_COMM_WORLD, &m_request_send_W);

			MPI_Irecv(recv_buffer->data,recv_buffer->size,MPI_FLOAT,source,999,MPI_COMM_WORLD,&m_request_W);
			MPI_Wait(&m_request_W,&m_status);
			add(W,recv_buffer,W);
		}

		//average and spread weight to all PCIe GPUs
		scalarMul(W,1.0/(float)m_cluster.MPI_SIZE,W);
		for(int i = 1; i < m_cluster.PCIe_RANKS.size(); i++)
		{
			MPI_Isend(W->data,W->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],998,MPI_COMM_WORLD,&m_request_send_W);

		}
	}
	else
	{
		MPI_Irecv(W->data,W->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_request_W);
		MPI_Wait(&m_request_W,&m_status);
	}

	if(m_cluster.MYRANK == 0)
			m_cluster.tick("MPI");


	/*
	cout << "W rows: " << W->rows << endl;
	cout << "W cols: " << W->cols << endl;
	cout << "myrank: " << m_myrank << endl;
	*/
	//printmat(W);

	cudaFree(recv_buffer->data);


}

