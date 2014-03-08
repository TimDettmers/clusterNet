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

using std::cout;
using std::endl;

void BatchAllocator::init(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int batch_size_cv)
{ init(X,y,cross_validation_size,batch_size,batch_size_cv,0,Single_GPU); }
void BatchAllocator::init(std::string path_X, std::string path_y, float cross_validation_size, int batch_size, int cv_batch_size, ClusterNet cluster, BatchAllocationMethod_t batchmethod)
{
	m_cluster = cluster;
	Matrix *X;
	Matrix *y;
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

	init(X,y,cross_validation_size,batch_size,cv_batch_size,m_cluster.MYGPUID, batchmethod);

}

void BatchAllocator::init(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int batch_size_cv, int mygpuid, BatchAllocationMethod_t batchmethod)
{
	m_full_X = X;
	m_full_y = y;

	if(batchmethod == Batch_split)
		MPI_get_dataset_dimensions(X,y);

	if(mygpuid == 0)
	{
		m_Rows = m_full_X->rows;
		m_Cols_X = m_full_X->cols;
		m_Cols_y = m_full_y->cols;

		cudaStreamCreate(&m_streamNext_batch_X);
		cudaStreamCreate(&m_streamNext_batch_y);
		cudaStreamCreate(&m_streamNext_batch_cv_X);
		cudaStreamCreate(&m_streamNext_batch_cv_y);
	}

	BATCH_SIZE = batch_size;
	BATCH_SIZE_CV = batch_size_cv;
	TRAIN_SET_SIZE = ceil(m_Rows * (1.0f-cross_validation_size));
	CV_SET_SIZE = m_Rows - TRAIN_SET_SIZE;
	TOTAL_BATCHES = ceil(TRAIN_SET_SIZE /(BATCH_SIZE*1.0f));
	TOTAL_BATCHES_CV = ceil((m_Rows - TRAIN_SET_SIZE)/(BATCH_SIZE_CV*1.0f));

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

	CURRENT_BATCH = empty(BATCH_SIZE,m_Cols_X);
	m_next_batch_X = empty(BATCH_SIZE,m_Cols_X);
	CURRENT_BATCH_Y = empty(BATCH_SIZE,m_Cols_y);
	m_next_batch_y = empty(BATCH_SIZE,m_Cols_y);

	CURRENT_BATCH_CV = empty(BATCH_SIZE_CV,m_Cols_X);
	m_next_batch_cv_X = empty(BATCH_SIZE_CV,m_Cols_X);
	CURRENT_BATCH_CV_Y = empty(BATCH_SIZE_CV,m_Cols_y);
	m_next_batch_cv_y = empty(BATCH_SIZE_CV,m_Cols_y);


	if(mygpuid == 0)
	{
		cudaMemcpy(&m_next_batch_X->data[0],&m_full_X->data[0],CURRENT_BATCH->bytes,cudaMemcpyDefault);
		cudaMemcpy(&m_next_batch_y->data[0],&m_full_y->data[0],CURRENT_BATCH_Y->bytes,cudaMemcpyDefault);
		cudaMemcpy(&m_next_batch_cv_X->data[0],&m_full_X->data[TRAIN_SET_SIZE*m_Cols_X],CURRENT_BATCH_CV->bytes,cudaMemcpyDefault);
		cudaMemcpy(&m_next_batch_cv_y->data[0],&m_full_y->data[TRAIN_SET_SIZE*m_Cols_y],CURRENT_BATCH_CV_Y->bytes,cudaMemcpyDefault);


		to_col_major(m_next_batch_X, CURRENT_BATCH);
		to_col_major(m_next_batch_y, CURRENT_BATCH_Y);
		to_col_major(m_next_batch_cv_X, CURRENT_BATCH_CV);
		to_col_major(m_next_batch_cv_y, CURRENT_BATCH_CV_Y);

		if(batchmethod != Single_GPU)
		{
			for(int i = 1; i < m_cluster.PCIe_RANKS.size(); i++)
			{
				MPI_Send(CURRENT_BATCH->data,CURRENT_BATCH->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i],999,MPI_COMM_WORLD);
				MPI_Send(CURRENT_BATCH_Y->data,CURRENT_BATCH_Y->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i],998,MPI_COMM_WORLD);
				MPI_Send(CURRENT_BATCH_CV->data,CURRENT_BATCH_CV->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i],997,MPI_COMM_WORLD);
				MPI_Send(CURRENT_BATCH_CV_Y->data,CURRENT_BATCH_CV_Y->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i],996,MPI_COMM_WORLD);
			}
		}


		m_next_batch_number = 1;
		m_next_batch_number_cv = 1;
	}
	else
	{

		MPI_Recv(CURRENT_BATCH->data,CURRENT_BATCH->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],999,MPI_COMM_WORLD,&m_status);
		MPI_Recv(CURRENT_BATCH_Y->data,CURRENT_BATCH_Y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_status);
		MPI_Recv(CURRENT_BATCH_CV->data,CURRENT_BATCH_CV->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],997,MPI_COMM_WORLD,&m_status);
		MPI_Recv(CURRENT_BATCH_CV_Y->data,CURRENT_BATCH_CV_Y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],996,MPI_COMM_WORLD,&m_status);
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

	cout << "cols x: " << m_Cols_X << " cols y" << m_Cols_y <<  " rows " << m_Rows << " my gpu rank " <<  m_cluster.MYGPUID << endl;
}


void BatchAllocator::allocate_next_batch_async()
{
	int copy_range_bytes_X = m_next_batch_X->bytes;
	int copy_range_bytes_y = m_next_batch_y->bytes;


	if((BATCH_SIZE * (m_next_batch_number + 1)) > TRAIN_SET_SIZE)
	{
		//the next batch is smaller than the given standard batch size

		int partial_batch_size = TRAIN_SET_SIZE % BATCH_SIZE;
		copy_range_bytes_X = partial_batch_size*m_full_X->cols*sizeof(float);
		copy_range_bytes_y = partial_batch_size*m_full_y->cols*sizeof(float);
		cudaFree(m_next_batch_X->data);
		cudaFree(m_next_batch_y->data);
		m_next_batch_X = empty(partial_batch_size, m_full_X->cols);
		m_next_batch_y = empty(partial_batch_size, m_full_y->cols);
	}

	cudaMemcpyAsync(&m_next_batch_X->data[0],&m_full_X->data[(m_full_X->cols * m_next_batch_number * BATCH_SIZE)],
					copy_range_bytes_X, cudaMemcpyHostToDevice,m_streamNext_batch_X);
	cudaMemcpyAsync(&m_next_batch_y->data[0],&m_full_y->data[(m_full_y->cols * m_next_batch_number * BATCH_SIZE)],
					copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_y);
}

void BatchAllocator::allocate_next_cv_batch_async()
{
	int copy_range_bytes_X = m_next_batch_cv_X->bytes;
	int copy_range_bytes_y = m_next_batch_cv_y->bytes;

	if((BATCH_SIZE_CV * (m_next_batch_number_cv + 1)) > (m_full_X->rows - TRAIN_SET_SIZE))
	{
		//the next batch is smaller than the given standard batch size
		int partial_batch_size = (m_full_X->rows - TRAIN_SET_SIZE) % BATCH_SIZE_CV;
		copy_range_bytes_X = partial_batch_size*m_full_X->cols*sizeof(float);
		copy_range_bytes_y = partial_batch_size*m_full_y->cols*sizeof(float);
		cudaFree(m_next_batch_cv_X->data);
		cudaFree(m_next_batch_cv_y->data);
		m_next_batch_cv_X = empty(partial_batch_size, m_full_X->cols);
		m_next_batch_cv_y = empty(partial_batch_size, m_full_y->cols);
	}

	cudaMemcpyAsync(&m_next_batch_cv_X->data[0],&m_full_X->data[(TRAIN_SET_SIZE * m_full_X->cols)  + (m_next_batch_number_cv * BATCH_SIZE_CV * m_full_X->cols)],
					copy_range_bytes_X, cudaMemcpyHostToDevice,m_streamNext_batch_cv_X);
	cudaMemcpyAsync(&m_next_batch_cv_y->data[0],&m_full_y->data[(TRAIN_SET_SIZE * m_full_y->cols)  + (m_next_batch_number_cv * BATCH_SIZE_CV * m_full_y->cols)],
					copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_cv_y);
}

void BatchAllocator::replace_current_batch_with_next()
{

	if(m_next_batch_X->rows != CURRENT_BATCH->rows)
	{
		cudaFree(CURRENT_BATCH->data);
		cudaFree(CURRENT_BATCH_Y->data);
		CURRENT_BATCH = empty(m_next_batch_X->rows,m_next_batch_X->cols);
		CURRENT_BATCH_Y = empty(m_next_batch_y->rows,m_next_batch_y->cols);
	}

	cudaStreamSynchronize(m_streamNext_batch_X);
	to_col_major(m_next_batch_X, CURRENT_BATCH);
	cudaStreamSynchronize(m_streamNext_batch_y);
	to_col_major(m_next_batch_y, CURRENT_BATCH_Y);
	m_next_batch_number += 1;

	if(m_next_batch_number == TOTAL_BATCHES)
	{
		//reset to the intial state
		m_next_batch_number = 0;
		if(CURRENT_BATCH->rows != BATCH_SIZE)
		{
			cudaFree(m_next_batch_X->data);
			cudaFree(m_next_batch_y->data);
			m_next_batch_X = empty(BATCH_SIZE,m_full_X->cols);
			m_next_batch_y = empty(BATCH_SIZE,m_full_y->cols);
		}
	}
}

void BatchAllocator::replace_current_cv_batch_with_next()
{

	if(m_next_batch_cv_X->rows != CURRENT_BATCH_CV->rows)
	{
		cudaFree(CURRENT_BATCH_CV->data);
		cudaFree(CURRENT_BATCH_CV_Y->data);
		CURRENT_BATCH_CV = empty(m_next_batch_cv_X->rows,m_next_batch_cv_X->cols);
		CURRENT_BATCH_CV_Y = empty(m_next_batch_cv_y->rows,m_next_batch_cv_y->cols);
	}

	cudaStreamSynchronize(m_streamNext_batch_cv_X);
	to_col_major(m_next_batch_cv_X,CURRENT_BATCH_CV);
	cudaStreamSynchronize(m_streamNext_batch_cv_y);
	to_col_major(m_next_batch_cv_y,CURRENT_BATCH_CV_Y);
	m_next_batch_number_cv += 1;

	if(m_next_batch_number_cv == TOTAL_BATCHES_CV)
	{
		//std::cout << "reset size" << std::endl;
		//reset to the intial state
		m_next_batch_number_cv = 0;
		if(CURRENT_BATCH_CV->rows != BATCH_SIZE_CV)
		{
			cudaFree(m_next_batch_cv_X->data);
			cudaFree(m_next_batch_cv_y->data);
			m_next_batch_cv_X = empty(BATCH_SIZE_CV,m_full_X->cols);
			m_next_batch_cv_y = empty(BATCH_SIZE_CV,m_full_y->cols);
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


