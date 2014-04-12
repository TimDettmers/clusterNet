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
{
	m_myrank = 0;
	m_mygpuID = 0;
	BATCH_METHOD = Single_GPU;
	m_full_X = X;
	m_full_y = y;

	init(cross_validation_size,batch_size,batch_size_cv);
}
void BatchAllocator::init(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int cv_batch_size, ClusterNet cluster, BatchAllocationMethod_t batchmethod)
{
	m_cluster = cluster;

	m_mygpuID = m_cluster.MYGPUID;
	m_myrank = m_cluster.MYRANK;
	if(m_cluster.MYGPUID != 0)
	{
		cudaFree(X->data);
		cudaFree(y->data);
		if(X->isSparse == 0)
		{
			X = zeros(1,1);
			y = zeros(1,1);
		}
		else
		{
			X = empty_sparse(1,1,1);
			y = empty_sparse(1,1,1);
		}
	}

	m_full_X = X;
	m_full_y = y;
	BATCH_METHOD = batchmethod;
	init(cross_validation_size,batch_size,cv_batch_size);

}
void BatchAllocator::init(std::string path_X, std::string path_y, float cross_validation_size, int batch_size, int cv_batch_size, ClusterNet cluster, BatchAllocationMethod_t batchmethod)
{
	m_cluster = cluster;
	Matrix *X;
	Matrix *y;
	m_mygpuID = m_cluster.MYGPUID;
	m_myrank = m_cluster.MYRANK;
	if(m_cluster.MYGPUID == 0 || batchmethod == Single_GPU)
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


	m_full_X = X;
	m_full_y = y;
	BATCH_METHOD = batchmethod;
	init(cross_validation_size,batch_size,cv_batch_size);
}

void BatchAllocator::init(float cross_validation_size, int batch_size, int batch_size_cv)
{
	SKIP_LAST_BATCH = false;

	if(BATCH_METHOD != Single_GPU)
		MPI_get_dataset_dimensions();
	else
	{
		m_Rows = m_full_X->rows;
		m_Cols_X = m_full_X->cols;
		m_Cols_y = m_full_y->cols;
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

	init_batch_buffer();

	int request_count = BATCH_METHOD == Distributed_weights_sparse ? 3 : 1;
	for(int i = 0; i < request_count; i++)
	{
		MPI_Request request_X;
		MPI_Request request_y;
		MPI_Request request_cv_X;
		MPI_Request request_cv_y;
		m_request_X.push_back(request_X);
		m_request_y.push_back(request_y);
		m_request_cv_X.push_back(request_cv_X);
		m_request_cv_y.push_back(request_cv_y);
	}

	if(m_mygpuID == 0)
	{
		if(BATCH_METHOD == Distributed_weights ||
		   BATCH_METHOD == Distributed_weights_sparse)
		{ m_myrank = m_cluster.MYRANK; }

		for(int i = 0; i < (m_cluster.PCIe_RANKS.size()-1)*request_count;i++)
		{
			MPI_Request send_X;
			MPI_Request send_y;
			MPI_Request send_cv_X;
			MPI_Request send_cv_y;
			m_requests_send_X.push_back(send_X);
			m_requests_send_y.push_back(send_y);
			m_requests_send_cv_X.push_back(send_cv_X);
			m_requests_send_cv_y.push_back(send_cv_y);
		}
		m_next_batch_number = 0;
		m_next_batch_number_cv = 0;

		init_copy_to_buffer();

		if(BATCH_METHOD == Distributed_weights)
		{
			for(int i = 0; i < m_cluster.PCIe_RANKS.size()-1; i++)
			{
				MPI_Send(m_next_buffer_X->data,m_next_buffer_X->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],999,MPI_COMM_WORLD);
				MPI_Send(m_next_buffer_y->data,m_next_buffer_y->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],998,MPI_COMM_WORLD);
				MPI_Send(m_next_buffer_cv_X->data,m_next_buffer_cv_X->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],997,MPI_COMM_WORLD);
				MPI_Send(m_next_buffer_cv_y->data,m_next_buffer_cv_y->size, MPI_FLOAT,m_cluster.PCIe_RANKS[i+1],996,MPI_COMM_WORLD);
			}
		}

	}
	else
	{

		if(BATCH_METHOD != Distributed_weights_sparse)
		{
			m_myrank = m_cluster.MYRANK;
			MPI_Recv(m_next_buffer_X->data,m_next_buffer_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],999,MPI_COMM_WORLD,&m_status);
			MPI_Recv(m_next_buffer_y->data,m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_status);
			MPI_Recv(m_next_buffer_cv_X->data,m_next_buffer_cv_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],997,MPI_COMM_WORLD,&m_status);
			MPI_Recv(m_next_buffer_cv_y->data,m_next_buffer_cv_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],996,MPI_COMM_WORLD,&m_status);
		}


	}

	if(BATCH_METHOD != Distributed_weights_sparse)
	{
		cudaMemcpy(m_next_batch_X->data,m_next_buffer_X->data, m_next_batch_X->bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(m_next_batch_y->data,m_next_buffer_y->data, m_next_batch_y->bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(m_next_batch_cv_X->data,m_next_buffer_cv_X->data, m_next_batch_cv_X->bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(m_next_batch_cv_y->data,m_next_buffer_cv_y->data, m_next_batch_cv_y->bytes, cudaMemcpyHostToDevice);

		to_col_major(m_next_batch_X, CURRENT_BATCH);
		to_col_major(m_next_batch_y, CURRENT_BATCH_Y);
		to_col_major(m_next_batch_cv_X, CURRENT_BATCH_CV);
		to_col_major(m_next_batch_cv_y, CURRENT_BATCH_CV_Y);

		m_next_batch_number += 1;
		m_next_batch_number_cv += 1;
	}
	else
	{
		broadcast_batch_to_processes();
		allocate_next_batch_async();
		replace_current_batch_with_next();
	}
}

void BatchAllocator::init_batch_buffer()
{



	if(BATCH_METHOD != Distributed_weights_sparse)
	{

		CURRENT_BATCH = empty(BATCH_SIZE,m_Cols_X);
		CURRENT_BATCH_Y = empty(BATCH_SIZE,m_Cols_y);
		CURRENT_BATCH_CV = empty(BATCH_SIZE_CV,m_Cols_X);
		CURRENT_BATCH_CV_Y = empty(BATCH_SIZE_CV,m_Cols_y);

		m_next_buffer_X = empty_pinned(BATCH_SIZE,m_Cols_X);
		m_next_buffer_y = empty_pinned(BATCH_SIZE,m_Cols_y);
		m_next_buffer_cv_X = empty_pinned(BATCH_SIZE_CV,m_Cols_X);
		m_next_buffer_cv_y = empty_pinned(BATCH_SIZE_CV,m_Cols_y);

		m_next_batch_X = empty(BATCH_SIZE,m_Cols_X);
		m_next_batch_y = empty(BATCH_SIZE,m_Cols_y);
		m_next_batch_cv_X = empty(BATCH_SIZE_CV,m_Cols_X);
		m_next_batch_cv_y = empty(BATCH_SIZE_CV,m_Cols_y);
	}
	else
	{
		float sparsity_X = 1.0;
		float sparsity_y = 1.0;

		if(m_mygpuID == 0)
		{
			sparsity_X = determine_max_sparsity(m_full_X,BATCH_SIZE);
			if(m_full_y->isSparse == 1)
				sparsity_y = determine_max_sparsity(m_full_y,BATCH_SIZE);

			for(int i = 1; i < m_cluster.PCIe_RANKS.size() && BATCH_METHOD != Single_GPU; i++)
			{
				MPI_Send(&sparsity_X,5,MPI_INT,m_cluster.PCIe_RANKS[i],10,MPI_COMM_WORLD);
				MPI_Send(&sparsity_y,5,MPI_INT,m_cluster.PCIe_RANKS[i],11,MPI_COMM_WORLD);
			}
		}
		else
		{
			MPI_Recv(&sparsity_X,5,MPI_INT,0,10,MPI_COMM_WORLD,&m_status);
			MPI_Recv(&sparsity_y,5,MPI_INT,0,11,MPI_COMM_WORLD, &m_status);
		}

		m_next_buffer_X = empty_pinned_sparse(BATCH_SIZE,m_Cols_X,sparsity_X,0.00f);
		m_next_buffer_cv_X = empty_pinned_sparse(BATCH_SIZE_CV,m_Cols_X,sparsity_X,0.00f);

		m_next_batch_X = empty_sparse(BATCH_SIZE,m_Cols_X,sparsity_X,0.00f);
		m_next_batch_cv_X = empty_sparse(BATCH_SIZE_CV,m_Cols_X,sparsity_X,0.00f);

		//CURRENT_BATCH = empty_sparse(BATCH_SIZE,m_Cols_X,sparsity_X,0.0f);
		//CURRENT_BATCH_CV = empty_sparse(BATCH_SIZE_CV,m_Cols_X,sparsity_X,0.0f);

		CURRENT_BATCH = m_next_batch_X;
		CURRENT_BATCH_CV = m_next_batch_cv_X;

		if(m_full_y->isSparse == 1)
		{

			m_next_buffer_y = empty_pinned_sparse(BATCH_SIZE,m_Cols_y, sparsity_y, 0.00f);
			m_next_buffer_cv_y = empty_pinned_sparse(BATCH_SIZE_CV,m_Cols_y, sparsity_y, 0.00f);

			m_next_batch_y = empty_sparse(BATCH_SIZE,m_Cols_y, sparsity_y, 0.00f);
			m_next_batch_cv_y = empty_sparse(BATCH_SIZE_CV,m_Cols_y, sparsity_y, 0.00f);

			//CURRENT_BATCH_Y = empty_sparse(BATCH_SIZE,m_Cols_y, sparsity_y, 0.0f);
			//CURRENT_BATCH_CV_Y = empty_sparse(BATCH_SIZE_CV,m_Cols_y, sparsity_y, 0.0f);

			CURRENT_BATCH_Y = m_next_batch_y;
			CURRENT_BATCH_CV_Y = m_next_batch_cv_y;
		}
		else
		{
			CURRENT_BATCH_Y = empty(BATCH_SIZE,m_Cols_y);
			CURRENT_BATCH_CV_Y = empty(BATCH_SIZE_CV,m_Cols_y);

			m_next_buffer_y = empty_pinned(BATCH_SIZE,m_Cols_y);
			m_next_buffer_cv_y = empty_pinned(BATCH_SIZE_CV,m_Cols_y);

			m_next_batch_y = empty(BATCH_SIZE,m_Cols_y);
			m_next_batch_cv_y = empty(BATCH_SIZE_CV,m_Cols_y);
		}

	}
}

void BatchAllocator::init_copy_to_buffer()
{
	if(BATCH_METHOD != Distributed_weights_sparse)
	{
		if(m_full_X->isSparse != 1)
		{

			memcpy(&m_next_buffer_X->data[0],&m_full_X->data[0], m_next_buffer_X->bytes);
			memcpy(&m_next_buffer_cv_X->data[0],&m_full_X->data[(TRAIN_SET_SIZE * m_full_X->cols)],	m_next_buffer_cv_X->bytes);
		}
		else
		{
			slice_sparse_to_dense(m_full_X,m_next_buffer_X,0,BATCH_SIZE);
			slice_sparse_to_dense(m_full_X,m_next_buffer_cv_X,TRAIN_SET_SIZE,BATCH_SIZE_CV);
		}

		if(m_full_y->isSparse != 1)
		{
			memcpy(&m_next_buffer_y->data[0],&m_full_y->data[0], m_next_buffer_y->bytes);
			memcpy(&m_next_buffer_cv_y->data[0],&m_full_y->data[(TRAIN_SET_SIZE * m_full_y->cols)],	m_next_buffer_cv_y->bytes);
		}
		else
		{
			slice_sparse_to_dense(m_full_y,m_next_buffer_y,0,BATCH_SIZE);
			slice_sparse_to_dense(m_full_y,m_next_buffer_cv_y,TRAIN_SET_SIZE,BATCH_SIZE_CV);
		}
	}
	else
	{

	}
}

void BatchAllocator::MPI_get_dataset_dimensions()
{
	if(m_cluster.MYGPUID == 0)
	{
		m_Cols_X = m_full_X->cols;
		m_Cols_y = m_full_y->cols;
		m_Rows = m_full_X->rows;

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




void BatchAllocator::broadcast_batch_to_processes()
{
	int partial_batch_size = BATCH_SIZE;
	int copy_range_bytes_X = m_next_buffer_X->bytes;
	int copy_range_bytes_y = m_next_buffer_y->bytes;
	if((BATCH_SIZE * (m_next_batch_number + 1)) > TRAIN_SET_SIZE)
	{
		//the next batch is smaller than the given standard batch size
		partial_batch_size = TRAIN_SET_SIZE % BATCH_SIZE;
		copy_range_bytes_X = partial_batch_size*m_Cols_X*sizeof(float);
		copy_range_bytes_y = partial_batch_size*m_Cols_y*sizeof(float);
	}

	if(BATCH_METHOD != Distributed_weights_sparse)
	{
		if(m_mygpuID == 0)
		{
			if(m_full_X->isSparse != 1)
				memcpy(m_next_buffer_X->data,&m_full_X->data[(m_full_X->cols * (m_next_batch_number) * BATCH_SIZE)], copy_range_bytes_X);
			else
				slice_sparse_to_dense(m_full_X,m_next_buffer_X,m_next_batch_number*BATCH_SIZE, partial_batch_size);

			if(m_full_y->isSparse != 1)
				memcpy(m_next_buffer_y->data,&m_full_y->data[(m_full_y->cols * (m_next_batch_number) * BATCH_SIZE)], copy_range_bytes_y);
			else
				slice_sparse_to_dense(m_full_y,m_next_buffer_y,m_next_batch_number*BATCH_SIZE, partial_batch_size);

			for(int i = 1; i < m_cluster.PCIe_RANKS.size() && BATCH_METHOD != Single_GPU; i++)
			{
				MPI_Isend(m_next_buffer_X->data,m_next_buffer_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],999,MPI_COMM_WORLD,&m_requests_send_X[i-1]);
				MPI_Isend(m_next_buffer_y->data,m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],998,MPI_COMM_WORLD,&m_requests_send_y[i-1]);
			}

		}
		else
		{
			MPI_Irecv(m_next_buffer_X->data,m_next_buffer_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],999,MPI_COMM_WORLD,&m_request_X[0]);
			MPI_Irecv(m_next_buffer_y->data,m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_request_y[0]);
		}
	}
	else
	{
		if(m_mygpuID == 0)
		{
			int next_batch_start_index = m_next_batch_number*BATCH_SIZE;
			int idx_from = m_full_X->ptr_rows[next_batch_start_index];
			int idx_to = m_full_X->ptr_rows[next_batch_start_index + partial_batch_size];
			int range = (idx_to - idx_from) -1;

			assert(m_next_buffer_X->size >= range);
			memcpy(m_next_buffer_X->data,&m_full_X->data[idx_from],sizeof(float)*range);
			memcpy(m_next_buffer_X->idx_cols,&m_full_X->idx_cols[idx_from],sizeof(int)*range);
			memcpy(m_next_buffer_X->ptr_rows,&m_full_X->ptr_rows[idx_from],sizeof(int)*(partial_batch_size +1));

			m_sparse_matrix_info_X[0] = range;
			m_sparse_matrix_info_X[1] = sizeof(float)*range;
			m_sparse_matrix_info_X[2] = sizeof(int)*range;
			m_sparse_matrix_info_X[3] = partial_batch_size;
			m_sparse_matrix_info_X[4] = (partial_batch_size+1)*sizeof(int);

			if(m_full_y->isSparse != 1)
				memcpy(m_next_buffer_y->data,&m_full_y->data[(m_full_y->cols * (m_next_batch_number) * BATCH_SIZE)], copy_range_bytes_y);
			else
			{
				idx_from = m_full_y->ptr_rows[next_batch_start_index];
				idx_to = m_full_y->ptr_rows[next_batch_start_index + partial_batch_size];
				range = (idx_to - idx_from);
				memcpy(&m_next_buffer_y->data[0],&m_full_y->data[idx_from],sizeof(float)*range);
				memcpy(&m_next_buffer_y->idx_cols[0],&m_full_y->idx_cols[idx_from],sizeof(int)*range);
				memcpy(&m_next_buffer_y->ptr_rows[0],&m_full_y->ptr_rows[idx_from],sizeof(int)*(BATCH_SIZE +1));

				m_sparse_matrix_info_y[0] = range;
				m_sparse_matrix_info_y[1] = sizeof(float)*range;
				m_sparse_matrix_info_y[2] = sizeof(int)*range;
				m_sparse_matrix_info_y[3] = partial_batch_size;
				m_sparse_matrix_info_y[4] = (partial_batch_size+1)*sizeof(int);
			}

			//update_next_buffer_matrix_info();

			for(int i = 1; i < m_cluster.PCIe_RANKS.size() && BATCH_METHOD != Single_GPU; i++)
			{
				int k = 3*(i-1);
				MPI_Isend(&m_next_buffer_X->data[0],m_next_buffer_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],999,MPI_COMM_WORLD,&m_requests_send_X[k + 0]);
				MPI_Isend(&m_next_buffer_X->idx_cols[0],m_next_buffer_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],9991,MPI_COMM_WORLD,&m_requests_send_X[k + 1]);
				MPI_Isend(&m_next_buffer_X->ptr_rows[0],m_next_buffer_X->rows+1,MPI_FLOAT,m_cluster.PCIe_RANKS[i],9992,MPI_COMM_WORLD,&m_requests_send_X[k + 2]);

				if(m_full_y->isSparse != 1)
					MPI_Isend(m_next_buffer_y->data,m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],998,MPI_COMM_WORLD,&m_requests_send_y[k]);
				else
				{
					MPI_Isend(&m_next_buffer_y->data[0],m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],998,MPI_COMM_WORLD,&m_requests_send_y[k + 0]);
					MPI_Isend(&m_next_buffer_y->idx_cols[0],m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],9981,MPI_COMM_WORLD,&m_requests_send_y[k + 1]);
					MPI_Isend(&m_next_buffer_y->ptr_rows[0],m_next_buffer_y->rows+1,MPI_FLOAT,m_cluster.PCIe_RANKS[i],9982,MPI_COMM_WORLD,&m_requests_send_y[k + 2]);
				}
			}

		}
		else
		{
			if(BATCH_METHOD != Distributed_weights_sparse)
			{
				MPI_Irecv(m_next_buffer_X->data,m_next_buffer_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],999,MPI_COMM_WORLD,&m_request_X[0]);
				MPI_Irecv(m_next_buffer_y->data,m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_request_y[0]);
			}
			else
			{
				MPI_Irecv(&m_next_buffer_X->data[0],m_next_buffer_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],999,MPI_COMM_WORLD,&m_request_X[0]);
				MPI_Irecv(&m_next_buffer_X->idx_cols[0],m_next_buffer_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],9991,MPI_COMM_WORLD,&m_request_X[1]);
				MPI_Irecv(&m_next_buffer_X->ptr_rows[0],m_next_buffer_X->rows+1,MPI_FLOAT,m_cluster.PCIe_RANKS[0],9992,MPI_COMM_WORLD,&m_request_X[2]);

				if(m_full_y->isSparse != 1)
					MPI_Irecv(m_next_buffer_y->data,m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_request_y[0]);
				else
				{
					MPI_Irecv(&m_next_buffer_y->data[0],m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_request_y[0]);
					MPI_Irecv(&m_next_buffer_y->idx_cols[0],m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],9981,MPI_COMM_WORLD,&m_request_y[1]);
					MPI_Irecv(&m_next_buffer_y->ptr_rows[0],m_next_buffer_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],9982,MPI_COMM_WORLD,&m_request_y[2]);
				}
			}
		}

	}
}


void BatchAllocator::update_next_batch_matrix_info()
{

	if(m_mygpuID == 0)
		for(int i = 1; i < m_cluster.PCIe_RANKS.size() && BATCH_METHOD != Single_GPU; i++)
		{
			MPI_Send(m_sparse_matrix_info_X,5,MPI_INT,m_cluster.PCIe_RANKS[i],10,MPI_COMM_WORLD);
			MPI_Send(m_sparse_matrix_info_y,5,MPI_INT,m_cluster.PCIe_RANKS[i],11,MPI_COMM_WORLD);
		}
	else
	{
		MPI_Recv(m_sparse_matrix_info_X,5,MPI_INT,0,10,MPI_COMM_WORLD,&m_status);
		MPI_Recv(m_sparse_matrix_info_y,5,MPI_INT,0,11,MPI_COMM_WORLD, &m_status);
	}

	m_next_batch_y->size = m_sparse_matrix_info_y[0];
	m_next_batch_y->bytes = m_sparse_matrix_info_y[1];
	m_next_batch_y->idx_bytes = m_sparse_matrix_info_y[2];
	m_next_batch_y->rows = m_sparse_matrix_info_y[3];
	m_next_batch_y->ptr_bytes = m_sparse_matrix_info_y[4];


	m_next_batch_X->size = m_sparse_matrix_info_X[0];
	m_next_batch_X->bytes = m_sparse_matrix_info_X[1];
	m_next_batch_X->idx_bytes = m_sparse_matrix_info_X[2];
	m_next_batch_X->rows = m_sparse_matrix_info_X[3];
	m_next_batch_X->ptr_bytes = m_sparse_matrix_info_X[4];
}

void BatchAllocator::update_next_buffer_matrix_info()
{
	m_next_buffer_y->size = m_sparse_matrix_info_y[0];
	m_next_buffer_y->bytes = m_sparse_matrix_info_y[1];
	m_next_buffer_y->idx_bytes = m_sparse_matrix_info_y[2];
	m_next_buffer_y->rows = m_sparse_matrix_info_y[3];
	m_next_buffer_y->ptr_bytes = m_sparse_matrix_info_y[4];


	m_next_buffer_X->size = m_sparse_matrix_info_X[0];
	m_next_buffer_X->bytes = m_sparse_matrix_info_X[1];
	m_next_buffer_X->idx_bytes = m_sparse_matrix_info_X[2];
	m_next_buffer_X->rows = m_sparse_matrix_info_X[3];
	m_next_buffer_X->ptr_bytes = m_sparse_matrix_info_X[4];
}



void BatchAllocator::broadcast_batch_cv_to_processes()
{

	int copy_range_bytes_X = m_next_buffer_cv_X->bytes;
	int copy_range_bytes_y = m_next_buffer_cv_y->bytes;
	int partial_batch_size = BATCH_SIZE_CV;
	if((BATCH_SIZE_CV * (m_next_batch_number_cv + 1)) > CV_SET_SIZE)
	{
		//the next batch is smaller than the given standard batch size
		partial_batch_size = CV_SET_SIZE % BATCH_SIZE_CV;
		copy_range_bytes_X = partial_batch_size*m_Cols_X*sizeof(float);
		copy_range_bytes_y = partial_batch_size*m_Cols_y*sizeof(float);
	}

	if(m_mygpuID == 0)
	{
		if(m_full_X->isSparse != 1)
			memcpy(m_next_buffer_cv_X->data,&m_full_X->data[(TRAIN_SET_SIZE * m_full_X->cols)  + ((m_next_batch_number_cv) * BATCH_SIZE_CV * m_full_X->cols)],
					copy_range_bytes_X);
		else
			slice_sparse_to_dense(m_full_X,m_next_buffer_cv_X,TRAIN_SET_SIZE + (m_next_batch_number_cv * BATCH_SIZE_CV), partial_batch_size);

		if(m_full_y->isSparse != 1)
			memcpy(m_next_buffer_cv_y->data,&m_full_y->data[(TRAIN_SET_SIZE * m_full_y->cols)  + ((m_next_batch_number_cv) * BATCH_SIZE_CV * m_full_y->cols)],
					copy_range_bytes_y);
		else
			slice_sparse_to_dense(m_full_y,m_next_buffer_cv_y,TRAIN_SET_SIZE + (m_next_batch_number_cv * BATCH_SIZE_CV), partial_batch_size);

		for(int i = 1; i < m_cluster.PCIe_RANKS.size() && BATCH_METHOD != Single_GPU; i++)
		{
			MPI_Isend(m_next_buffer_cv_X->data,m_next_buffer_cv_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],999,MPI_COMM_WORLD,&m_requests_send_cv_X[i-1]);
			MPI_Isend(m_next_buffer_cv_y->data,m_next_buffer_cv_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[i],998,MPI_COMM_WORLD,&m_requests_send_cv_y[i-1]);
		}
	}
	else
	{
		MPI_Irecv(m_next_buffer_cv_X->data,m_next_buffer_cv_X->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],999,MPI_COMM_WORLD,&m_request_cv_X[0]);
		MPI_Irecv(m_next_buffer_cv_y->data,m_next_buffer_cv_y->size,MPI_FLOAT,m_cluster.PCIe_RANKS[0],998,MPI_COMM_WORLD,&m_request_cv_y[0]);
	}
}



void BatchAllocator::allocate_next_batch_async()
{
	int copy_range_bytes_X = m_next_batch_X->bytes;
	int copy_range_bytes_y = m_next_batch_y->bytes;
	int partial_batch_size = BATCH_SIZE;
	if((BATCH_SIZE * (m_next_batch_number + 1)) > TRAIN_SET_SIZE)
	{

		if(BATCH_METHOD != Distributed_weights_sparse)
		{
			//the next batch is smaller than the given standard batch size
			partial_batch_size = TRAIN_SET_SIZE % BATCH_SIZE;
			copy_range_bytes_X = partial_batch_size*m_Cols_X*sizeof(float);
			copy_range_bytes_y = partial_batch_size*m_Cols_y*sizeof(float);
			cudaFree(m_next_batch_X->data);
			cudaFree(m_next_batch_y->data);
			m_next_batch_X = empty(partial_batch_size, m_Cols_X);
			m_next_batch_y = empty(partial_batch_size, m_Cols_y);
		}
		else
		{
			partial_batch_size = TRAIN_SET_SIZE % BATCH_SIZE;
		}
	}



	if(BATCH_METHOD == Distributed_weights_sparse)
		update_next_batch_matrix_info();



	if(m_mygpuID != 0)
	{
		if(BATCH_METHOD != Distributed_weights_sparse)
		{
			MPI_Wait(&m_request_X[0],&m_status);
			MPI_Wait(&m_request_y[0],&m_status);
		}
		else
		{
			MPI_Wait(&m_request_X[0],&m_status);
			MPI_Wait(&m_request_X[1],&m_status);
			MPI_Wait(&m_request_X[2],&m_status);

			if(m_full_y->isSparse != 1)
				MPI_Wait(&m_request_y[0],&m_status);
			else
			{
				MPI_Wait(&m_request_y[0],&m_status);
				MPI_Wait(&m_request_y[1],&m_status);
				MPI_Wait(&m_request_y[2],&m_status);
			}
		}
	}
	else
	{
		for(int i = 0; i < m_cluster.PCIe_RANKS.size()-1  && BATCH_METHOD != Single_GPU;i++)
		{

			if(BATCH_METHOD != Distributed_weights_sparse)
			{
				MPI_Wait(&m_requests_send_X[i],&m_status);
				MPI_Wait(&m_requests_send_y[i],&m_status);
			}
			else
			{
				int k = 3*i;
				MPI_Wait(&m_requests_send_X[k + 0],&m_status);
				MPI_Wait(&m_requests_send_X[k + 1],&m_status);
				MPI_Wait(&m_requests_send_X[k + 2],&m_status);
				if(m_full_y->isSparse != 1)
					MPI_Wait(&m_requests_send_y[k],&m_status);
				else
				{
					MPI_Wait(&m_requests_send_y[k + 0],&m_status);
					MPI_Wait(&m_requests_send_y[k + 1],&m_status);
					MPI_Wait(&m_requests_send_y[k + 2],&m_status);
				}
			}
		}
	}

	if(BATCH_METHOD != Distributed_weights_sparse)
	{
		cudaMemcpyAsync(m_next_batch_X->data,m_next_buffer_X->data,	copy_range_bytes_X, cudaMemcpyHostToDevice,m_streamNext_batch_X);
		cudaMemcpyAsync(m_next_batch_y->data,m_next_buffer_y->data,	copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_y);
	}
	else
	{

		cudaMemcpyAsync(&m_next_batch_X->data[0],&m_next_buffer_X->data[0],m_next_buffer_X->bytes, cudaMemcpyHostToDevice,m_streamNext_batch_X);
		cudaMemcpyAsync(&m_next_batch_X->idx_cols[0],&m_next_buffer_X->idx_cols[0],m_next_buffer_X->idx_bytes, cudaMemcpyHostToDevice,m_streamNext_batch_X);
		cudaMemcpyAsync(&m_next_batch_X->ptr_rows[0],&m_next_buffer_X->ptr_rows[0],m_next_buffer_X->ptr_bytes, cudaMemcpyHostToDevice,m_streamNext_batch_X);

		if(m_full_y->isSparse != 1)
			cudaMemcpyAsync(m_next_batch_y->data,m_next_buffer_y->data,	copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_y);
		else
		{
			cudaMemcpyAsync(&m_next_batch_y->data[0],&m_next_buffer_y->data[0],m_next_buffer_y->bytes, cudaMemcpyHostToDevice,m_streamNext_batch_y);
			cudaMemcpyAsync(&m_next_batch_y->idx_cols[0],&m_next_buffer_y->idx_cols[0],m_next_buffer_y->idx_bytes, cudaMemcpyHostToDevice,m_streamNext_batch_y);
			cudaMemcpyAsync(&m_next_batch_y->ptr_rows[0],&m_next_buffer_y->ptr_rows[0],m_next_buffer_y->ptr_bytes, cudaMemcpyHostToDevice,m_streamNext_batch_y);
		}
	}
}

void BatchAllocator::allocate_next_cv_batch_async()
{
	int copy_range_bytes_X = m_next_batch_cv_X->bytes;
	int copy_range_bytes_y = m_next_batch_cv_y->bytes;

	if((BATCH_SIZE_CV * (m_next_batch_number_cv + 1)) > CV_SET_SIZE)
	{
		//the next batch is smaller than the given standard batch size
		int partial_batch_size = CV_SET_SIZE % BATCH_SIZE_CV;
		copy_range_bytes_X = partial_batch_size*m_Cols_X*sizeof(float);
		copy_range_bytes_y = partial_batch_size*m_Cols_y*sizeof(float);
		cudaFree(m_next_batch_cv_X->data);
		cudaFree(m_next_batch_cv_y->data);
		m_next_batch_cv_X = empty(partial_batch_size, m_Cols_X);
		m_next_batch_cv_y = empty(partial_batch_size, m_Cols_y);
	}

	if(m_mygpuID != 0)
	{
		MPI_Wait(&m_request_cv_X[0],&m_status);
		MPI_Wait(&m_request_cv_y[0],&m_status);
	}
	else
	{
		for(int i = 0; i < m_cluster.PCIe_RANKS.size()-1  && BATCH_METHOD != Single_GPU;i++)
		{
			MPI_Wait(&m_requests_send_cv_X[i],&m_status);
			MPI_Wait(&m_requests_send_cv_y[i],&m_status);
		}
	}


	cudaMemcpyAsync(m_next_batch_cv_X->data,m_next_buffer_cv_X->data, copy_range_bytes_X, cudaMemcpyHostToDevice,m_streamNext_batch_cv_X);
	cudaMemcpyAsync(m_next_batch_cv_y->data,m_next_buffer_cv_y->data, copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_cv_y);

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
	cudaStreamSynchronize(m_streamNext_batch_y);

	if(BATCH_METHOD != Distributed_weights_sparse)
	{
		to_col_major(m_next_batch_X, CURRENT_BATCH);
		to_col_major(m_next_batch_y, CURRENT_BATCH_Y);
	}
	else
	{
		update_next_batch_matrix_info();
	}

	m_next_batch_number += 1;

	if(((m_next_batch_number + 1) == TOTAL_BATCHES) && SKIP_LAST_BATCH)
		m_next_batch_number = 0;

	if(m_next_batch_number >= TOTAL_BATCHES)
	{
		//reset to the intial state
		m_next_batch_number = 0;
		if(CURRENT_BATCH->rows != BATCH_SIZE && BATCH_METHOD != Distributed_weights_sparse)
		{
			cudaFree(m_next_batch_X->data);
			cudaFree(m_next_batch_y->data);
			m_next_batch_X = empty(BATCH_SIZE,m_Cols_X);
			m_next_batch_y = empty(BATCH_SIZE,m_Cols_y);
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

	if(((m_next_batch_number_cv + 1) == TOTAL_BATCHES_CV) && SKIP_LAST_BATCH)
		m_next_batch_number_cv = 0;

	if(m_next_batch_number_cv >= TOTAL_BATCHES_CV)
	{
		//reset to the intial state
		m_next_batch_number_cv = 0;
		if(CURRENT_BATCH_CV->rows != BATCH_SIZE_CV)
		{
			cudaFree(m_next_batch_cv_X->data);
			cudaFree(m_next_batch_cv_y->data);
			m_next_batch_cv_X = empty(BATCH_SIZE_CV,m_Cols_X);
			m_next_batch_cv_y = empty(BATCH_SIZE_CV,m_Cols_y);
		}
	}
}




void BatchAllocator::finish_batch_allocator()
{
	cudaDeviceSynchronize();

	/*
	cudaFree(m_next_buffer_X->data);
	cudaFree(m_next_buffer_y->data);
	cudaFree(m_next_buffer_cv_X->data);
	cudaFree(m_next_buffer_cv_y->data);


	cudaFree(m_next_batch_X->data);
	cudaFree(m_next_batch_y->data);
	cudaFree(m_next_batch_cv_X->data);
	cudaFree(m_next_batch_cv_y->data);

	cudaFree(CURRENT_BATCH->data);
	cudaFree(CURRENT_BATCH_CV->data);
	cudaFree(CURRENT_BATCH_Y->data);
	cudaFree(CURRENT_BATCH_CV_Y->data);
	*/

	cudaStreamDestroy(m_streamNext_batch_X);
	cudaStreamDestroy(m_streamNext_batch_y);
	cudaStreamDestroy(m_streamNext_batch_cv_X);
	cudaStreamDestroy(m_streamNext_batch_cv_y);
}
