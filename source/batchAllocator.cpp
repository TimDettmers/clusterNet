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

using std::cout;
using std::endl;
BatchAllocator::BatchAllocator(){}
BatchAllocator::BatchAllocator(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int batch_size_cv)
{
	float * pinned_memory_X;
	cudaHostAlloc(&pinned_memory_X, X->bytes, cudaHostAllocPortable);
	memcpy(pinned_memory_X,X->data,X->bytes);
	free(X->data);

	m_full_X = (Matrix*)malloc(sizeof(Matrix));
	m_full_X->rows = X->rows;
	m_full_X->cols = X->cols;
	m_full_X->bytes = X->bytes;
	m_full_X->size = X->size;
	m_full_X->data = pinned_memory_X;

	float * pinned_memory_y;
	cudaHostAlloc(&pinned_memory_y, y->bytes, cudaHostAllocPortable);
	memcpy(pinned_memory_y,y->data,y->bytes);
	free(y->data);

	m_full_y = (Matrix*)malloc(sizeof(Matrix));
	m_full_y->rows = y->rows;
	m_full_y->cols = y->cols;
	m_full_y->bytes = y->bytes;
	m_full_y->size = y->size;
	m_full_y->data = pinned_memory_y;

	m_batch_size = batch_size;
	m_batch_size_cv = batch_size_cv;
	m_cv_beginning = ceil(X->rows * (1.0f-cross_validation_size));
	TOTAL_BATCHES = ceil(m_cv_beginning /(m_batch_size*1.0f));
	TOTAL_BATCHES_CV = ceil((X->rows - m_cv_beginning)/(m_batch_size_cv*1.0f));

	if(m_batch_size_cv > (X->rows*cross_validation_size))
	{
		std::cout << "ERROR: Cross validation batch size must be smaller than the cross validation set." << std::endl;
		throw "Cross validation batch size must be smaller than the cross validation set.";
	}

	if(m_batch_size > m_cv_beginning)
	{
		std::cout << "ERROR: Batch size must be smaller than the training set." << std::endl;
		throw "ERROR: Batch size must be smaller than the training set.";
	}

	cudaStreamCreate(&m_streamNext_batch_X);
	cudaStreamCreate(&m_streamNext_batch_y);
	cudaStreamCreate(&m_streamNext_batch_cv_X);
	cudaStreamCreate(&m_streamNext_batch_cv_y);

	m_current_batch_X = empty(m_batch_size,m_full_X->cols);
	m_next_batch_X = empty(m_batch_size,m_full_X->cols);
	m_current_batch_y = empty(m_batch_size,m_full_y->cols);
	m_next_batch_y = empty(m_batch_size,m_full_y->cols);

	m_current_batch_cv_X = empty(m_batch_size_cv,m_full_X->cols);
	m_next_batch_cv_X = empty(m_batch_size_cv,m_full_X->cols);
	m_current_batch_cv_y = empty(m_batch_size_cv,m_full_y->cols);
	m_next_batch_cv_y = empty(m_batch_size_cv,m_full_y->cols);


	cudaMemcpy(&m_current_batch_X->data[0],&m_full_X->data[0],m_current_batch_X->bytes,cudaMemcpyDefault);
	cudaMemcpy(&m_current_batch_y->data[0],&m_full_y->data[0],m_current_batch_y->bytes,cudaMemcpyDefault);
	cudaMemcpy(&m_current_batch_cv_X->data[0],&m_full_X->data[m_cv_beginning*m_full_X->cols],m_current_batch_cv_X->bytes,cudaMemcpyDefault);
	cudaMemcpy(&m_current_batch_cv_y->data[0],&m_full_y->data[m_cv_beginning*m_full_y->cols],m_current_batch_cv_y->bytes,cudaMemcpyDefault);

	Matrix * X_T = to_col_major(m_current_batch_X);
	m_current_batch_X = X_T;
	Matrix * y_T = to_col_major(m_current_batch_y);
	m_current_batch_y = y_T;

	Matrix * X_T_cv = to_col_major(m_current_batch_cv_X);
	m_current_batch_cv_X = X_T_cv;
	Matrix * y_T_cv = to_col_major(m_current_batch_cv_y);
	m_current_batch_cv_y = y_T_cv;

	m_next_batch_number = 1;
	m_next_batch_number_cv = 1;
}


void BatchAllocator::allocate_next_batch_async()
{
	int copy_range_bytes_X = m_next_batch_X->bytes;
	int copy_range_bytes_y = m_next_batch_y->bytes;


	if((m_batch_size * (m_next_batch_number + 1)) > m_cv_beginning)
	{
		//the next batch is smaller than the given standard batch size

		int partial_batch_size = m_cv_beginning % m_batch_size;
		copy_range_bytes_X = partial_batch_size*m_full_X->cols*sizeof(float);
		copy_range_bytes_y = partial_batch_size*m_full_y->cols*sizeof(float);
		cudaFree(m_next_batch_X->data);
		cudaFree(m_next_batch_y->data);
		m_next_batch_X = empty(partial_batch_size, m_full_X->cols);
		m_next_batch_y = empty(partial_batch_size, m_full_y->cols);
	}

	cudaMemcpyAsync(&m_next_batch_X->data[0],&m_full_X->data[(m_full_X->cols * m_next_batch_number * m_batch_size)],
					copy_range_bytes_X, cudaMemcpyHostToDevice,m_streamNext_batch_X);
	cudaMemcpyAsync(&m_next_batch_y->data[0],&m_full_y->data[(m_full_y->cols * m_next_batch_number * m_batch_size)],
					copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_y);
}

void BatchAllocator::allocate_next_cv_batch_async()
{
	int copy_range_bytes_X = m_next_batch_cv_X->bytes;
	int copy_range_bytes_y = m_next_batch_cv_y->bytes;

	if((m_batch_size_cv * (m_next_batch_number_cv + 1)) > (m_full_X->rows - m_cv_beginning))
	{
		//the next batch is smaller than the given standard batch size
		int partial_batch_size = (m_full_X->rows - m_cv_beginning) % m_batch_size_cv;
		copy_range_bytes_X = partial_batch_size*m_full_X->cols*sizeof(float);
		copy_range_bytes_y = partial_batch_size*m_full_y->cols*sizeof(float);
		cudaFree(m_next_batch_cv_X->data);
		cudaFree(m_next_batch_cv_y->data);
		m_next_batch_cv_X = empty(partial_batch_size, m_full_X->cols);
		m_next_batch_cv_y = empty(partial_batch_size, m_full_y->cols);
	}

	cudaMemcpyAsync(&m_next_batch_cv_X->data[0],&m_full_X->data[(m_cv_beginning * m_full_X->cols)  + (m_next_batch_number_cv * m_batch_size_cv * m_full_X->cols)],
					copy_range_bytes_X, cudaMemcpyHostToDevice,m_streamNext_batch_cv_X);
	cudaMemcpyAsync(&m_next_batch_cv_y->data[0],&m_full_y->data[(m_cv_beginning * m_full_y->cols)  + (m_next_batch_number_cv * m_batch_size_cv * m_full_y->cols)],
					copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_cv_y);
}

void BatchAllocator::replace_current_batch_with_next()
{

	if(m_next_batch_X->rows != m_current_batch_X->rows)
	{
		cudaFree(m_current_batch_X->data);
		cudaFree(m_current_batch_y->data);
		m_current_batch_X = empty(m_next_batch_X->rows,m_next_batch_X->cols);
		m_current_batch_y = empty(m_next_batch_y->rows,m_next_batch_y->cols);
	}

	cudaStreamSynchronize(m_streamNext_batch_X);
	to_col_major(m_next_batch_X, m_current_batch_X);
	cudaStreamSynchronize(m_streamNext_batch_y);
	to_col_major(m_next_batch_y, m_current_batch_y);
	m_next_batch_number += 1;

	if(m_next_batch_number == TOTAL_BATCHES)
	{
		//reset to the intial state
		m_next_batch_number = 0;
		if(m_current_batch_X->rows != m_batch_size)
		{
			cudaFree(m_next_batch_X->data);
			cudaFree(m_next_batch_y->data);
			m_next_batch_X = empty(m_batch_size,m_full_X->cols);
			m_next_batch_y = empty(m_batch_size,m_full_y->cols);
		}
	}
}

void BatchAllocator::replace_current_cv_batch_with_next()
{

	if(m_next_batch_cv_X->rows != m_current_batch_cv_X->rows)
	{
		cudaFree(m_current_batch_cv_X->data);
		cudaFree(m_current_batch_cv_y->data);
		m_current_batch_cv_X = empty(m_next_batch_cv_X->rows,m_next_batch_cv_X->cols);
		m_current_batch_cv_y = empty(m_next_batch_cv_y->rows,m_next_batch_cv_y->cols);
	}

	cudaStreamSynchronize(m_streamNext_batch_cv_X);
	to_col_major(m_next_batch_cv_X,m_current_batch_cv_X);
	cudaStreamSynchronize(m_streamNext_batch_cv_y);
	to_col_major(m_next_batch_cv_y,m_current_batch_cv_y);
	m_next_batch_number_cv += 1;

	if(m_next_batch_number_cv == TOTAL_BATCHES_CV)
	{
		//std::cout << "reset size" << std::endl;
		//reset to the intial state
		m_next_batch_number_cv = 0;
		if(m_current_batch_cv_X->rows != m_batch_size_cv)
		{
			cudaFree(m_next_batch_cv_X->data);
			cudaFree(m_next_batch_cv_y->data);
			m_next_batch_cv_X = empty(m_batch_size_cv,m_full_X->cols);
			m_next_batch_cv_y = empty(m_batch_size_cv,m_full_y->cols);
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


