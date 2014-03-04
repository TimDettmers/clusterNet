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

	BATCH_SIZE = batch_size;
	BATCH_SIZE_CV = batch_size_cv;
	TRAIN_SET_SIZE = ceil(X->rows * (1.0f-cross_validation_size));
	CV_SET_SIZE = X->rows - TRAIN_SET_SIZE;
	TOTAL_BATCHES = ceil(TRAIN_SET_SIZE /(BATCH_SIZE*1.0f));
	TOTAL_BATCHES_CV = ceil((X->rows - TRAIN_SET_SIZE)/(BATCH_SIZE_CV*1.0f));

	if(BATCH_SIZE_CV > (X->rows*cross_validation_size))
	{
		std::cout << "ERROR: Cross validation batch size must be smaller than the cross validation set." << std::endl;
		throw "Cross validation batch size must be smaller than the cross validation set.";
	}

	if(BATCH_SIZE > TRAIN_SET_SIZE)
	{
		std::cout << "ERROR: Batch size must be smaller than the training set." << std::endl;
		throw "ERROR: Batch size must be smaller than the training set.";
	}

	cudaStreamCreate(&m_streamNext_batch_X);
	cudaStreamCreate(&m_streamNext_batch_y);
	cudaStreamCreate(&m_streamNext_batch_cv_X);
	cudaStreamCreate(&m_streamNext_batch_cv_y);

	CURRENT_BATCH = empty(BATCH_SIZE,m_full_X->cols);
	m_next_batch_X = empty(BATCH_SIZE,m_full_X->cols);
	CURRENT_BATCH_Y = empty(BATCH_SIZE,m_full_y->cols);
	m_next_batch_y = empty(BATCH_SIZE,m_full_y->cols);

	CURRENT_BATCH_CV = empty(BATCH_SIZE_CV,m_full_X->cols);
	m_next_batch_cv_X = empty(BATCH_SIZE_CV,m_full_X->cols);
	CURRENT_BATCH_CV_Y = empty(BATCH_SIZE_CV,m_full_y->cols);
	m_next_batch_cv_y = empty(BATCH_SIZE_CV,m_full_y->cols);


	cudaMemcpy(&CURRENT_BATCH->data[0],&m_full_X->data[0],CURRENT_BATCH->bytes,cudaMemcpyDefault);
	cudaMemcpy(&CURRENT_BATCH_Y->data[0],&m_full_y->data[0],CURRENT_BATCH_Y->bytes,cudaMemcpyDefault);
	cudaMemcpy(&CURRENT_BATCH_CV->data[0],&m_full_X->data[TRAIN_SET_SIZE*m_full_X->cols],CURRENT_BATCH_CV->bytes,cudaMemcpyDefault);
	cudaMemcpy(&CURRENT_BATCH_CV_Y->data[0],&m_full_y->data[TRAIN_SET_SIZE*m_full_y->cols],CURRENT_BATCH_CV_Y->bytes,cudaMemcpyDefault);

	Matrix * X_T = to_col_major(CURRENT_BATCH);
	CURRENT_BATCH = X_T;
	Matrix * y_T = to_col_major(CURRENT_BATCH_Y);
	CURRENT_BATCH_Y = y_T;

	Matrix * X_T_cv = to_col_major(CURRENT_BATCH_CV);
	CURRENT_BATCH_CV = X_T_cv;
	Matrix * y_T_cv = to_col_major(CURRENT_BATCH_CV_Y);
	CURRENT_BATCH_CV_Y = y_T_cv;

	m_next_batch_number = 1;
	m_next_batch_number_cv = 1;
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


