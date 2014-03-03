#ifndef BatchAllocator_H
#define BatchAllocator_H

#include <cublas_v2.h>
#include <basicOps.cuh>
#include <curand.h>
#include <mpi.h>
#include <list>

class BatchAllocator
{

public:
	 BatchAllocator();
	 BatchAllocator(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int cv_batch_size);
	 Matrix *m_current_batch_X;
	 Matrix *m_current_batch_y;
	 Matrix *m_current_batch_cv_X;
	 Matrix *m_current_batch_cv_y;
	 int TOTAL_BATCHES;
	 int TOTAL_BATCHES_CV;

	 void finish_batch_allocator();
	 void allocate_next_batch_async();
	 void allocate_next_cv_batch_async();
	 void replace_current_batch_with_next();
	 void replace_current_cv_batch_with_next();
private:
	 Matrix *m_next_batch_X;
	 Matrix *m_next_batch_y;
	 Matrix *m_next_batch_cv_X;
	 Matrix *m_next_batch_cv_y;
	 Matrix *m_full_X;
	 Matrix *m_full_y;

	 int m_batch_size;
	 int m_next_batch_number;
	 int m_cv_beginning;
	 int m_batch_size_cv;
	 int m_next_batch_number_cv;

	 cudaStream_t m_streamNext_batch_X;
	 cudaStream_t m_streamNext_batch_y;
	 cudaStream_t m_streamNext_batch_cv_X;
	 cudaStream_t m_streamNext_batch_cv_y;
};
#endif

