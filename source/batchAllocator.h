#ifndef BatchAllocator_H
#define BatchAllocator_H

#include <cublas_v2.h>
#include <basicOps.cuh>
#include <curand.h>
#include <mpi.h>
#include <list>
#include <string>

class BatchAllocator
{

public:
	 Matrix *CURRENT_BATCH;
	 Matrix *CURRENT_BATCH_Y;
	 Matrix *CURRENT_BATCH_CV;
	 Matrix *CURRENT_BATCH_CV_Y;
	 int TOTAL_BATCHES;
	 int TOTAL_BATCHES_CV;
	 int BATCH_SIZE;
	 int BATCH_SIZE_CV;
	 int TRAIN_SET_SIZE;
	 int CV_SET_SIZE;

	 void finish_batch_allocator();
	 void allocate_next_batch_async();
	 void allocate_next_cv_batch_async();
	 void replace_current_batch_with_next();
	 void replace_current_cv_batch_with_next();

	 typedef enum Batchtype_t
	 {
		 Single_GPU = 0,
		 Batch_split = 1
	 } Batchtype_t;

	 void init(std::string path_X, std::string path_y, float cross_validation_size, int batch_size, int cv_batch_size, int myrank, int mygpuID, int nodes, Batchtype_t);
	 void init(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int cv_batch_size, int myrank, int mygpuID, int nodes, Batchtype_t);
	 void init(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int cv_batch_size);
private:
	 Matrix *m_next_batch_X;
	 Matrix *m_next_batch_y;
	 Matrix *m_next_batch_cv_X;
	 Matrix *m_next_batch_cv_y;
	 Matrix *m_full_X;
	 Matrix *m_full_y;

	 int m_next_batch_number;
	 int m_next_batch_number_cv;

	 cudaStream_t m_streamNext_batch_X;
	 cudaStream_t m_streamNext_batch_y;
	 cudaStream_t m_streamNext_batch_cv_X;
	 cudaStream_t m_streamNext_batch_cv_y;

	 void MPI_get_dataset_dimensions(Matrix *X, Matrix *y);
};
#endif

