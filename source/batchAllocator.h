#ifndef BatchAllocator_H
#define BatchAllocator_H

#include <cublas_v2.h>
#include <basicOps.cuh>
#include <curand.h>
#include <mpi.h>
#include <list>
#include <string>
#include <clusterNet.h>
#include <cuda_runtime_api.h>


typedef enum BatchAllocationMethod_t
{
	Single_GPU = 0,
	Batch_split = 1
} BatchAllocationMethod_t;

class BatchAllocator
{

public:
	 Matrix *CURRENT_BATCH;
	 Matrix *CURRENT_BATCH_Y;
	 Matrix *CURRENT_BATCH_CV;
	 Matrix *CURRENT_BATCH_CV_Y;
	 int TOTAL_ITERATIONS;
	 int TOTAL_ITERATIONS_CV;
	 int TOTAL_BATCHES;
	 int TOTAL_BATCHES_CV;
	 int BATCH_SIZE;
	 int BATCH_SIZE_CV;
	 int TRAIN_SET_SIZE;
	 int CV_SET_SIZE;

	 void finish_batch_allocator();
	 void broadcast_batch_to_PCI();
	 void broadcast_cv_batch_to_PCI();
	 void allocate_next_batch_async();
	 void allocate_next_cv_batch_async();
	 void replace_current_batch_with_next();
	 void replace_current_cv_batch_with_next();

	 void init(std::string path_X, std::string path_y, float cross_validation_size, int batch_size, int cv_batch_size, ClusterNet cluster, BatchAllocationMethod_t batchmethod);
	 void init(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int cv_batch_size, BatchAllocationMethod_t batchmethod);
	 void init(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int cv_batch_size);

	 int m_next_batch_number_cv;
private:
	 Matrix *m_next_batch_X;
	 Matrix *m_next_batch_y;
	 Matrix *m_next_batch_cv_X;
	 Matrix *m_next_batch_cv_y;
	 Matrix *m_full_X;
	 Matrix *m_full_y;


	 std::vector<Matrix*> m_next_matricies_X;
	 std::vector<Matrix*> m_next_matricies_y;
	 std::vector<Matrix*> m_next_matricies_cv_X;
	 std::vector<Matrix*> m_next_matricies_cv_y;

	 int m_next_batch_number;
	 int m_Cols_X;
	 int m_Cols_y;
	 int m_Rows;
	 int m_mygpuID;
	 int m_myrank;

	 BatchAllocationMethod_t m_batchmethod;
	 ClusterNet m_cluster;
	 MPI_Status m_status;

	 cudaStream_t m_streamNext_batch_X;
	 cudaStream_t m_streamNext_batch_y;
	 cudaStream_t m_streamNext_batch_cv_X;
	 cudaStream_t m_streamNext_batch_cv_y;
	 MPI_Request m_request_send_X;
	 MPI_Request m_request_send_y;
	 MPI_Request m_request_send_cv_X;
	 MPI_Request m_request_send_cv_y;

	 MPI_Request m_request_X;
	 MPI_Request m_request_y;
	 MPI_Request m_request_cv_X;
	 MPI_Request m_request_cv_y;

	 void MPI_get_dataset_dimensions(Matrix *X, Matrix *y);


};
#endif

