#ifndef ClusterNet_H
#define ClusterNet_H

#include <cublas_v2.h>
#include <basicOps.cuh>
#include <curand.h>
#include <mpi.h>
#include <list>

class ClusterNet
{

public:
	 ClusterNet();
	 ClusterNet(int seed);
	 ClusterNet(int argc, char *argv[], int seed);

	 int m_rank;
	 Matrix *m_current_batch_X;
	 Matrix *m_current_batch_y;
	 Matrix *m_current_batch_cv_X;
	 Matrix *m_current_batch_cv_y;
	 int TOTAL_BATCHES;
	 int TOTAL_BATCHES_CV;

	 Matrix *dot(Matrix *A, Matrix *B);
	 Matrix *Tdot(Matrix *A, Matrix *B);
	 Matrix *dotT(Matrix *A, Matrix *B);
	 void dot(Matrix *A, Matrix *B, Matrix *out);
	 void Tdot(Matrix *A, Matrix *B, Matrix *out);
	 void dotT(Matrix *A, Matrix *B, Matrix *out);
	 Matrix *dotMPI_unitSlice(Matrix *A, Matrix *B);
	 Matrix *dotMPI_batchSlice(Matrix *A, Matrix *B);

	 Matrix *rand(int rows, int cols);
	 void rand(int rows, int cols, Matrix *out);
	 Matrix *randn(int rows, int cols);
	 Matrix *randn(int rows, int cols, float mean, float std);
	 void randn(int rows, int cols, float mean, float std, Matrix *out);
	 Matrix *dropout(Matrix *A, float dropout_rate);

	 void tick(std::string name);
	 void tick();
	 void tock(std::string name);
	 void tock();

	 void benchmark_dot();
	 void shutdown_MPI();


	 void finish_batch_allocator();
	 void init_batch_allocator(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int cv_batch_size);
	 void allocate_next_batch_async();
	 void allocate_next_cv_batch_async();
	 void replace_current_batch_with_next();
	 void replace_current_cv_batch_with_next();
private:
	 cublasHandle_t m_handle;
	 curandGenerator_t m_generator;
	 std::map<std::string,cudaEvent_t*> m_dictTickTock;
	 std::map<std::string,float> m_dictTickTockCumulative;
	 std::list<MPI_Request*> m_requests;
	 std::map<std::string,Matrix*> m_matrixCache;
	 std::map<std::string,int> m_matrixCacheUsage;
	 Matrix *m_next_batch_X;
	 Matrix *m_next_batch_y;
	 Matrix *m_full_X;
	 Matrix *m_full_y;
	 int m_batch_size;
	 int m_next_batch_number;
	 cudaStream_t m_streamNext_batch_X;
	 cudaStream_t m_streamNext_batch_y;

	 int m_cv_beginning;
	 int m_batch_size_cv;
	 int m_next_batch_number_cv;
	 Matrix *m_next_batch_cv_X;
	 Matrix *m_next_batch_cv_y;
	 cudaStream_t m_streamNext_batch_cv_X;
	 cudaStream_t m_streamNext_batch_cv_y;


	 int m_nodes;
	 bool m_hasMPI;
	 MPI_Status m_status;

	 void dot(Matrix *A, Matrix *B, Matrix *out, cublasOperation_t T1, cublasOperation_t T2);
	 void init(int seed);
	 void init_MPI(int argc, char *argv[]);
	 void waitForAllRequests();
};
#endif

