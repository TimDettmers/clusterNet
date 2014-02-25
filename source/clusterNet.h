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
	 Matrix m_current_batch_X;
	 Matrix m_current_batch_y;
	 int m_total_batches;

	 Matrix dot(Matrix A, Matrix B);
	 void dot(Matrix A, Matrix B, Matrix out);
	 Matrix dotMPI_unitSlice(Matrix A, Matrix B);
	 Matrix dotMPI_batchSlice(Matrix A, Matrix B);

	 Matrix rand(int rows, int cols);
	 void rand(int rows, int cols, Matrix out);
	 Matrix randn(int rows, int cols);
	 Matrix randn(int rows, int cols, float mean, float std);
	 void randn(int rows, int cols, float mean, float std, Matrix out);

	 void tick(std::string name);
	 void tick();
	 void tock(std::string name);
	 void tock();

	 void benchmark_dot();
	 void shutdown_MPI();

	 void finish_batch_allocator();
	 void init_batch_allocator(Matrix X, Matrix y, int batch_size);
	 void allocate_next_batch_async();
	 void replace_current_batch_with_next();
private:
	 cublasHandle_t m_handle;
	 curandGenerator_t m_generator;
	 std::map<std::string,cudaEvent_t*> m_dictTickTock;
	 std::map<std::string,float> m_dictTickTockCumulative;
	 std::list<MPI_Request*> m_requests;
	 std::map<std::string,Matrix> m_matrixCache;
	 std::map<std::string,int> m_matrixCacheUsage;
	 float* m_next_batch_X_cpu;
	 float* m_next_batch_y_cpu;
	 Matrix m_next_batch_X;
	 Matrix m_next_batch_y;
	 Matrix m_full_X;
	 Matrix m_full_y;
	 int m_batch_size;
	 int m_next_batch_number;
	 cudaStream_t m_streamNext_batch_X;
	 cudaStream_t m_streamNext_batch_y;


	 int m_nodes;
	 bool m_hasMPI;
	 MPI_Status m_status;

	 void init(int seed);
	 void init_MPI(int argc, char *argv[]);
	 void waitForAllRequests();
};
#endif

