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
	 Matrix *rand_int(int rows, int cols, int low, int high);

	 void tick(std::string name);
	 void tick();
	 void tock(std::string name);
	 void tock();

	 void benchmark_dot();
	 void shutdown_MPI();
	 Matrix *rdmSqrtWeight(int rows, int cols);
private:
	 cublasHandle_t m_handle;
	 curandGenerator_t m_generator;
	 std::map<std::string,cudaEvent_t*> m_dictTickTock;
	 std::map<std::string,float> m_dictTickTockCumulative;
	 std::list<MPI_Request*> m_requests;
	 std::map<std::string,Matrix*> m_matrixCache;
	 std::map<std::string,int> m_matrixCacheUsage;

	 int m_nodes;
	 bool m_hasMPI;
	 MPI_Status m_status;

	 void dot(Matrix *A, Matrix *B, Matrix *out, cublasOperation_t T1, cublasOperation_t T2);
	 void init(int seed);
	 void init_MPI(int argc, char *argv[]);
	 void waitForAllRequests();
};
#endif

