#ifndef ClusterNet_H
#define ClusterNet_H

#include <cublas_v2.h>
#include <basicOps.cuh>
#include <curand.h>
#include <mpi.h>

class ClusterNet
{

public:
	 ClusterNet();
	 ClusterNet(int seed);
	 ClusterNet(int argc, char *argv[], int seed);
	 //~ClusterNet();
	 int m_rank;

	 Matrix dot(Matrix A, Matrix B);
	 void dot(Matrix A, Matrix B, Matrix out);
	 Matrix dotMPI(Matrix A, Matrix B);
	 void dotMPI(Matrix A, Matrix B, Matrix out);

	 Matrix rand(int rows, int cols);
	 void rand(int rows, int cols, Matrix out);
	 Matrix randn(int rows, int cols);
	 Matrix randn(int rows, int cols, float mean, float std);
	 void randn(int rows, int cols, float mean, float std, Matrix out);

	 void shutdown_MPI();
private:
	 cublasHandle_t m_handle;
	 curandGenerator_t m_generator;

	 int m_nodes;
	 bool m_hasMPI;
	 MPI_Status m_status;

	 void init(int seed);
	 void init_MPI(int argc, char *argv[]);
};
#endif
