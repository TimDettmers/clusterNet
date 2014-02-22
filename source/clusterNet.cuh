#ifndef ClusterNet_H
#define ClusterNet_H

#include <cublas_v2.h>
#include <basicOps.cuh>
#include <curand.h>

class ClusterNet
{

public:
	 ClusterNet();
	 ClusterNet(int seed);

	 Matrix dot(Matrix A, Matrix B);
	 void dot(Matrix A, Matrix B, Matrix out);

	 Matrix rand(int rows, int cols);
	 void rand(int rows, int cols, Matrix out);
	 Matrix randn(int rows, int cols);
	 Matrix randn(int rows, int cols, float mean, float std);
	 void randn(int rows, int cols, float mean, float std, Matrix out);
private:
	 cublasHandle_t m_handle;
	 curandGenerator_t m_generator;

	 void init(int seed);
};
#endif
