#ifndef cudaLibraryOps
#include <basicOps.cuh>
#include <curand.h>
#define cudaLibraryOps
Matrix dot(Matrix A, Matrix B);
void dot(Matrix A, Matrix B, Matrix out);
void dot2(Matrix A, Matrix B, Matrix out);
void ourSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc );

curandGenerator_t random_init();
curandGenerator_t random_init(int seed);
Matrix rand(curandGenerator_t gen, int rows, int cols);
void rand(curandGenerator_t gen, int rows, int cols, Matrix out);
Matrix randn(curandGenerator_t gen, int rows, int cols);
Matrix randn(curandGenerator_t gen, int rows, int cols, float mean, float std);
void randn(curandGenerator_t gen, int rows, int cols, float mean, float std, Matrix out);
#endif
