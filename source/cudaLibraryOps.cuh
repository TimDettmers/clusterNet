#ifndef cudaLibraryOps
#include <basicOps.cuh>
#include <curand.h>
#define cudaLibraryOps
Matrix dot(Matrix A, Matrix B);
void dot(Matrix A, Matrix B, Matrix out);

curandGenerator_t random_init();
curandGenerator_t random_init(int seed);
Matrix rand(curandGenerator_t gen, int rows, int cols);
void rand(curandGenerator_t gen, int rows, int cols, Matrix out);
Matrix randn(curandGenerator_t gen, int rows, int cols);
Matrix randn(curandGenerator_t gen, int rows, int cols, float mean, float std);
void randn(curandGenerator_t gen, int rows, int cols, float mean, float std, Matrix out);
#endif
