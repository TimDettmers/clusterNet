#ifndef cudaLibraryOps
#define cudaLibraryOps
#include <basicOps.cuh>
Matrix dot(Matrix A, Matrix B);
void dot(Matrix A, Matrix B, Matrix out);
Matrix rand(int rows, int cols);
Matrix rand(int rows, int cols, int seed);
Matrix randn(int rows, int cols);
Matrix randn(int rows, int cols, int seed);
#endif
