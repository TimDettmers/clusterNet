#ifndef util
#define util
#include <basicOps.cuh>
Matrix read_csv(char* filename);
cudaEvent_t* tick();
void tock(cudaEvent_t *startstop);
int test_eq(float f1, float f2, char* message);
int test_eq(float f1, float f2, int idx1, int idx2, char* message);
int test_eq(int i1, int i2, char* message);
int test_eq(int i1, int i2, int idx1, int idx2, char* message);
int test_matrix(Matrix A, int rows, int cols);
void print_matrix(Matrix A);
void print_gpu_matrix(Matrix A);
#endif
