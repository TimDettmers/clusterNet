#ifndef clusterKernels
#include "curand.h"
#include "curand_kernel.h"
#define clusterKernels
__global__ void kFill_with(float *m, float fill_value, int size);
__global__ void kAdd(float *A,float *B, float *out, int size);
__global__ void kSub(float *A,float *B, float *out, int size);
__global__ void kMul(float *A,float *B, float *out, int size);
__global__ void kDiv(float *A,float *B, float *out, int size);
__global__ void kExp(float *A, float *out, int size);
__global__ void kLog(float *A, float *out, int size);
__global__ void kSqrt(float *A, float *out, int size);
__global__ void kSquare(float *A, float *out, int size);
__global__ void kScalarMul(float *A, float scalar, float *out, int size);
__global__ void kTranspose(float *A, float *out, int width, int height); 
__global__ void setup_kernel(curandState *state, int seed);
__global__ void generate_uniform_kernel(curandState *state, int size, float *out);
__global__ void generate_normal_kernel(curandState *state, int size, float *out);
__global__ void slice_rows(float *A, float *out, int size_out, int rows_A, int start, int end);
__global__ void slice_cols(float *A, float *out, int start, int rows, int size_out);
__global__ void vStack(float *A, float *B, float *out, int size_out, int rows_a, int rows, int cols);
__global__ void hStack(float *A, float *B, float *out, int size_out, int size_a);
__global__ void kSoftMax(float* mat, float* target, unsigned int width, unsigned int height);
__device__ void reduceToMax(float* sdata, unsigned int tid);
__device__ void reduceToSumLocal(float* sdata, unsigned int tid);
__global__ void kSubMatrixVector(float *A, float *v, float *out, int rows, int size);
__global__ void kArgMaxRowwise(float* A, float* out, unsigned int height, unsigned int width);
__global__ void kCreate_t_matrix(float *labels, float *out, int rows, int size);
__global__ void kEqual(float *A, float *B, float *out, int size);
__global__ void vectorSum(float *v, float *out, int size);
#endif
