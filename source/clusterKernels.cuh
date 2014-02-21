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
__global__ void slice_rows(float *A, int start, int end, int cols, float *out);
__global__ void slice_cols(float *A, int start, int end, int rows, int cols, float *out);
__global__ void kMerge(float *A, float *B, float *out, int size_a, int size_b);
#endif
