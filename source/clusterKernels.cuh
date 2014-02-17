#ifndef clusterKernels
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
#endif
