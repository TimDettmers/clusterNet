#ifndef clusterKernels
#define clusterKernels
__global__ void kFill_with(float *m, float fill_value, int size);
__global__ void kAdd(float *m1,float *m2, float *m_out, int size);
__global__ void kSub(float *m1,float *m2, float *m_out, int size);
__global__ void kMul(float *m1,float *m2, float *m_out, int size);
__global__ void kDiv(float *m1,float *m2, float *m_out, int size);
#endif
