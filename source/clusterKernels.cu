#include <basicOps.cuh>
#include "curand.h"
#include "curand_kernel.h"

__global__ void kFill_with(float *m, float fill_value, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m[i] = fill_value;
  }
}

__global__ void kAdd(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       out[i] = A[i] + B[i];
  }
}

__global__ void kMul(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       out[i] = A[i] * B[i];
  }
}

__global__ void kSub(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       out[i] = A[i] - B[i];
  }
}

__global__ void kDiv(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       out[i] = A[i] / B[i];
  }
}

__global__ void kExp(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       out[i] = __expf(A[i]);
  }
}

__global__ void kSqrt(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       out[i] = sqrt(A[i]);
  }
}

__global__ void kLog(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       out[i] = __logf(A[i]);
  }
}

__global__ void kSquare(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       out[i] = __powf(A[i], 2);
  }
}

__global__ void kScalarMul(float *A, float scalar, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       out[i] = scalar*A[i];
  }
}

 
__global__ void kTranspose(float *A, float *out, int width, int height) 
{
    __shared__ float block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < width) && (yIndex < height)) 
    {
        unsigned int index_in = yIndex * width + xIndex;

        block[threadIdx.y][threadIdx.x] = A[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < height) && (yIndex < width)) 
    {
        unsigned int index_out = yIndex * height + xIndex;

        out[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

__global__ void setup_kernel(curandState *state, int seed)
{
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}


__global__ void generate_uniform_kernel(curandState *state, int size, float *out)
{	
  //each thread generates 256 random numbers
  //max random numbers generated in one go: 256*1024*1024 = 268435456

  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  curandState localState = state[idx];	 
  int start = (blockIdx.x * blockDim.x * 256) + threadIdx.x *256;
  int end = start+256;

  for(unsigned int i = start; (i < end) && (i < size); i++)
  { 
    out[i]  = curand_uniform(&localState);	 
  }

  state[idx] = localState;
}

__global__ void generate_normal_kernel(curandState *state, int size, float *out)
{	
  //each thread generates 256 random numbers
  //max random numbers generated in one go: 256*1024*1024 = 268435456

  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  curandState localState = state[idx];	 
  int start = (blockIdx.x * blockDim.x * 256) + threadIdx.x *256;
  int end = start+256;

  for(unsigned int i = start; (i < end) && (i < size); i++)
  { 
    out[i]  = curand_normal(&localState);	 
  }

  state[idx] = localState;
}








