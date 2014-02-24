#include <basicOps.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

__global__ void kFill_with(float *m, float fill_value, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       m[i] = fill_value;
}

//vertical stack for column major format
__global__ void vStack(float *A, float *B, float *out, int size_out, int rows_a, int rows, int cols)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  int current_col = 0;
  int current_row = 0;
  int offset = 0;
  const int rows_b = rows - rows_a;

  for (unsigned int i = idx;i < size_out; i += numThreads)
  {
	  current_col = i / rows; //int arithmetic
	  offset = (current_col*rows);
	  current_row = i - offset;

	  if(current_row >= rows_a)
	  {
		  //fetch b value
		  out[i] = B[(current_col*rows_b) + current_row - rows_a];
	  }
	  else
	  {
		  //fetch a value
		  out[i] = A[(current_col*rows_a) + current_row];
	  }
  }
}

__global__ void hStack(float *A, float *B, float *out, int size_out, int size_a)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for(unsigned int i = idx; i < size_out; i+=numThreads)
  {
	  if(i >= size_a)
	  {
		  //append B
		  out[i] = B[i - size_a];
	  }
	  else
	  {
		  //append A
		  out[i] = A[i];
	  }
  }

}

__global__ void kAdd(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = A[i] + B[i];
}

__global__ void kMul(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = A[i] * B[i];
}

__global__ void kSub(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = A[i] - B[i];
}

__global__ void kDiv(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = A[i] / B[i];
}

__global__ void kExp(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = __expf(A[i]);
}

__global__ void kSqrt(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = sqrt(A[i]);
}

__global__ void kLog(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = __logf(A[i]);
}

__global__ void kSquare(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = __powf(A[i], 2);
}

__global__ void kScalarMul(float *A, float scalar, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = scalar*A[i];
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

//for column major data
__global__ void slice_rows(float *A, float *out, int size_out, int rows_A, int start, int end)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int current_col = 0;
  int current_row = 0;
  int offset = 0;
  int rows_out = (end - start) + 1;

  for (unsigned int i = idx;i < size_out; i += numThreads)
  {
	  current_col = i / rows_out; //note: int arithmetic
	  current_row = i - (current_col*rows_out);
	  offset = rows_A*current_col;

	  out[i] = A[offset + start + current_row];
  }
}

//for column major data
__global__ void slice_cols(float *A, float *out, int start, int rows, int size_out)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx; i < size_out; i += numThreads)
  {
     out[i] = A[i+(start*rows)];
  }
}


