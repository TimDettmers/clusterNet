#include <basicOps.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <float.h>
const int NUM_THREADS = 32;

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

    // read the Matrix *tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < width) && (yIndex < height)) 
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = A[index_in];
    }

    __syncthreads();

    // write the transposed Matrix *tile to global memory
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



__device__ void reduceToMax(float* sdata, unsigned int tid)
{

  //Synchronize threads to share shared memory data
  __syncthreads();

  float mySum = sdata[tid];

  // do reduction in shared mem
  if (NUM_THREADS >= 512) { if (tid < 256) { sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 256]); } __syncthreads(); }
  if (NUM_THREADS >= 256) { if (tid < 128) { sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 128]); } __syncthreads(); }
  if (NUM_THREADS >= 128) { if (tid <  64) { sdata[tid] = mySum = fmaxf(mySum, sdata[tid +  64]); } __syncthreads(); }

  if (NUM_THREADS == 32){
    if (tid < 16)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (NUM_THREADS >=  32) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 16]); }
      if (NUM_THREADS >=  16) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  8]); }
      if (NUM_THREADS >=   8) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  4]); }
      if (NUM_THREADS >=   4) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  2]); }
      if (NUM_THREADS >=   2) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  1]); }
    }
  }
  else
  {
    if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (NUM_THREADS >=  64) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 32]); }
      if (NUM_THREADS >=  32) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 16]); }
      if (NUM_THREADS >=  16) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  8]); }
      if (NUM_THREADS >=   8) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  4]); }
      if (NUM_THREADS >=   4) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  2]); }
      if (NUM_THREADS >=   2) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  1]); }
    }
  }
}

__device__ void reduceToSumLocal(float* sdata, unsigned int tid)
{

  //Synchronize threads to share shared memory data
  __syncthreads();

  float mySum = sdata[tid];

  // do reduction in shared mem
  if (NUM_THREADS >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
  if (NUM_THREADS >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
  if (NUM_THREADS >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

  if (NUM_THREADS == 32){
    if (tid < 16)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (NUM_THREADS >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
      if (NUM_THREADS >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
      if (NUM_THREADS >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
      if (NUM_THREADS >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
      if (NUM_THREADS >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }
  }
  else
  {
    if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (NUM_THREADS >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
      if (NUM_THREADS >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
      if (NUM_THREADS >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
      if (NUM_THREADS >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
      if (NUM_THREADS >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
      if (NUM_THREADS >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }
  }
}

//taken from cudamat
__global__ void kSoftMax(float* mat, float* target, unsigned int rows, unsigned int cols)
{
  extern __shared__ float max_vals[] ;
  float cur_max = -FLT_MAX;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < rows) {
    float *cur_data = &mat[column * cols] ;
    max_vals[threadIdx.x]=-FLT_MAX;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
      val = cur_data[i];
      if (val > cur_max) {
        cur_max = val;
      }
    }
    max_vals[threadIdx.x] = cur_max;
    reduceToMax(max_vals, threadIdx.x);
    __syncthreads();
    cur_max = max_vals[0] ;
    __syncthreads();
    val = 0;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
      val += __expf(cur_data[i]-cur_max);
    }
    max_vals[threadIdx.x] = val;
    reduceToSumLocal(max_vals, threadIdx.x);
    __syncthreads();
    float norm = max_vals[0] ;
    float *cur_target = &target[column * cols] ;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
      cur_target[i] = __expf(cur_data[i]-cur_max) / norm ;
    }
  }
}

//for column major data
__global__ void kSubMatrixVector(float *A, float *v, float *out, int rows, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  //offset = current_column * rows
  int offset = 0;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  offset = (i / rows)*rows; //note: int arithmetic
	  out[i] =  A[i] - v[i - offset];
  }
}

//cudamat kernel for column major data
__global__ void kArgMaxRowwise(float* A, float* out, unsigned int rows, unsigned int cols)
{
  __shared__ float max_vals[32];
  __shared__ unsigned int max_val_args[32];
  float cur_max = -FLT_MAX;
  unsigned int cur_argmax = 0;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < rows) {
    float *cur_data = &A[column * cols] ;
    for (unsigned int i = threadIdx.x; i < cols; i += blockDim.x) {
      val = cur_data[i];
      if (val > cur_max) {
        cur_max = val;
        cur_argmax = i;
      }
    }
    max_vals[threadIdx.x] = cur_max;
    max_val_args[threadIdx.x] = cur_argmax;
    __syncthreads();
    if (threadIdx.x == 0) {
      cur_max = -FLT_MAX;
      cur_argmax = 0;
      for (unsigned int i = 0; i < blockDim.x; i++)
        if (max_vals[i] > cur_max) {
          cur_max = max_vals[i];
          cur_argmax = max_val_args[i];
        }
      out[column] = cur_argmax;
    }
  }
}

__global__ void kCreate_t_matrix(float *labels, float *out, int rows, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  int label = 0;
	  int offset = 0;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  label = (int)(labels[i]);
		  //offset = (label*rows) gives the current column; i gives the current row
		  offset = (label*rows) + i;
		  out[offset] = 1.0f;
	  }

}

__global__ void kEqual(float *A, float *B, float *out, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  out[i] = (float)(A[i] == B[i]);
	  }
}
