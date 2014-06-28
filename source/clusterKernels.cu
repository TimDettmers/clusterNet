#include <basicOps.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
const int NUM_THREADS = 32;


__global__ void kGetNonZeroElements(float *A, float *out, int size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	out[0] = 0.0f;
	for (unsigned int i = idx;i < size; i += numThreads)
		 atomicAdd(&out[0],A[i] != 0.0f ? 1.0f : 0.0f);
}



__global__ void kFill_with(float *m, float fill_value, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       m[i] = fill_value;
}

__global__ void kFill_with(int *m, int fill_value, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       m[i] = fill_value;
}

__global__ void kCreateRdmSqrtWeight_Logistic(float *A, int in, int out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const float lower_limit = -4.0f*sqrtf(6.0f/((float)in + out));
  const float upper_limit =  4.0f*sqrtf(6.0f/((float)in + out));
  const float range = upper_limit-lower_limit;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       A[i] = lower_limit + (A[i]*range);
  }
}

__global__ void kCreateSparseRdmWeight(float *rdm, float* indicies, float *out, int rows, int cols, int connections)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int connection_idx = 0;
  float rdm_value = 0.0f;
  int size = connections*cols;
  int current_col = 0;

  //each thread fills one row
  for (unsigned int i = idx; i < size; i += numThreads)
  {
	  connection_idx = (int)indicies[i];
	  rdm_value = rdm[i];
	  current_col = i/(connections);
	  out[(current_col*rows)+connection_idx] = rdm_value;
  }
}

__global__ void kRandInt(float *A, int lower_limit, int upper_limit, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int range = upper_limit-lower_limit + 1;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  //use uniform random sample to get integers
       A[i] = (float)(((int)((A[i]*range))) + lower_limit);
  }
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

__global__ void hStackN(float **arrA, int general_size, float *out, int size_out, int matrices_count)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int current_matrix = 0;

  for(unsigned int i = idx; i < size_out; i+=numThreads)
  {
	  current_matrix = i / general_size;
	  current_matrix = current_matrix == matrices_count ? current_matrix - 1 : current_matrix;
	  out[i] = arrA[current_matrix][i - (current_matrix*general_size)];
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

__global__ void kSub_Sparse(float *A, float *data, int *ptr_rows, int *idx_cols, float *out, int rows, int cols, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row_idx = 0;

  for (unsigned int i = idx;i < rows*cols; i += numThreads)
	  out[i] = A[i];

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  for(int j = 0; j < rows + 1; j++)
	  {
		  if(ptr_rows[j] > i)
		  {
			  row_idx = j-1;
			  break;
		  }
	  }
      out[(idx_cols[i] * rows) + row_idx] = A[(idx_cols[i] * rows) + row_idx] - data[i];
  }
}

__global__ void kDiv(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = fdividef(A[i],B[i]);
}

__global__ void kExp(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = expf(A[i]);
}

__global__ void kLogistic(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = 1.0f / (1.0 + expf(-A[i]));

}

__global__ void kLogisticGrad(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = A[i]*(1 - A[i]);

}

__global__ void kSqrt(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = sqrtf(A[i]);
}

__global__ void kLog(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = logf(A[i]);
}

__global__ void kSquare(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = powf(A[i], 2.0f);
}

__global__ void kScalarMul(float *A, float scalar, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = scalar*A[i];
}

__global__ void kScalarAdd(float *A, float scalar, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = A[i]+scalar;
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

__device__ void reduceToMaxAndArgMax(float* sdataMax, float* sdataArgMax, unsigned int tid, int threads)
{

	//Synchronize threads to share shared memory data
	__syncthreads();

  	  float mySum = sdataMax[tid];
  	if(threads == 32)
  	{
		if (tid < 16)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile float* smemMax = sdataMax;
			volatile float* smemArgMax = sdataArgMax;
			if (NUM_THREADS >=  32) if(mySum < smemMax[tid + 16]){smemMax[tid] = mySum = smemMax[tid + 16];  smemArgMax[tid] = smemArgMax[tid + 16]; }
			if (NUM_THREADS >=  16) if(mySum < smemMax[tid +  8]){smemMax[tid] = mySum = smemMax[tid +  8];  smemArgMax[tid] = smemArgMax[tid +  8]; }
			if (NUM_THREADS >=   8) if(mySum < smemMax[tid +  4]){smemMax[tid] = mySum = smemMax[tid +  4];  smemArgMax[tid] = smemArgMax[tid +  4]; }
			if (NUM_THREADS >=   4) if(mySum < smemMax[tid +  2]){smemMax[tid] = mySum = smemMax[tid +  2];  smemArgMax[tid] = smemArgMax[tid +  2]; }
			if (NUM_THREADS >=   2) if(mySum < smemMax[tid +  1]){smemMax[tid] = mySum = smemMax[tid +  1];  smemArgMax[tid] = smemArgMax[tid +  1]; }
		}
  	}
	else
	{
		if (tid < 32)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile float* smemMax = sdataMax;
			volatile float* smemArgMax = sdataArgMax;
			if (NUM_THREADS >=  64) if(mySum < smemMax[tid + 32]){smemMax[tid] = mySum = smemMax[tid + 32];  smemArgMax[tid] = smemArgMax[tid + 32]; }
			if (NUM_THREADS >=  32) if(mySum < smemMax[tid + 16]){smemMax[tid] = mySum = smemMax[tid + 16];  smemArgMax[tid] = smemArgMax[tid + 16]; }
			if (NUM_THREADS >=  16) if(mySum < smemMax[tid +  8]){smemMax[tid] = mySum = smemMax[tid +  8];  smemArgMax[tid] = smemArgMax[tid +  8]; }
			if (NUM_THREADS >=   8) if(mySum < smemMax[tid +  4]){smemMax[tid] = mySum = smemMax[tid +  4];  smemArgMax[tid] = smemArgMax[tid +  4]; }
			if (NUM_THREADS >=   4) if(mySum < smemMax[tid +  2]){smemMax[tid] = mySum = smemMax[tid +  2];  smemArgMax[tid] = smemArgMax[tid +  2]; }
			if (NUM_THREADS >=   2) if(mySum < smemMax[tid +  1]){smemMax[tid] = mySum = smemMax[tid +  1];  smemArgMax[tid] = smemArgMax[tid +  1]; }
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

__global__ void kSoftMax(float* A, float* out, unsigned int rows, unsigned int cols)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float col_value = 0.0f;

	__shared__ float max_values[THREADS_PER_BLOCKS];
	__shared__ float row_sums[THREADS_PER_BLOCKS];

	for (unsigned int row = idx; row < rows; row += numThreads)
	{
		//fill with min values
		max_values[idx] = -FLT_MAX;
		row_sums[idx] = 0.0f;

		 //calc max value of the row
		for (unsigned int i = 0; i < cols; i++)
		{
			col_value = A[(i*rows) + row];
			if(col_value > max_values[idx])
			{
				max_values[idx] = col_value;
			}
		}

		//calc the row sum
		for (unsigned int i = 0; i < cols; i++)
		{
			row_sums[idx] += __expf(A[(i*rows) + row] - max_values[idx]);
		}

		//calc the value of each element in the row
		for (unsigned int i = 0; i < cols; i++)
		{
			out[(i*rows) + row] = __expf(A[(i*rows) + row] - max_values[idx])/row_sums[idx];
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

//for column major data
__global__ void kAddMatrixVector(float *A, float *v, float *out, int rows, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  //offset = current_column * rows
  int offset = 0;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  offset = (i / rows); //note: int arithmetic
	  out[i] =  A[i] + v[offset];
  }
}

__global__ void kArgmax(float* A, float* out, unsigned int rows, unsigned int cols)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float max_value = -FLT_MAX;
	  float max_i = 0;
	  float col_value = 0.0f;

	  for (unsigned int row = idx; row < rows; row += numThreads)
	  {
		  for (unsigned int i = 0; i < cols; i++)
		  {
			  col_value = A[(i*rows) + row];
			  if(col_value > max_value)
			  {
				  max_value = col_value;
				  max_i = i;
			  }

		  }
		  out[row] = max_i;
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

__global__ void kRectifiedLinear(float *A, float *out, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
		  out[i] = A[i] > 0.0f ? A[i] : 0.0f;

}

__global__ void kRectifiedLinear_Derivative(float *A, float *out, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
		  out[i] = A[i] > 0.0f ? 1.0f : 0.0f;

}

__global__ void kDoubleRectifiedLinear(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  float value = 0.0f;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  value = (A[i] > 0.0f) ? A[i] : 0.0f;
      out[i] = (value < 1.0f) ? value : 1.0f;
  }
}

__global__ void kDoubleRectifiedLinear_Derivative(float *A, float *out, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  out[i] = (A[i] <= 0.0f) || (A[i] >=1.0f) ? 0.0f : 1.0f;
	  }

}

__global__ void kHardTanH(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  float value = 0.0f;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  value = (A[i] > 1.0f) ? A[i] : 1.0f;
      out[i] = (value < -1.0f) ? value : -1.0f;
  }
}

__global__ void kPairwise_ranking(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  float value = 0.0f;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  value = 1.0f - A[i] + B[i];
      out[i] = value < 0.0f ? 0.0f : value;
  }
}

__global__ void kPairwise_ranking_derivative(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
      out[i] = (1.0f - A[i] + B[i]) > 0.0f ? 1.0f : 0.0f;

}

__global__ void kHardTanH_Derivative(float *A, float *out, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
		  out[i] = (A[i] < -1.0f) || (A[i] >1.0f) ? 0.0f : 1.0f;

}

__global__ void kSquaredError(float *A, float *t, float *out, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
		  out[i] = powf(A[i] -t[i],2.0f);
}

__global__ void kSum(float *v, float *out, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  out[0] = 0.0f;
	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  atomicAdd(&out[0],v[i]);
	  }
}


__global__ void kArange(float *out, int start, int rows, int cols, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  int offset = 0;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  offset = (i % rows)*cols;

		  out[i] = (float)(offset + (i/rows) + start);
	  }
}

__global__ void kDropout(float *A, float *rdm, float dropout, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
		  rdm[i] = rdm[i] > dropout ? A[i] : 0.0f;

}

__global__ void kRMSprop(float *RMS, float *grad, float RMS_multiplier, float learning_rate, int batch_size, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = fdividef(grad[i],(float)batch_size);
		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);

		  grad[i] = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));
		  RMS[i] = RMS_value;
	  }

}

__global__ void kRMSprop_with_nesterov_weight_update(float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;
	  float weight_value = 0.0f;
	  float momentum_matrix_value = 0.0f;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = fdividef(grad[i],(float)batch_size);
		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
		  grad_value = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));
		  momentum_matrix_value = m[i];
		  weight_value = w[i]-momentum_matrix_value;
		  momentum_matrix_value -= grad_value;

		  RMS[i] = RMS_value;
		  m[i] = momentum_matrix_value;
		  w[i] = weight_value + momentum_matrix_value;
	  }
}

__global__ void kSparseDot(int m, int n, int k, float *data, int* indptr, int* indices, float *dense_data, float* target, float beta, float alpha)
{
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n)
  {
	  /*
	  for(int i = 0; i < indptr[m+1];i++)
		  if(indices[i] > 23)
		  {
			  printf("ERROR: \n");
			  printf("%i \n", indices[i]);
	    	  printf("col: %i \n", col);
	    	  printf("row: %i \n", row);
		  }
		  */

	  int max_idx = indptr[m+1];
	  for(int i = 0; i < m+1;i++)
		  if(indptr[i] > max_idx)
		  {
			  printf("ERROR: \n");
			  printf("%i \n", indptr[i]);
	    	  printf("max_idx: %i \n", max_idx);
		  }


    const int start = indptr[row];
    const int end = indptr[row + 1];
    float sum = 0.f;
    for (int i = start; i < end; i++)
    {
    	/*
    	for(int a = start; a < end;a++)
    			  if(indices[a] > 23)
    			  {
    				  printf("ERROR: \n");
    				  printf("%i \n", indices[a]);
    		    	  printf("a: %i \n", a);
    			  }
    			  */


      sum += data[i]  * dense_data[(col * k) + indices[i]];
      if(sum > 500000 || sum < -500000)
      {

    	  printf("start: %i ", start);
    	  printf("end: %i ", end);
    	  printf("i: %i ", i);
    	  printf("k: %i ", k);
    	  printf("col: %i ", col);
    	  printf("data idx %i ", indices[i]);
    	  printf("full idx %i ", (col * k) + indices[i]);
    	  printf("data sparse %f ", data[i]);
    	  printf("data dense %f ", dense_data[col * k + indices[i]]);
    	 printf("data point %f ", data[i]  * dense_data[col * k + indices[i]]);
         printf(" sum %f\n", sum);



         return;
      }
    }
    const int pos = col * m + row;
    target[pos] = alpha * sum + ((beta == 0) ? 0 : beta * target[pos]);
  }
}

__global__ void kPrintData(float *A, int size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	__syncthreads();
	if(idx == 0)
		printf("[");
	for (unsigned int i = idx;i < size; i += numThreads)
		 printf("%f ",A[i]);
	__syncthreads();
	if(idx == 0)
	printf("]\n");
}

__global__ void kMaxout(float *A, float *out, float *outargmax, int maxout_level, unsigned int cols, unsigned int rows)
{
  __shared__ float max_values[32];
  __shared__ float argmax_values[32];
  float const min_value = -FLT_MAX;

  for(int row = blockIdx.x; row < rows; row +=blockDim.x)
  {
	  int softout_block_idx = row + (blockIdx.y*maxout_level*rows);
	  if(threadIdx.x < maxout_level)
	  {
		  max_values[threadIdx.x] = A[softout_block_idx+(threadIdx.x*rows)];
		  argmax_values[threadIdx.x] = (float)((blockIdx.y*maxout_level)+threadIdx.x);
	  }
	  else
	  {
		  max_values[threadIdx.x] = min_value;
		  argmax_values[threadIdx.x] = -1.0f;
	  }

	  //reduceToMax(max_values, threadIdx.x);
	  reduceToMaxAndArgMax(max_values, argmax_values, threadIdx.x, 32);
	  __syncthreads();
	  if(threadIdx.x == 0) out[row + (blockIdx.y*rows)] = max_values[0];
	  if(threadIdx.x == 1) outargmax[row + (blockIdx.y*rows)] = argmax_values[0];
  }
}


__global__ void kMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height)
{
  extern __shared__ float max_vals[];
  float cur_max = -FLT_MAX;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float *cur_data = &mat[column * height] ;
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      val = cur_data[i];
      if (val > cur_max) cur_max = val;
    }
    max_vals[threadIdx.x] = cur_max;
    reduceToMax(max_vals, threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) target[column] = max_vals[0];
  }
}


__global__ void kExpandToMaxoutGrad(float* error, float* indexes, float *out, int error_size, int error_rows, int maxout_level)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int grad_size = maxout_level*error_size;

    for (unsigned int i = idx;i < grad_size; i += numThreads)
    	out[i] = 0.0f;

	for (unsigned int i = idx;i < error_size; i += numThreads)
	{
		int row_idx = idx - ((idx / error_rows)*error_rows);
		out[row_idx + (((int)indexes[idx])*error_rows)] = error[i];
	}
}

__global__ void kConstructVocabMatrix(float *vocab_idx, float *vocab_idx_y, float* vocab, float *rdm_idx, float *batch_X, float *batch_Y)
{
	int middleIdx = (gridDim.y/2);
	int myIdx = 0;
	int myRdmIdx = 0;

	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	//middle index is replaced by rdm word for batch_Y, but we still need to write the correct into batch_X!
	if(blockIdx.y != middleIdx)
	{
		myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];//gridDim.x = vocab_idx_vector rows == batch size
		vocab_idx_y[blockIdx.x+(blockIdx.y*gridDim.x)] = (float)myIdx;
	}
	else
	{
		myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
		myRdmIdx = (int)rdm_idx[blockIdx.x];
		vocab_idx_y[blockIdx.x+(blockIdx.y*gridDim.x)] = (float)myRdmIdx;
	}

	int myVocabIdx = blockDim.x*myIdx;
	int myVocabRdmIdx = blockDim.x*myRdmIdx;

	if(blockIdx.y != middleIdx)
	{
		batch_X[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)] = vocab[myVocabIdx + threadIdx.x];
		batch_Y[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)] = vocab[myVocabIdx + threadIdx.x];
	}
	else
	{
		batch_X[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)] = vocab[myVocabIdx + threadIdx.x];
		batch_Y[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)] = vocab[myVocabRdmIdx + threadIdx.x];
	}



}


/*
 //numerically unstable?
__global__ void kUpdateVocabWithGradient(float *grad, float *vocab_idx, float* vocab, float learning_rate)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	int myIdx = 0;
	float multiplier = -fdividef(learning_rate,float(gridDim.x));
	myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myVocabIdx = blockDim.x*myIdx;



	//printf("%f ",grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]*multiplier);
	//printf("%f ",vocab[myVocabIdx + threadIdx.x]);
	//printf("%f ",vocab[myVocabIdx + threadIdx.x]+ (grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]*multiplier));
	if(myIdx > 10000)
		atomicAdd(&vocab[myVocabIdx + threadIdx.x],grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]*multiplier);
	//vocab[myVocabIdx + threadIdx.x] +=grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)];
	//printf("%s ",!isfinite(grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]*multiplier));

}
*/


/*
//numerically unstable?
__global__ void kUpdateVocabWithGradient(float *grad, float *vocab_idx, float* vocab, float learning_rate, int batch_size, int window_size, int vocab_rows, int vocab_cols)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	int myIdx = 0;
	float multiplier = -fdividef(learning_rate,float(batch_size));
	int size = window_size*batch_size;
	int myVocabIdx = 0;
	int grad_row = 0;
	int grad_col = 0;
	for(int i = threadIdx.x; i < size; i+=blockDim.x)
	{
		myIdx = (int)vocab_idx[i];
		grad_row = i/window_size;
		grad_col = i - grad_row;

		myVocabIdx = vocab_rows*myIdx;
		for(int j = 0; j < vocab_rows; j++)
		{
			atomicAdd(&vocab[myVocabIdx + j],grad[grad_row + (grad_col*vocab_rows)]*multiplier);
		}
	}
}

*/

__global__ void kUpdateVocabWithGradient(float *gradX, float *gradY, float *vocab_idx_X, float *vocab_idx_Y, float* vocab,
										 float *vocab_grad, float *vocab_grad_idx, float learning_rate, int grad_size)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y


	float multiplier = fdividef(learning_rate,(float)(gridDim.x*2));
	int myIdx_X = (int)vocab_idx_X[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myIdx_Y = (int)vocab_idx_Y[blockIdx.x+(blockIdx.y*gridDim.x)];
	int grad_cols = grad_size/blockDim.x;

	int myVocabIdx_X = blockDim.x*myIdx_X;
	int myVocabIdx_Y = blockDim.x*myIdx_Y;


	atomicAdd(&vocab_grad[myVocabIdx_X + threadIdx.x],gradX[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]);
	atomicAdd(&vocab_grad[myVocabIdx_Y + threadIdx.x],gradY[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]);
	/*
	vocab_grad_idx[myIdx_X] = 1.0f;
	vocab_grad_idx[myIdx_Y] = 1.0f;

	__syncthreads();




	int block_idx = (blockIdx.y*gridDim.x) + blockIdx.x;
	int threads_blocks = gridDim.x*gridDim.y;
	for(int i = block_idx; i < grad_cols; i+=threads_blocks)
	{
		if(vocab_grad_idx[i] == 1.0f)
		{
			vocab[(i*blockDim.x) + threadIdx.x] -= vocab_grad[(i*blockDim.x) + threadIdx.x]*multiplier;
		}
	}

	*/



}




