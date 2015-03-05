#include <basicOps.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

const int NUM_THREADS = 32;


__global__ void kGetNonZeroElements(float *A, float *out, int size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	for (unsigned int i = idx;i < size; i += numThreads)
		 atomicAdd(&out[0],A[i] != 0.0f ? 1.0f : 0.0f);
}

__global__ void kGetNonZeroColumns(float *A, float *out, int rows, int cols)
{
	const int myCol = (blockIdx.x * blockDim.x) + threadIdx.x;
	float result = 0.0f;

	if(myCol < cols)
	{
		for (unsigned int i = 0;i < rows; i++)
		{
			if(A[(myCol*rows) + i] != 0.0f)
				result = 1.0f;
		}

		atomicAdd(&out[0],result);
	}
}

__global__ void kRenormalizeWeights(float *w, float *unit_sums, float limit, int rows, int cols)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int size = rows*cols;

	int myCol = 0;
	float rel_diff = 0.0f;
	for (unsigned int i = idx;i < size; i += numThreads)
	{
		myCol = i/rows;
		if(unit_sums[myCol] > limit)
		{
			rel_diff = 1.0f/unit_sums[myCol];
			w[i] *= rel_diff;
		}
		else{ continue; }

	}

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

__global__ void kRdmNumbers(float *seed, int size, float *out)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned long long s[ 2 ];
	//s[0] = (long long)seed[(gridDim.x*blockIdx.x)  + threadIdx.x];
	//s[1] = (long long)seed[(gridDim.x*(blockIdx.x+1))  + threadIdx.x];

	s[0] = 17;
	s[1] = 83;
	unsigned long long s1 = s[ 0 ];
	unsigned long long s0 = s[ 1 ];
	unsigned long long rdm64 = 23459867034598355;


	if(idx == 0)
	{
		printf("rdm: %i\n", rdm64);
		printf("rdm1: %i\n", (unsigned int)(rdm64&0xffffffff));
		printf("rdm2: %i\n", (unsigned int)((rdm64>>32)&0xffffffff));
	}

    unsigned int rdm32_1 = 0;
    unsigned int rdm32_2 = 0;
	//printf("seed 1: %i\n", seed[(gridDim.x*blockIdx.x)  + threadIdx.x]);
	//printf("seed 2: %i\n", seed[(gridDim.x*(blockIdx.x+1))  + threadIdx.x]);
	//printf("idx: %i\n", idx);
	for(int i = idx*2; i < size; i+=numThreads*2)
	{
		s1 = s[0];
		s0 = s[1];
		s[0] = s0;
		s1 ^= s1 << 23; // a

		rdm64 =  (s[1 ] = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0; // b, c

		rdm32_1 = (rdm64&0xffffffff);
		rdm32_2 = ((rdm64>>32)&0xffffffff);
		out[i] = rdm32_1;
		out[i+1] = rdm32_2;

	}

	seed[(gridDim.x*blockIdx.x)  + threadIdx.x] = s[0];
	seed[(gridDim.x*(blockIdx.x+1))  + threadIdx.x] = s[1];

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

__global__ void vStackN(float **arrA, float *out, int rows, int cols)
{
  int size = rows*cols;
  int offset = rows*cols*blockIdx.x;

  for(unsigned int i = threadIdx.x; i < size; i+=blockDim.x)
	  out[offset + i] = arrA[blockIdx.x][i];

}

__global__ void AddGradientsN(float **arrA, int size, int myrank, int matrix_count, float multiplier)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	for(int matrix_idx = 0; matrix_idx < matrix_count; matrix_idx++)
	{
		if(matrix_idx == myrank){ continue; }

		for(unsigned int i = idx; i < size; i+=numThreads)
			arrA[myrank][i] += arrA[matrix_idx][i];
	}
	//better numerical stability to do it afterwards
	for(unsigned int i = idx; i < size; i+=numThreads)
		arrA[myrank][i] *=multiplier;

}

__global__ void hStackN(Matrix **arrA, int general_size, float *out, int size_out, int matrices_count)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int current_matrix = 0;

  for(unsigned int i = idx; i < size_out; i+=numThreads)
  {
	  current_matrix = i / general_size;
	  current_matrix = current_matrix == matrices_count ? current_matrix - 1 : current_matrix;
	  out[i] = arrA[current_matrix]->data[i - (current_matrix*general_size)];
  }

}

__global__ void kAdd_to_z(float *z, float *z1, float *y, float *y_count, int rows, int cols, float *out)
{
	float value = 0;
	for(int row = blockIdx.x; row < rows; row +=gridDim.x)
	{
		int cls = (int)y[row];
		if(threadIdx.x == 0)
			atomicAdd(&y_count[cls],1.0f);
		for (unsigned int col = threadIdx.x; col < cols; col += blockDim.x)
		{
			value = z1[row + (col*rows)];
			atomicAdd(&out[cls+(col*rows)],value);
		}
	}

	__syncthreads();

	for(int row = blockIdx.x; row < rows; row +=gridDim.x)
	{
		int cls = (int)y[row];
		for (unsigned int col = threadIdx.x; col < cols; col += blockDim.x)
		{
			if(y_count[cls] > 0)
				out[cls+(col*rows)] /= y_count[cls];
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

__global__ void kAbs(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       out[i] = fabsf(A[i]);
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

//for column major data
__global__ void kAddScaledMatrixVector(float *A, float *v, float weight, float *out, int rows, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  //offset = current_column * rows
  int offset = 0;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  offset = (i / rows); //note: int arithmetic
	  out[i] =  A[i] + (v[offset]*weight);
  }
}

//for column major data
__global__ void kMulMatrixVector(float *A, float *v, float *out, int rows, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  //offset = current_column * rows
  int offset = 0;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  offset = (i / rows); //note: int arithmetic
	  out[i] =  A[i] * v[offset];
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

__global__ void kLinear(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
      out[i] = A[i];

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

__global__ void kDropout_cached(float *A, float *dropout, float *out, int current_idx, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	  int shifted_idx = 0;
	  int offset = 0;
	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  shifted_idx = i +current_idx;
		  offset = shifted_idx/10000;
		  out[i] = dropout[shifted_idx - (offset*10000)] == 1.0f ? A[i] : 0.0f;
	  }

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

__global__ void kRMSprop_with_momentum_update (float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;
	  float momentum_matrix_value = 0.0f;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = fdividef(grad[i],(float)batch_size);
		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
		  grad_value = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));
		  momentum_matrix_value = m[i];
		  momentum_matrix_value -= grad_value;

		  RMS[i] = RMS_value;
		  m[i] = momentum_matrix_value;
	  }
}




__global__ void kLocalGrad (float *z, float *w, float *y, float *m, float learning_rate, int batch_size, int size, float momentum)
{

}

__global__ void kRMSprop_with_momentum_weight_update (float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;
	  float momentum_matrix_value = 0.0f;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = fdividef(grad[i],(float)batch_size);
		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
		  grad_value = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));
		  momentum_matrix_value = m[i] = (momentum*momentum_matrix_value) - grad_value;

		  RMS[i] = RMS_value;
		  w[i] += momentum_matrix_value;

	  }
}

__global__ void kRMSprop_with_nesterov_weight_update (float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {

		  grad_value = fdividef(grad[i],(float)batch_size);
		  m[i] = (momentum*m[i]) - (learning_rate*grad_value);

		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
		  grad_value = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));

		  RMS[i] = RMS_value;
		  w[i] -= grad_value;

		  /*
		  grad_value = learning_rate*fdividef(grad[i],(float)batch_size);
		  m[i] = (momentum*m[i]) - grad_value;
		  w[i] -= grad_value;
			*/
	  }
}

__global__ void kNesterov_weight_update (float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = learning_rate*fdividef(grad[i],(float)batch_size);
		  m[i] = (momentum*m[i]) - grad_value;
		  w[i] -= grad_value;

	  }
}


__global__ void kCompression_8bit_test(float *tbl, float *A, float precision, int size, float *out)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float absnumber = 0.0;
	float multiplier = 0.1f/precision;
	float threshold = precision/1.e6f;

	__shared__ float tbl_values[128];
	if(threadIdx.x < 126)
		tbl_values[threadIdx.x] = tbl[threadIdx.x];

	__syncthreads();

	  for (int i = idx;i < size; i += numThreads)
	  {
		  int isNegative = 0;
		  int pivot = 63;
		  int upper_pivot = 125;
		  int lower_pivot = 0;
		  absnumber = A[i]*multiplier;
		  if(absnumber < 0.0f){isNegative = 1; absnumber=-absnumber; }
		  if(absnumber < threshold){ out[i] = 0.0f; continue; }
		  for(int j = 32; j > 0; j>>=1)
		  {
			  if(absnumber > tbl_values[pivot])
			  {
				  lower_pivot = pivot;
				  pivot+=j;
			  }
			  else
			  {
				  upper_pivot = pivot;
				  pivot-=j;
			  }

		  }

		  if(lower_pivot == pivot)
			  if(fabsf(tbl_values[pivot]-absnumber) < (tbl_values[upper_pivot]-absnumber))
				  out[i] = tbl_values[pivot]/(isNegative == 1 ? -multiplier : multiplier);
			  else
				  out[i] = tbl_values[upper_pivot]/(isNegative == 1 ? -multiplier : multiplier);
		  else
			  if((tbl_values[pivot]-absnumber) < fabsf(tbl_values[lower_pivot]-absnumber))
				  out[i] = tbl_values[pivot]/(isNegative == 1 ? -multiplier : multiplier);
			  else
				  out[i] = tbl_values[lower_pivot]/(isNegative == 1 ? -multiplier : multiplier);



	  }
}

__global__ void kDecompression_8bit(float *flt_tbl, unsigned char *A, float precision, int size, float *out)
{

	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	__shared__ float tbl_floats[256];
	if(threadIdx.x < 126)
	{
		tbl_floats[threadIdx.x] = flt_tbl[threadIdx.x]*precision;
		tbl_floats[threadIdx.x+128] = -tbl_floats[threadIdx.x];
	}


	tbl_floats[126] = 0.0f;
	tbl_floats[254] = -0.0f;
	tbl_floats[127] = precision;
	tbl_floats[255] = -precision;

	__syncthreads();

	for (int i = idx;i < size; i += numThreads)
	{
		out[i] = tbl_floats[A[i]];
	}
}


__global__ void kCompression_8bit(float *flt_tbl, float *A, float precision, int size, unsigned char *out)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float absnumber = 0.0f;
	float threshold_lower = 0.0000015;
	float threshold_upper = 0.995703;
	int isNegative = 0;
	int pivot = 63;
	int upper_pivot = 125;
	int lower_pivot = 0;

	__shared__ float tbl_floats[128];
	if(threadIdx.x < 126)
		tbl_floats[threadIdx.x] = flt_tbl[threadIdx.x];


	__syncthreads();

	  for (int i = idx;i < size; i += numThreads)
	  {
		  isNegative = 0;
		  pivot = 63;
		  upper_pivot = 125;
		  lower_pivot = 0;
		  absnumber = A[i]/precision;
		  if(absnumber < 0.0f){isNegative = 1; absnumber=-absnumber; }
		  if(absnumber < threshold_lower){ out[i] = (unsigned char)126; continue; }
		  if(absnumber > threshold_upper){ out[i] = (isNegative == 0 ? (unsigned char)127 : (unsigned char)255); continue; }
		  for(int j = 32; j > 0; j>>=1)
		  {
			  if(absnumber > tbl_floats[pivot])
			  {
				  lower_pivot = pivot;
				  pivot+=j;
			  }
			  else
			  {
				  upper_pivot = pivot;
				  pivot-=j;
			  }

		  }

		  if(lower_pivot == pivot)
			  if(fabsf(tbl_floats[pivot]-absnumber) < (tbl_floats[upper_pivot]-absnumber))
				  if(isNegative == 1)
					  out[i] =  pivot | 1 << 7;
				  else
					  out[i] =  pivot;
			  else
				  if(isNegative == 1)
					  out[i] =  upper_pivot | 1 << 7;
				  else
					  out[i] =  upper_pivot;
		  else
			  if((tbl_floats[pivot]-absnumber) < fabsf(tbl_floats[lower_pivot]-absnumber))
				  if(isNegative == 1)
					  out[i] =  (pivot | 1 << 7);
				  else
					  out[i] =  pivot;
			  else
		  	  	  if(isNegative == 1)
		  	  		  out[i] =  lower_pivot | 1 << 7;
		  		  else
		  			  out[i] =  lower_pivot;

	  }
}


__global__ void kRMSprop_with_weight_update (float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = fdividef(grad[i],(float)batch_size) ;
		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
		  grad_value = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));

		  RMS[i] = RMS_value;
		  w[i] -= grad_value;

	  }
}



__global__ void kRMSprop_with_weight_update_8bit(float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
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

	//middle index is replaced by rdm word for batch_Y, but we still need to write the correct word into batch_X!
	if(blockIdx.y != middleIdx)
	{
		myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
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


__global__ void concat_batches(float **batch_X, float **batch_Y, float *out_X, float *out_Y)
{
	//gridDim.z = matrix_count
	//gridDim.y = batch size
	//gridDim.x = window_size
	//blockDim.x = partial vocab size

	int full_vocab_size = gridDim.z*blockDim.x;
	int cols = gridDim.x*full_vocab_size;
	int partial_cols = blockDim.x*gridDim.x;

	//full_size times current row = current row idx
	//current window position times partial_threads times current matrix = current word idx
	//threadIdx.x current parameter within a word
	out_X[(blockIdx.y *cols) + (blockIdx.x*full_vocab_size) + (blockIdx.z*blockDim.x)  +threadIdx.x] = batch_X[blockIdx.z][(blockIdx.y *partial_cols) + (blockIdx.x*blockDim.x)  + threadIdx.x];
	out_Y[(blockIdx.y *cols) + (blockIdx.x*full_vocab_size) + (blockIdx.z*blockDim.x)  +threadIdx.x] = batch_Y[blockIdx.z][(blockIdx.y *partial_cols) + (blockIdx.x*blockDim.x)  + threadIdx.x];

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



//numerically unstable?
__global__ void kUpdateVocabWithGradient(float *grad, float *vocab_idx, float* vocab, float learning_rate)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	int myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myVocabIdx = blockDim.x*myIdx;
	atomicAdd(&vocab[myVocabIdx + threadIdx.x],-grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]*learning_rate);
}






__global__ void kExpandDoubleVocabGradient(float *gradX, float *gradY, float *vocab_idx_X, float *vocab_idx_Y, float* vocab,
										 float *vocab_grad, float *vocab_grad_idx, float learning_rate, int grad_size)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y


	//float multiplier = fdividef(learning_rate,(float)(gridDim.x*2));
	int myIdx_X = (int)vocab_idx_X[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myIdx_Y = (int)vocab_idx_Y[blockIdx.x+(blockIdx.y*gridDim.x)];
	//int grad_cols = grad_size/blockDim.x;

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


/*
__global__ void kExpandVocabGradient_sharedMemory(float *grad, float *vocab_idx, float *vocab_grad, float *sorted_vocab_idx, vocab_idx_size)
{
	//vocab_vector_size = blockDim.x;
	//batch_size = gridDim.x
	//try different configs for gridDim.x, e.g 16, 32 etc.

	//will have vocab_vector_size = blockDim.x elements e.g. 64
	extern __shared__ float sGrads[];

	float myWordIdx = 0.0f;
	float last_word = 0.0f;
	float currentIdx = 0.0f;

	sGrads[threadIdx.x] = 0.0f;

	for(int word = blockIdx.x; currentIdx < vocab_idx_size; word++)
	{
		for(int i = currentIdx; i < vocab_idx_size; i++, currentIdx++)
		{

		}
	}
}
*/


__global__ void kExpandVocabGradient(float *grad, float *vocab_idx, float *vocab_grad)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	int myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myVocabIdx = blockDim.x*myIdx;
	atomicAdd(&vocab_grad[myVocabIdx + threadIdx.x],grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]);

}

__global__ void kExpandPartialVocabGradient(float *grad, float *vocab_idx, float *vocab_grad, int matrix_idx, int matrix_count)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y
	int offset = matrix_idx*gridDim.x*blockDim.x;
	int myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myVocabIdx = blockDim.x*myIdx;
	atomicAdd(&vocab_grad[myVocabIdx + threadIdx.x],grad[blockIdx.x + (blockIdx.y*(blockDim.x*matrix_count)*gridDim.x) + (threadIdx.x*gridDim.x) + offset]);

}

__global__ void kExpandVocabGradientMiddleWord(float *grad, float *vocab_idx, float *vocab_grad)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	if(blockIdx.x+(blockIdx.y*gridDim.x) == gridDim.y/2)
	{
		int myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
		int myVocabIdx = blockDim.x*myIdx;
		atomicAdd(&vocab_grad[myVocabIdx + threadIdx.x],grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]);
	}

}




__global__ void kDot8bit(unsigned char *A, unsigned char *B, float *out, int rowsA, int colsA, int colsB, float *flt_tbl, float precisionA, float precisionB)
{
	const unsigned int threads_per_block = blockDim.x*blockDim.y;
	const int mygrid = blockIdx.x;
	const int myidx = (threadIdx.y*blockDim.x)+threadIdx.x;

	__shared__ float tbl_floatsA[256];
	__shared__ float tbl_floatsB[256];
	for(int i = myidx; i < 126; i++)
	{
		tbl_floatsA[i] = flt_tbl[i]*precisionA;
		tbl_floatsA[i+128] = -tbl_floatsA[i];
		tbl_floatsB[i] = flt_tbl[i]*precisionB;
		tbl_floatsB[i+128] = -tbl_floatsB[i];
	}
	tbl_floatsA[126] = 0.0f;
	tbl_floatsB[126] = 0.0f;
	tbl_floatsA[127] = precisionA;
	tbl_floatsB[127] = -precisionA;
	tbl_floatsA[254] = -0.0f;
	tbl_floatsB[254] = -0.0f;
	tbl_floatsA[255] = precisionB;
	tbl_floatsB[255] = -precisionB;

	__syncthreads();



	for(int Arow = mygrid; Arow < rowsA; Arow+=gridDim.x)
	{
		for(int Bcol = myidx; Bcol < colsB; Bcol+=threads_per_block)
		{
			int idxout = (Bcol*rowsA) + Arow;
			for(int Acol = 0; Acol < colsA; Acol++)
				out[idxout] += tbl_floatsA[A[(Acol*rowsA)+Arow]] * tbl_floatsB[B[(colsA*Bcol)  + Acol]];

		}


	}





}

__global__ void kDot8bit_shared(unsigned char *A, unsigned char *B, float *out, int rowsA, int colsA, int colsB, float *flt_tbl, float precisionA, float precisionB)
{
	int myidx = (threadIdx.y*blockDim.x)+threadIdx.x;

	__shared__ unsigned char A_tile[64][256]; //64x32 banks
	__shared__ unsigned char B_tile[64][256];//256x8 banks

	__shared__ float tbl_floatsA[256];
	__shared__ float tbl_floatsB[256];
	for(int i = myidx; i < 126; i++)
	{
		tbl_floatsA[i] = flt_tbl[i]*precisionA;
		tbl_floatsA[i+128] = -tbl_floatsA[i];
		tbl_floatsB[i] = flt_tbl[i]*precisionB;
		tbl_floatsB[i+128] = -tbl_floatsB[i];
	}
	tbl_floatsA[126] = 0.0f;
	tbl_floatsB[126] = 0.0f;
	tbl_floatsA[127] = precisionA;
	tbl_floatsB[127] = -precisionA;
	tbl_floatsA[254] = -0.0f;
	tbl_floatsB[254] = -0.0f;
	tbl_floatsA[255] = precisionB;
	tbl_floatsB[255] = -precisionB;

	__syncthreads();


	myidx = threadIdx.y*16;
	int Arow = threadIdx.x+(blockIdx.x*64);
	for(int i = 0; i < colsB; i++){ out[((i)*rowsA) + Arow] = 0.0f; }//zero output

	int Acol = (threadIdx.y*16)+(blockIdx.y*256);

		for(int i = 0; i < 16; i++)
			A_tile[threadIdx.x][myidx+i] = A[((Acol+i)*rowsA)+ Arow];



		int offset = 0;
		for(int Bcol = threadIdx.x ; Bcol < colsB; Bcol+=64)
		{

			for(int i = 0; i < 16; i++)
				B_tile[threadIdx.x][myidx+i] = B[(Bcol*colsA)+ Acol+i];//B_tile is transposed to avoid bank conflicts with 64 threads

			__syncthreads();

			for(int Bcol2 = offset; Bcol2 < 64 + offset; Bcol2++)
			{
				for (int i = 0; i < 16; ++i)
					atomicAdd(&out[((Bcol2)*rowsA) + Arow],tbl_floatsA[A_tile[threadIdx.x][myidx + i]] * tbl_floatsB[B_tile[Bcol2-offset][myidx + i]]);



			}
			offset +=64;
		}





}

__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)   As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
         else                                                   As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)   Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
         else                                                   Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}

static __device__ void saxpy(float alpha, const float*  b, float*        c )
{
    c[0]  += alpha * b[0];
    c[1]  += alpha * b[1];
    c[2]  += alpha * b[2];
    c[3]  += alpha * b[3];
    c[4]  += alpha * b[4];
    c[5]  += alpha * b[5];
    c[6]  += alpha * b[6];
    c[7]  += alpha * b[7];
    c[8]  += alpha * b[8];
    c[9]  += alpha * b[9];
    c[10] += alpha * b[10];
    c[11] += alpha * b[11];
    c[12] += alpha * b[12];
    c[13] += alpha * b[13];
    c[14] += alpha * b[14];
    c[15] += alpha * b[15];
}
__global__ void sgemm_kernel_N_N_64_16_16_16_4(float* C,const float* A,const float* B, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta )
{
    __shared__ float Bb[16][17];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int ibx = blockIdx.x * 64;
    int iby = blockIdx.y * 16;

    const int idt = ty * 16 + tx;

    /*
        Taking care of invalid memory access in dimension M
    */
    if ( ibx+idt >= m )
        A += ibx+0;
    else
        A += ibx + idt;

    C += ibx + idt + __mul24(iby, ldc);

    B += tx+__mul24(iby, ldb);

    /*
        These variables guide the threads to avoid invalid memory accesses
        in dimension N.
        Simply it's the stopping criterion.
        or you can say that access index wraps around to a valid memory location.
    */
    int s1=0, s2=4*ldb, s3=8*ldb, s4=12*ldb;

    if ( iby+ty    >= n ) { s1=1;  s2=0*ldb;  s3=0*ldb;  s4=0*ldb; } else
    if ( iby+ty+4  >= n ) { s1=0;  s2=0*ldb;  s3=0*ldb;  s4=0*ldb; } else
    if ( iby+ty+8  >= n ) { s1=0;  s2=4*ldb;  s3=0*ldb;  s4=0*ldb; } else
    if ( iby+ty+12 >= n ) { s1=0;  s2=4*ldb;  s3=8*ldb;  s4=0*ldb; }

    if ( s1 == 0 )
        B += __mul24(ty, ldb);
    else
        s1=0;

    const float *Bend = B + k - k % 16;

    float Cb[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    if ( k > 15 ) {
        do {
            float Ab[4] = {A[0], A[lda], A[2*lda], A[3*lda]};

            Bb[tx][ty+0 ] = B[s1];
            Bb[tx][ty+4 ] = B[s2];
            Bb[tx][ty+8 ] = B[s3];
            Bb[tx][ty+12] = B[s4];

            __syncthreads();

            A += 4 * lda;
            saxpy( Ab[0], &Bb[0][0], Cb );  Ab[0] = A[0*lda];
            saxpy( Ab[1], &Bb[1][0], Cb );  Ab[1] = A[1*lda];
            saxpy( Ab[2], &Bb[2][0], Cb );  Ab[2] = A[2*lda];
            saxpy( Ab[3], &Bb[3][0], Cb );  Ab[3] = A[3*lda];

            A += 4 * lda;
            saxpy( Ab[0], &Bb[4][0], Cb );  Ab[0] = A[0*lda];
            saxpy( Ab[1], &Bb[5][0], Cb );  Ab[1] = A[1*lda];
            saxpy( Ab[2], &Bb[6][0], Cb );  Ab[2] = A[2*lda];
            saxpy( Ab[3], &Bb[7][0], Cb );  Ab[3] = A[3*lda];

            A += 4 * lda;
            saxpy( Ab[0], &Bb[8][0],  Cb );  Ab[0] = A[0*lda];
            saxpy( Ab[1], &Bb[9][0],  Cb );  Ab[1] = A[1*lda];
            saxpy( Ab[2], &Bb[10][0], Cb );  Ab[2] = A[2*lda];
            saxpy( Ab[3], &Bb[11][0], Cb );  Ab[3] = A[3*lda];

            A += 4 * lda;
            saxpy( Ab[0], &Bb[12][0], Cb );
            saxpy( Ab[1], &Bb[13][0], Cb );
            saxpy( Ab[2], &Bb[14][0], Cb );
            saxpy( Ab[3], &Bb[15][0], Cb );

            B += 16;

            __syncthreads();
        } while (B < Bend);
    }

    /*
        Common sub expression elimination.
    */
    ibx = ibx + idt - m;

    /*
        remembering k dimension
    */
    ldb = m = k;

    /*
        k changed to support the generic case and reuse valuable registers
    */
    k = k % 16;

    m -= k;

    /*
        Here we are taking care of k % dim_k portions
    */
    if ( k != 0 ) {
        /*
            Avoid Invalid Memory access in dimension K
            If some thread enters this if ( ) block first access to B
            should be valid as K isn't divisible by blk_K
            Note that dimension N has been taken care of by s1, s2, s3, s4
            But depending upon K and thread index tx, some memory access
            may be still invalid, so take care of them now by setting
            s1, s2, s3, s4 = 0
            B might have been advanced in the previous loop, take care
            of that, this is about right bottom corner.
        */
        if ( m + tx >= ldb ) {
            s1 = s2 = s3 = s4 = 0;
            B -= tx;
        }

        Bb[tx][ty+0 ] = B[s1];
        Bb[tx][ty+4 ] = B[s2];
        Bb[tx][ty+8 ] = B[s3];
        Bb[tx][ty+12] = B[s4];
        __syncthreads();

        for(int i=0; i < k; i++) {
            saxpy( A[0], &Bb[i+0][0], Cb );
            A += lda;
        }
    }

    /*
        Now taking care of dimension M, N that doesnt fit into blocks
    */
    if ( (iby+16) >= n ) {
        lda = n - iby;
    }
    else {
        lda = 16;
    }
    if ( ibx >= 0 )
        lda = 0;
    else
        lda = lda;

    switch(lda) {
        case 16:
                C[ 0    ] = alpha * Cb[ 0] + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[ 1] + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[ 2] + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[ 3] + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[ 4] + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[ 5] + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[ 6] + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[ 7] + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[ 8] + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[ 9] + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                C[13*ldc] = alpha * Cb[13] + beta * C[13*ldc];
                C[14*ldc] = alpha * Cb[14] + beta * C[14*ldc];
                C[15*ldc] = alpha * Cb[15] + beta * C[15*ldc];
                break;
        case 15:
                C[ 0    ] = alpha * Cb[ 0] + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[ 1] + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[ 2] + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[ 3] + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[ 4] + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[ 5] + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[ 6] + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[ 7] + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[ 8] + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[ 9] + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                C[13*ldc] = alpha * Cb[13] + beta * C[13*ldc];
                C[14*ldc] = alpha * Cb[14] + beta * C[14*ldc];
                break;
        case 14:
                C[ 0    ] = alpha * Cb[ 0] + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[ 1] + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[ 2] + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[ 3] + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[ 4] + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[ 5] + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[ 6] + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[ 7] + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[ 8] + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[ 9] + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                C[13*ldc] = alpha * Cb[13] + beta * C[13*ldc];
                break;
        case 13:
                C[ 0    ] = alpha * Cb[ 0] + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[ 1] + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[ 2] + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[ 3] + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[ 4] + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[ 5] + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[ 6] + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[ 7] + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[ 8] + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[ 9] + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                break;
        case 12:
                C[ 0    ] = alpha * Cb[ 0] + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[ 1] + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[ 2] + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[ 3] + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[ 4] + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[ 5] + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[ 6] + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[ 7] + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[ 8] + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[ 9] + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                break;
        case 11:
                C[ 0    ] = alpha * Cb[ 0] + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[ 1] + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[ 2] + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[ 3] + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[ 4] + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[ 5] + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[ 6] + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[ 7] + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[ 8] + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[ 9] + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                break;
        case 10:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                C[7*ldc] = alpha * Cb[7] + beta * C[7*ldc];
                C[8*ldc] = alpha * Cb[8] + beta * C[8*ldc];
                C[9*ldc] = alpha * Cb[9] + beta * C[9*ldc];
                break;
        case 9:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                C[7*ldc] = alpha * Cb[7] + beta * C[7*ldc];
                C[8*ldc] = alpha * Cb[8] + beta * C[8*ldc];
                break;
        case 8:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                C[7*ldc] = alpha * Cb[7] + beta * C[7*ldc];
                break;
        case 7:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                break;
        case 6:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                break;
        case 5:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                break;
        case 4:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                break;
        case 3:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                break;
        case 2:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                break;
        case 1:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                break;
        case 0:
                break;
    }
}

__global__ void sgemmNN( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * 64;
    const int iby = blockIdx.y * 16;
    const int id = inx + iny*16;

    A += ibx + id;
    B += inx + __mul24( iby + iny, ldb );
    C += ibx + id  + __mul24( iby, ldc );

    const float *Blast = B + k;

    float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    __shared__ float bs[16][17];
    do
    {
#pragma unroll
        for( int i = 0; i < 16; i += 4 )
            bs[inx][iny+i]  = B[i*ldb];
        __syncthreads();

#pragma unroll
        for( int i = 0; i < 16; i++, A += lda )
            saxpy( A[0], &bs[i][0], c );

        B += 16;
        __syncthreads();
    } while( B < Blast );

    for( int i = 0; i < 16; i++, C += ldc )
        C[0] = alpha*c[i] + beta*C[0];
}

__global__ void sgemm_kernel_N_T_64_16_4_16_4(float* C, const float* A, const float* B, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta )
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int ibx = blockIdx.x * 64;
    const int iby = blockIdx.y * 16;

    const int idt = ty * 16 + tx;

    if ( iby + tx >= n )
        B += iby + 0;
    else
        B += iby + tx;
    /*
        Taking care of boundary cases where K < 4.
    */
    if ( ty >= k )
        B += __mul24( 0, ldb );
    else
        B += __mul24( ty, ldb );

    if ( ibx + idt >= m )
        A += ibx + 0;
    else
        A += ibx + idt;

    int s2=lda, s3=2*lda, s4=3*lda;

    switch (k) {
        case 1: s2=0;    s3=0;      s4=0;  break;
        case 2: s2=lda;  s3=0;      s4=0;  break;
        case 3: s2=lda;  s3=2*lda;  s4=0;  break;
    }

    C += ibx + idt + __mul24( iby, ldc );

    float Ap[4] = { A[0], A[s2], A[s3], A[s4] };

    float b = B[0];

    const float *Bend = B + ldb*(k - k % 4);

    B += 4*ldb;
    A += 4*lda;

    __shared__ float Bb[4][16];

    float Cb[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    if ( k > 7 ) {
        do {
            float Ab[4] = {Ap[0], Ap[1], Ap[2], Ap[3]};

            Bb[ty][tx]=b;

            __syncthreads();

            Ap[0] = A[0];
            Ap[1] = A[s2];
            Ap[2] = A[s3];
            Ap[3] = A[s4];

            b=B[0];

            saxpy( Ab[0], &Bb[0][0], Cb );
            saxpy( Ab[1], &Bb[1][0], Cb );
            saxpy( Ab[2], &Bb[2][0], Cb );
            saxpy( Ab[3], &Bb[3][0], Cb );

            A += 4*lda;
            B += 4*ldb;

            __syncthreads();
        } while (B < Bend);
    }

    if ( k > 3 ) {
        Bb[ty][tx]=b;
        int k1 = k - k % 4;

        if ( (k1+ty) >= k )
            B -= 4*ldb;
        else
            B -= 0*ldb;

        if ( (k1+0) >= k ) {s2=0;    s3=0*lda;  s4=0;  A -= 4*lda; } else
        if ( (k1+1) >= k ) {s2=0;    s3=0*lda;  s4=0;  A -= 0*lda; } else
        if ( (k1+2) >= k ) {s2=lda;  s3=0*lda;  s4=0;  A -= 0*lda; } else
        if ( (k1+3) >= k ) {s2=lda;  s3=2*lda;  s4=0;  A -= 0*lda; }

        __syncthreads();

        b=B[0];

        saxpy( Ap[0], &Bb[0][0], Cb );  Ap[0] = A[0];
        saxpy( Ap[1], &Bb[1][0], Cb );  Ap[1] = A[s2];
        saxpy( Ap[2], &Bb[2][0], Cb );  Ap[2] = A[s3];
        saxpy( Ap[3], &Bb[3][0], Cb );  Ap[3] = A[s4];
    }

    k = k % 4;

    if ( k != 0 ) {
        __syncthreads();

        Bb[ty][tx]=b;

        __syncthreads();

        for(int i=0; i < k; i++) {
            saxpy( Ap[i], &Bb[i][0], Cb );
        }
    }

    if ( (iby+16)>=n) {
        lda = n-iby;
    }
    else{
        lda = 16;
    }

    if ( (ibx+idt) >= m )
        lda = 0;
    else
        lda = lda;

    switch(lda) {
        case 16:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                C[13*ldc] = alpha * Cb[13] + beta * C[13*ldc];
                C[14*ldc] = alpha * Cb[14] + beta * C[14*ldc];
                C[15*ldc] = alpha * Cb[15] + beta * C[15*ldc];
                break;
        case 15:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                C[13*ldc] = alpha * Cb[13] + beta * C[13*ldc];
                C[14*ldc] = alpha * Cb[14] + beta * C[14*ldc];
                break;
        case 14:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                C[13*ldc] = alpha * Cb[13] + beta * C[13*ldc];
                break;
        case 13:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                break;
        case 12:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                break;
        case 11:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                break;
        case 10:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                C[7*ldc] = alpha * Cb[7] + beta * C[7*ldc];
                C[8*ldc] = alpha * Cb[8] + beta * C[8*ldc];
                C[9*ldc] = alpha * Cb[9] + beta * C[9*ldc];
                break;
        case 9:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                C[7*ldc] = alpha * Cb[7] + beta * C[7*ldc];
                C[8*ldc] = alpha * Cb[8] + beta * C[8*ldc];
                break;
        case 8:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                C[7*ldc] = alpha * Cb[7] + beta * C[7*ldc];
                break;
        case 7:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                break;
        case 6:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                break;
        case 5:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                break;
        case 4:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                break;
        case 3:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                break;
        case 2:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                break;
        case 1:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                break;
        case 0:
                break;
    }
}

__global__ void sgemm_kernel_T_N_32_32_8_8_8(float* C, const float* A, const float* B, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta )
{
    const int ibx = blockIdx.x * 32;
    const int iby = blockIdx.y * 32;

    const int tx =  threadIdx.y;
    const int ty =  threadIdx.x;

    int idt = tx*8 + ty;

    if ( ty >= k )
        A += __mul24(ibx, lda) + 0;
    else
        A += __mul24(ibx, lda) + ty;

    if ( (ibx + tx) >= m )
        A += __mul24(0, lda);
    else
        A += __mul24(tx, lda);

    if ( (iby+tx) >= n )
        B += __mul24(iby+0, ldb);
    else
        B += __mul24(iby+tx, ldb);
    if ( ty >= k )
        B += 0;
    else
        B += ty;

    C += ibx + idt % 32 + __mul24( iby + 16*(idt/32), ldc );

    lda = lda * 8;
    ldb = ldb * 8;

    int as1=0, as2=lda, as3=2*lda, as4=3*lda;
    int bs1=0, bs2=ldb, bs3=2*ldb, bs4=3*ldb;

    switch(k) {
        case 1: as2=0;    as3=0*lda;  as4=0;  bs2=0;    bs3=0*ldb;  bs4=0;  break;
        case 2: as2=lda;  as3=0*lda;  as4=0;  bs2=ldb;  bs3=0*ldb;  bs4=0;  break;
        case 3: as2=lda;  as3=2*lda;  as4=0;  bs2=ldb;  bs3=2*ldb;  bs4=0;  break;
    }

    if ( (ibx + tx     ) >= m ) { as1=0;  as2=0*lda;  as3=0*lda;  as4=0*lda; } else
    if ( (ibx + tx + 8 ) >= m ) { as1=0;  as2=0*lda;  as3=0*lda;  as4=0*lda; } else
    if ( (ibx + tx + 16) >= m ) { as1=0;  as2=1*lda;  as3=0*lda;  as4=0*lda; } else
    if ( (ibx + tx + 24) >= m ) { as1=0;  as2=1*lda;  as3=2*lda;  as4=0*lda; }

    if ( (iby + tx     ) >= n ) { bs1=0;  bs2=0*ldb;  bs3=0*ldb;  bs4=0*ldb; } else
    if ( (iby + tx + 8 ) >= n ) { bs1=0;  bs2=0*ldb;  bs3=0*ldb;  bs4=0*ldb; } else
    if ( (iby + tx + 16) >= n ) { bs1=0;  bs2=1*ldb;  bs3=0*ldb;  bs4=0*ldb; } else
    if ( (iby + tx + 24) >= n ) { bs1=0;  bs2=1*ldb;  bs3=2*ldb;  bs4=0*ldb; }

    float b  = B[bs1];
    float b1 = B[bs2];
    float b2 = B[bs3];
    float b3 = B[bs4];

    float Ap[4] = { A[as1], A[as2], A[as3], A[as4] };

    const float *Bend = B + (k - k % 8);

    B += 8;
    A += 8;

    __shared__ float Bb[8][33];
    __shared__ float ABb[32][9];

    float Cb[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const int l = 17*(idt/32);
    int idt1 = idt;
    idt = idt % 32;
    if ( k > 15 ) {
        do {
            Bb[ty][tx   ] = b;
            Bb[ty][tx+8 ] = b1;
            Bb[ty][tx+17] = b2;
            Bb[ty][tx+25] = b3;

            ABb[tx   ][ty] = Ap[0];
            ABb[tx+8 ][ty] = Ap[1];
            ABb[tx+16][ty] = Ap[2];
            ABb[tx+24][ty] = Ap[3];

            __syncthreads();

            saxpy( ABb[idt][0], &Bb[0][l], Cb );  Ap[0]=A[as1];
            saxpy( ABb[idt][1], &Bb[1][l], Cb );  Ap[1]=A[as2];
            saxpy( ABb[idt][2], &Bb[2][l], Cb );  Ap[2]=A[as3];
            saxpy( ABb[idt][3], &Bb[3][l], Cb );  Ap[3]=A[as4];

            saxpy( ABb[idt][4], &Bb[4][l], Cb );  b=B[bs1];
            saxpy( ABb[idt][5], &Bb[5][l], Cb );  b1=B[bs2];
            saxpy( ABb[idt][6], &Bb[6][l], Cb );  b2=B[bs3];
            saxpy( ABb[idt][7], &Bb[7][l], Cb );  b3=B[bs4];

            B += 8;
            A += 8;

            __syncthreads();
        } while (B < Bend);
    }
    if ( k > 7 ) {
        Bb[ty][tx   ] = b;
        Bb[ty][tx+8 ] = b1;
        Bb[ty][tx+17] = b2;
        Bb[ty][tx+25] = b3;

        ABb[tx   ][ty] = Ap[0];
        ABb[tx+8 ][ty] = Ap[1];
        ABb[tx+16][ty] = Ap[2];
        ABb[tx+24][ty] = Ap[3];

        __syncthreads();
        as1 = k - k % 8;

        if ( as1+ty >= k ) { bs1=0*ldb;  bs2=0*ldb;  bs3=0*ldb;  bs4=0*ldb;  B -= 8; }
        if ( as1+ty >= k ) { as1=0*lda;  as2=0*lda;  as3=0*lda;  as4=0*lda;  A -= 8; }

        as1=0;
        saxpy( ABb[idt][0], &Bb[0][l], Cb );  Ap[0]=A[as1];
        saxpy( ABb[idt][1], &Bb[1][l], Cb );  Ap[1]=A[as2];
        saxpy( ABb[idt][2], &Bb[2][l], Cb );  Ap[2]=A[as3];
        saxpy( ABb[idt][3], &Bb[3][l], Cb );  Ap[3]=A[as4];

        saxpy( ABb[idt][4], &Bb[4][l], Cb );  b=B[bs1];
        saxpy( ABb[idt][5], &Bb[5][l], Cb );  b1=B[bs2];
        saxpy( ABb[idt][6], &Bb[6][l], Cb );  b2=B[bs3];
        saxpy( ABb[idt][7], &Bb[7][l], Cb );  b3=B[bs4];
    }
    k = k % 8;
    if ( k != 0 ) {
        __syncthreads();

        Bb[ty][tx   ] = b;
        Bb[ty][tx+8 ] = b1;
        Bb[ty][tx+17] = b2;
        Bb[ty][tx+25] = b3;

        ABb[tx   ][ty] = Ap[0];
        ABb[tx+8 ][ty] = Ap[1];
        ABb[tx+16][ty] = Ap[2];
        ABb[tx+24][ty] = Ap[3];
        __syncthreads();

        for(int i=0; i < k; i++) {
            saxpy( ABb[idt][i], &Bb[i][l], Cb );
        }
    }

    if ( (iby+16*(idt1/32+1)) >= n ) {
        lda = n - iby - 16*(idt1/32);
    }
    else {
        lda = 16;
    }
    if ( (ibx+idt) >= m )
        lda = 0;
    else
        lda = lda;

    switch(lda) {
        case 16:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                C[13*ldc] = alpha * Cb[13] + beta * C[13*ldc];
                C[14*ldc] = alpha * Cb[14] + beta * C[14*ldc];
                C[15*ldc] = alpha * Cb[15] + beta * C[15*ldc];
                break;
        case 15:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                C[13*ldc] = alpha * Cb[13] + beta * C[13*ldc];
                C[14*ldc] = alpha * Cb[14] + beta * C[14*ldc];
                break;
        case 14:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                C[13*ldc] = alpha * Cb[13] + beta * C[13*ldc];
                break;
        case 13:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                C[12*ldc] = alpha * Cb[12] + beta * C[12*ldc];
                break;
        case 12:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                C[11*ldc] = alpha * Cb[11] + beta * C[11*ldc];
                break;
        case 11:
                C[ 0    ] = alpha * Cb[0]  + beta * C[ 0    ];
                C[ 1*ldc] = alpha * Cb[1]  + beta * C[ 1*ldc];
                C[ 2*ldc] = alpha * Cb[2]  + beta * C[ 2*ldc];
                C[ 3*ldc] = alpha * Cb[3]  + beta * C[ 3*ldc];
                C[ 4*ldc] = alpha * Cb[4]  + beta * C[ 4*ldc];
                C[ 5*ldc] = alpha * Cb[5]  + beta * C[ 5*ldc];
                C[ 6*ldc] = alpha * Cb[6]  + beta * C[ 6*ldc];
                C[ 7*ldc] = alpha * Cb[7]  + beta * C[ 7*ldc];
                C[ 8*ldc] = alpha * Cb[8]  + beta * C[ 8*ldc];
                C[ 9*ldc] = alpha * Cb[9]  + beta * C[ 9*ldc];
                C[10*ldc] = alpha * Cb[10] + beta * C[10*ldc];
                break;
        case 10:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                C[7*ldc] = alpha * Cb[7] + beta * C[7*ldc];
                C[8*ldc] = alpha * Cb[8] + beta * C[8*ldc];
                C[9*ldc] = alpha * Cb[9] + beta * C[9*ldc];
                break;
        case 9:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                C[7*ldc] = alpha * Cb[7] + beta * C[7*ldc];
                C[8*ldc] = alpha * Cb[8] + beta * C[8*ldc];
                break;
        case 8:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                C[7*ldc] = alpha * Cb[7] + beta * C[7*ldc];
                break;
        case 7:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                C[6*ldc] = alpha * Cb[6] + beta * C[6*ldc];
                break;
        case 6:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                C[5*ldc] = alpha * Cb[5] + beta * C[5*ldc];
                break;
        case 5:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                C[4*ldc] = alpha * Cb[4] + beta * C[4*ldc];
                break;
        case 4:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                C[3*ldc] = alpha * Cb[3] + beta * C[3*ldc];
                break;
        case 3:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                C[2*ldc] = alpha * Cb[2] + beta * C[2*ldc];
                break;
        case 2:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                C[1*ldc] = alpha * Cb[1] + beta * C[1*ldc];
                break;
        case 1:
                C[0    ] = alpha * Cb[0] + beta * C[0    ];
                break;
        case 0:
                break;
    }
}
