#include <basicOps.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
const int NUM_THREADS = 32;

__global__ void kFill_with(float *m, float fill_value, int size)
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



