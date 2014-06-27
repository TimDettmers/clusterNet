#include <stdio.h>
#include <basicOps.cuh>
#include <clusterKernels.cuh>
#include <assert.h>
#include <util.cuh>
#include <cublas_v2.h>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

Matrix *to_gpu(Matrix *A){ return to_gpu(A, 0); }
Matrix *to_gpu(Matrix *A, int is_col_major)
{
  float * gpu_data;
  cudaMalloc((void**)&gpu_data,A->bytes);
  cudaMemcpy(gpu_data,A->data,A->bytes,cudaMemcpyDefault);
  Matrix *out = (Matrix*)malloc(sizeof(Matrix));
  out->rows = A->rows;
  out->cols = A->cols;
  out->bytes = A->bytes;
  out->size = A->size;
  out->data = gpu_data;
  out->isDistributed = 0;
  out->cols_distributed = 0;

  if(is_col_major == 0)
	  out = to_col_major(out);

  return out;
}

Matrix *to_host(Matrix *A){ return to_host(A, 0); }
Matrix *to_host(Matrix *A, int is_row_major)
{
	Matrix *row_major;
	 if(is_row_major == 0 && A->isSparse == 0)
		 row_major = to_row_major(A);
	 else
		 row_major = A;

  Matrix *out = (Matrix*)malloc(sizeof(Matrix));
  float *cpu_data;
  if(row_major->isSparse != 1)
  {
	  cpu_data = (float*)malloc(row_major->bytes);
	  cudaMemcpy(cpu_data,row_major->data,row_major->bytes,cudaMemcpyDefault);
	  out->rows = row_major->rows;
	  out->cols = row_major->cols;
	  out->bytes = row_major->bytes;
	  out->size = row_major->size;
	  out->data = cpu_data;
	  out->isDistributed = 0;
	  out->cols_distributed = 0;
	  out->isSparse = 0;
	  out->ptr_bytes = 0;
	  out->idx_bytes = 0;
  }
  else
  {
	  cpu_data = (float*)malloc(row_major->bytes);
	  int *idx_cols = (int*)malloc(sizeof(int)*row_major->size);
	  int *ptr_rows = (int*)malloc(sizeof(int)*(row_major->rows+1));
	  cudaMemcpy(cpu_data,row_major->data,row_major->bytes,cudaMemcpyDefault);
	  cudaMemcpy(idx_cols,row_major->idx_cols,row_major->idx_bytes,cudaMemcpyDefault);
	  cudaMemcpy(ptr_rows,row_major->ptr_rows,row_major->ptr_bytes,cudaMemcpyDefault);
	  out->rows = row_major->rows;
	  out->cols = row_major->cols;
	  out->bytes = row_major->bytes;
	  out->size = row_major->size;
	  out->data = cpu_data;
	  out->isDistributed = 0;
	  out->cols_distributed = 0;
	  out->isSparse = 1;

	  out->idx_bytes = sizeof(int)*row_major->size;
	  out->ptr_bytes = sizeof(int)*(row_major->rows+1);
	  out->idx_cols = idx_cols;
	  out->ptr_rows = ptr_rows;
  }

	 if(is_row_major == 0 && A->isSparse == 0)
		 cudaFree(row_major->data);

  return out;
}


static inline void T(Matrix *A, Matrix *out, int rows, int cols)
{
  // setup execution parameters
  int grid_x = rows / COPY_BLOCK_SIZE;
  if (rows % COPY_BLOCK_SIZE)
    grid_x++;

  int grid_y = cols / COPY_BLOCK_SIZE;
  if (cols % COPY_BLOCK_SIZE)
    grid_y++;

  dim3 grid(grid_x, grid_y, 1);
  dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);
  kTranspose<<< grid, threads >>>(A->data, out->data, rows, cols);

}

Matrix *to_col_major(Matrix *A)
{
  Matrix *out = empty(A->rows,A->cols);
  T(A, out, A->cols,A->rows);
  //cudaFree(A->data);
  return out;
}

void to_col_major(Matrix *A, Matrix *out)
{
  T(A, out, A->cols,A->rows);
}

Matrix *to_row_major(Matrix *A)
{
  Matrix *out = empty(A->rows,A->cols);
  T(A, out, A->rows,A->cols);

  return out;
}

Matrix *T(Matrix *A)
{
  Matrix *out = empty(A->cols,A->rows);
  T(A, out, A->rows,A->cols);

  out->rows = A->cols;
  out->cols = A->rows;
  return out;
}




Matrix *slice_rows(Matrix *A, int start, int end)
{
  //align memory in contiguous array

  Matrix *out = empty((end - start) + 1, A->cols);
  int block_size = (out->size/THREADS_PER_BLOCKS) + 1;
  slice_rows<<<block_size,THREADS_PER_BLOCKS>>>(A->data, out->data, out->size, A->rows, start, end);

  return out;
}

Matrix *slice_cols(Matrix *A, int start, int end)
{
  Matrix *out = empty(A->rows, end - start + 1);
  int block_size = (out->size/THREADS_PER_BLOCKS) + 1;
  slice_cols<<<block_size,THREADS_PER_BLOCKS>>>(A->data, out->data, start, A->rows, out->size);

  return out;
}

Matrix *zeros(int rows, int cols)
{
  return fill_matrix(rows, cols, 0.0f);
}

Matrix *ones(int rows, int cols)
{
  return fill_matrix(rows, cols, 1.0f);
}

void rand_int(Matrix *uniform_rdm,int low, int high)
{
	int block_size = (uniform_rdm->size/THREADS_PER_BLOCKS) + 1;
	kRandInt<<<block_size,THREADS_PER_BLOCKS>>>(uniform_rdm->data, low, high, uniform_rdm->size);
}

void uniformSqrtWeight(Matrix * uniform_rdm)
{
	int block_size = (uniform_rdm->size/THREADS_PER_BLOCKS) + 1;
	kCreateRdmSqrtWeight_Logistic<<<block_size,THREADS_PER_BLOCKS>>>(uniform_rdm->data, uniform_rdm->rows, uniform_rdm->cols,uniform_rdm->size);
}

void uniformSqrtWeight(Matrix * uniform_rdm, int in, int out)
{
	int block_size = (uniform_rdm->size/THREADS_PER_BLOCKS) + 1;
	kCreateRdmSqrtWeight_Logistic<<<block_size,THREADS_PER_BLOCKS>>>(uniform_rdm->data, in, out, uniform_rdm->size);
}

Matrix *arange(int rows, int cols){	return arange(0, rows, cols); }
Matrix *arange(int start, int rows, int cols)
{
	Matrix *out = empty(rows, cols);
	int block_size = (out->size/THREADS_PER_BLOCKS) + 1;
	kArange<<<block_size,THREADS_PER_BLOCKS>>>(out->data, start, out->rows, out->cols, out->size);
	return out;
}

Matrix *empty(int rows, int cols)
{
  float *gpu_data;
  int size = rows*cols;
  size_t bytes = rows*cols*sizeof(float);
  cudaMalloc((void**)&gpu_data, bytes);
  
  Matrix *out = (Matrix*)malloc(sizeof(Matrix));
  out->rows = rows;
  out->cols = cols;
  out->bytes = bytes;
  out->size = size;
  out->data = gpu_data;
  out->isDistributed = 0;
  out->cols_distributed = 0;
  out->isSparse = 0;
  out->idx_bytes = 0;
  out->idx_cols = 0;
  out->ptr_bytes = 0;
  out->ptr_rows = 0;


  return out;
}

Matrix *empty_sparse(int rows, int cols, float max_sparsity, float sparsity_buffer)
{ return empty_sparse(rows, cols, ceil(rows*cols*(max_sparsity + sparsity_buffer)) + 1); }
Matrix *empty_sparse(int rows, int cols, int nonzeros)
{
	int elements = nonzeros;
	float *data;
	int *idx_cols;
	int *ptr_rows;
	size_t bytes = elements*sizeof(float);
	size_t idx_bytes = elements*sizeof(int);
	size_t ptr_bytes = (rows+1)*sizeof(int);
	cudaMalloc((void**)&data, bytes);
	cudaMalloc((void**)&idx_cols, idx_bytes);
	cudaMalloc((void**)&ptr_rows, ptr_bytes);

	kFill_with<<<(elements/THREADS_PER_BLOCKS) + 1, THREADS_PER_BLOCKS>>>(data,0.0f,elements);
	kFill_with<<<(elements/THREADS_PER_BLOCKS) + 1, THREADS_PER_BLOCKS>>>(idx_cols,0,elements);
	kFill_with<<<(rows/THREADS_PER_BLOCKS) + 1, THREADS_PER_BLOCKS>>>(ptr_rows,0,rows + 1);

	Matrix *out = (Matrix*)malloc(sizeof(Matrix));
	out->rows = rows;
	out->cols = cols;
	out->bytes = bytes;
	out->size = elements;
	out->data = data;
	out->isDistributed = 0;
	out->cols_distributed = 0;
	out->isSparse = 1;
	out->idx_bytes = idx_bytes;
	out->idx_cols = idx_cols;
	out->ptr_bytes = ptr_bytes;
	out->ptr_rows = ptr_rows;

	return out;

}




Matrix *empty_pinned_sparse(int rows, int cols, float max_sparsity, float sparsity_buffer)
{ return empty_pinned_sparse(rows, cols, ceil(rows*cols*(max_sparsity + sparsity_buffer)) + 1); }
Matrix *empty_pinned_sparse(int rows, int cols, int nonzeros)
{
	int elements = nonzeros;
	float *data;
	int *idx_cols;
	int *ptr_rows;
	int size = elements;
	size_t bytes = elements*sizeof(float);
	size_t idx_bytes = elements*sizeof(int);
	size_t ptr_bytes = (rows+1)*sizeof(int);
	cudaHostAlloc(&data, bytes, cudaHostAllocPortable);
	cudaHostAlloc(&idx_cols, idx_bytes, cudaHostAllocPortable);
	cudaHostAlloc(&ptr_rows, ptr_bytes, cudaHostAllocPortable);

	for(int i = 0; i < elements; i++)
	{
		data[i] = 0.0f;
		idx_cols[i] = 0.0f;
	}
	for(int i = 0; i < rows +1; i++)
		ptr_rows[i] = 0.0f;

	Matrix *out = (Matrix*)malloc(sizeof(Matrix));
	out->rows = rows;
	out->cols = cols;
	out->bytes = bytes;
	out->size = size;
	out->data = data;
	out->isDistributed = 0;
	out->cols_distributed = 0;
	out->isSparse = 1;
	out->idx_bytes = idx_bytes;
	out->idx_cols = idx_cols;
	out->ptr_bytes = ptr_bytes;
	out->ptr_rows = ptr_rows;

	return out;
}

Matrix *empty_pinned(int rows, int cols)
{
  float *pinned_data;
  int size = rows*cols;
  size_t bytes = rows*cols*sizeof(float);
  cudaHostAlloc(&pinned_data, bytes, cudaHostAllocPortable);

  Matrix *out = (Matrix*)malloc(sizeof(Matrix));
  out->rows = rows;
  out->cols = cols;
  out->bytes = bytes;
  out->size = size;
  out->data = pinned_data;
  out->isDistributed = 0;
  out->cols_distributed = 0;
  out->isSparse = 0;
  out->idx_bytes = 0;
  out->idx_cols = 0;
  out->ptr_bytes = 0;
  out->ptr_rows = 0;

  return out;
}

Matrix *empty_cpu(int rows, int cols)
{

  int size = rows*cols;
  size_t bytes = rows*cols*sizeof(float);
  float *data = (float*)malloc(bytes);

  Matrix *out = (Matrix*)malloc(sizeof(Matrix));
  out->rows = rows;
  out->cols = cols;
  out->bytes = bytes;
  out->size = size;
  out->data = data;
  out->isDistributed = 0;
  out->cols_distributed = 0;
  out->isSparse = 0;
  out->idx_bytes = 0;
  out->idx_cols = 0;
  out->ptr_bytes = 0;
  out->ptr_rows = 0;

  return out;
}


Matrix *fill_matrix(int rows, int cols, float fill_value)
{
  if(rows < 1 || cols < 1)
  {
    printf("Error: Dimensions must be greater than zero!\n");
    assert(0);  
  }
 
  Matrix *out = empty(rows, cols);
  
  thrust::device_ptr<float> ptr(out->data);
  thrust::fill(ptr, ptr + out->size,fill_value);
 
  return out;
}

void fill_matrix(Matrix *A, const float fill_value)
{
	thrust::device_ptr<float> ptr(A->data);
	thrust::fill(ptr, ptr + A->size,fill_value);
}

void fill_gpuarray(float *A, const float fill_value, int size)
{

	thrust::device_ptr<float> ptr(A);
	thrust::fill(ptr, ptr + size,fill_value);
}
void fill_gpuarray(int *A, const int fill_value, int size)
{
	thrust::device_ptr<int> ptr(A);
	thrust::fill(ptr, ptr + size,fill_value);
}

void fill_sparse_with_zeros(Matrix *A)
{
	assert(A->isSparse == 1);
	fill_matrix(A,0.0f);
	thrust::device_ptr<int> ptr_idx(A->idx_cols);
	thrust::fill(ptr_idx, ptr_idx + A->size,0);
	thrust::device_ptr<int> ptr_ptr(A->ptr_rows);
	thrust::fill(ptr_ptr, ptr_ptr + A->rows + 1,0);
}


Matrix *add(Matrix *A, Matrix *B)
{
  Matrix *out = empty(A->rows,A->cols);
  add(A, B, out);

  return out;
}

void add(Matrix *A, Matrix *B, Matrix *out)
{
  checkMatrixOperation(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N, 0);
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kAdd<<<block_size,THREADS_PER_BLOCKS>>>(A->data, B->data, out->data, A->size);
}

Matrix *sub(Matrix *A, Matrix *B)
{
  Matrix *out = empty(A->rows,A->cols);
  sub(A, B, out);
  return out;
}

Matrix *vStack(Matrix *A, Matrix *B)
{

  Matrix *out;
  if(A->cols == B->cols)
  {
	  out = empty(A->rows + B->rows,A->cols);
  }
  else
  {
	  out = empty(1,1);
	  printf("Wrong merge sizes!");
	  assert(0);
  }
  int block_size = (out->size/512) + 1;
  vStack<<<block_size,512>>>(A->data, B->data, out->data, out->size, A->rows, A->rows + B->rows,A->cols);

  return out;
}

void vStack(Matrix *A, Matrix *B, Matrix *out)
{
  if(A->cols != B->cols)
  {
	  printf("Wrong merge sizes!");
	  assert(0);
  }

  int block_size = (out->size/512) + 1;
  vStack<<<block_size,512>>>(A->data, B->data, out->data, out->size, A->rows, A->rows + B->rows,A->cols);
}

Matrix *hStack(Matrix *A, Matrix *B)
{

  Matrix *out;
  if(A->rows == B->rows)
  {
	  out = empty(A->rows,A->cols + B->cols);
  }
  else
  {
	  out = empty(1,1);
	  printf("Wrong merge sizes!");
	  assert(0);
  }
  int block_size = (out->size/512) + 1;
  hStack<<<block_size,512>>>(A->data, B->data, out->data, out->size, A->size);

  return out;
}

void hStackN(float** arrA, int general_size, Matrix *out, int matrices_count)
{
	int blocks = (out->size/THREADS_PER_BLOCKS) + 1;
	hStackN<<<blocks,THREADS_PER_BLOCKS>>>(arrA, general_size, out->data,  out->size, matrices_count);
}

void hStack(Matrix *A, Matrix *B, Matrix *out)
{
  if(A->rows != B->rows)
  {
	  printf("Wrong merge sizes!");
	  assert(0);
  }

  int block_size = (out->size/512) + 1;
  hStack<<<block_size,512>>>(A->data, B->data, out->data, out->size, A->size);
}

void sub(Matrix *A, Matrix *B, Matrix *out)
{
	checkMatrixOperation(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N, 0);
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	if(B->isSparse == 0)
		kSub<<<block_size,THREADS_PER_BLOCKS>>>(A->data, B->data, out->data, A->size);
	else
		kSub_Sparse<<<block_size,THREADS_PER_BLOCKS>>>(A->data, B->data,B->ptr_rows,B->idx_cols,out->data, A->rows,A->cols, B->size);
}

Matrix *mul(Matrix *A, Matrix *B)
{
  Matrix *out = empty(A->rows,A->cols);
  mul(A, B, out);

  return out;
}

void mul(Matrix *A, Matrix *B, Matrix *out)
{
	checkMatrixOperation(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N, 0);
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	kMul<<<block_size,THREADS_PER_BLOCKS>>>(A->data, B->data, out->data, A->size);
}

Matrix *div(Matrix *A, Matrix *B)
{
  Matrix *out = empty(A->rows,A->cols);
  
  div(A, B, out);

  return out;
}

void printData(Matrix *A)
{
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	kPrintData<<<block_size,THREADS_PER_BLOCKS>>>(A->data, A->size);
	cudaDeviceSynchronize();
}

void div(Matrix *A, Matrix *B, Matrix *out)
{
	checkMatrixOperation(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N, 0);
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	kDiv<<<block_size,THREADS_PER_BLOCKS>>>(A->data, B->data, out->data, A->size);
}

Matrix *scalarMul(Matrix *A, float a)
{
  Matrix *out = empty(A->rows,A->cols);
  scalarMul(A, a, out);

  return out;
}

void scalarMul(Matrix *A, float a, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kScalarMul<<<block_size,THREADS_PER_BLOCKS>>>(A->data, a, out->data, A->size);
}

Matrix *scalarAdd(Matrix *A, float a)
{
  Matrix *out = empty(A->rows,A->cols);
  scalarAdd(A, a, out);

  return out;
}

void scalarAdd(Matrix *A, float a, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kScalarAdd<<<block_size,THREADS_PER_BLOCKS>>>(A->data, a, out->data, A->size);
}

Matrix *gpuExp(Matrix *A)
{
  Matrix *out = empty(A->rows,A->cols);
  gpuExp(A, out);

  return out;
}

void gpuExp(Matrix *A, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kExp<<<block_size,THREADS_PER_BLOCKS>>>(A->data, out->data, A->size);
}

Matrix *logistic(Matrix *A)
{
  Matrix *out = empty(A->rows,A->cols);
  logistic(A, out);

  return out;
}

void logistic(Matrix *A, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kLogistic<<<block_size,THREADS_PER_BLOCKS>>>(A->data, out->data, A->size);
}

Matrix *logisticGrad(Matrix *A)
{
  Matrix *out = empty(A->rows,A->cols);
  logisticGrad(A, out);

  return out;
}

void logisticGrad(Matrix *A, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kLogisticGrad<<<block_size,THREADS_PER_BLOCKS>>>(A->data, out->data, A->size);
}

Matrix *gpuLog(Matrix *A)
{
  Matrix *out = empty(A->rows,A->cols);
  gpuLog(A, out);

  return out;
}

void gpuLog(Matrix *A, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kLog<<<block_size,THREADS_PER_BLOCKS>>>(A->data, out->data, A->size);
}

Matrix *gpuSqrt(Matrix *A)
{
  Matrix *out = empty(A->rows,A->cols);
  gpuSqrt(A, out);

  return out;
}

void gpuSqrt(Matrix *A, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kSqrt<<<block_size,THREADS_PER_BLOCKS>>>(A->data, out->data, A->size);
}

Matrix *doubleRectifiedLinear(Matrix *A)
{
  Matrix *out = empty(A->rows,A->cols);
  doubleRectifiedLinear(A, out);

  return out;
}

void doubleRectifiedLinear(Matrix *A, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kDoubleRectifiedLinear<<<block_size,THREADS_PER_BLOCKS>>>(A->data, out->data, A->size);
}

Matrix *hardTanH(Matrix *A)
{
  Matrix *out = empty(A->rows,A->cols);
  hardTanH(A, out);

  return out;
}

void hardTanH(Matrix *A, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kHardTanH<<<block_size,THREADS_PER_BLOCKS>>>(A->data, out->data, A->size);
}

Matrix *pairwise_ranking(Matrix *A, Matrix *B)
{
  Matrix *out = empty(A->rows,A->cols);
  pairwise_ranking(A, B, out);

  return out;
}

void pairwise_ranking(Matrix *A, Matrix *B, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kPairwise_ranking<<<block_size,THREADS_PER_BLOCKS>>>(A->data, B->data, out->data, A->size);
}

Matrix *pairwise_ranking_derivative(Matrix *A, Matrix *B)
{
  Matrix *out = empty(A->rows,A->cols);
  pairwise_ranking_derivative(A, B, out);

  return out;
}

void pairwise_ranking_derivative(Matrix *A, Matrix *B, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kPairwise_ranking_derivative<<<block_size,THREADS_PER_BLOCKS>>>(A->data, B->data, out->data, A->size);
}

Matrix *square(Matrix *A)
{
  Matrix *out = empty(A->rows,A->cols);
  square(A, out);

  return out;
}

void square(Matrix *A, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kSquare<<<block_size,THREADS_PER_BLOCKS>>>(A->data, out->data, A->size);
}

int blnFaultySizes(Matrix *A, Matrix *B, Matrix *C)
{
  if((A->rows == B->rows) &&
     (A->cols == B->cols) &&
     (C->rows == A->rows) &&
     (C->cols == A->cols))
  {
    return 0;
  }
  else
  {
    return 1;
  }
}

int blnFaultyMatrixProductSizes(Matrix *A, Matrix *B, Matrix *C, cublasOperation_t T1, cublasOperation_t T2)
{
	int A_rows = A->rows, A_cols = A->cols, B_rows = B->rows, B_cols = B->cols;

	if (T1 == CUBLAS_OP_T)
	{
		A_rows = A->cols;
		A_cols = A->rows;
	}
	if (T2 == CUBLAS_OP_T)
	{
		B_cols = B->rows;
		B_rows = B->cols;
	}

   if((A_cols == B_rows) &&
      (A_rows == C->rows) &&
      (B_cols == C->cols))
  {
    return 0;
  }
  else
  {
    return 1;
  }
}

void printFaultySizeError(Matrix *A, Matrix *B, Matrix *C)
{
  printf("Error: Faulty element-wise matrix operation: \n");
  if(A->rows != B->rows || A->cols != B->cols)
  {
    printf("Matrix *A is of size %ix%i while Matrix *B is of size %ix%i.\n",
           A->rows,A->cols,B->rows,B->cols);
    assert(0);
  }
  else if((A->rows == B->rows)  && (A->cols == B->cols) &&          
  	  ((C->rows != A->rows) || (C->cols != A->cols)))
  {
    printf("Output Matrix *is of size %ix%i while the other matrices are of size %ix%i.\n",
           C->rows,C->cols,B->rows,B->cols);
    assert(0);
  }
}
void printFaultyMatrixProductSizeError(Matrix *A, Matrix *B, Matrix *C, cublasOperation_t T1, cublasOperation_t T2)
{
	int A_rows = A->rows, A_cols = A->cols, B_rows = B->rows, B_cols = B->cols;

	if (T1 == CUBLAS_OP_T)
	{
		A_rows = A->cols;
		A_cols = A->rows;
	}
	if (T2 == CUBLAS_OP_T)
	{
		B_cols = B->rows;
		B_rows = B->cols;
	}


    printf("Error: Faulty dot product matrix operation:\n");
	if(A_cols != B_rows)
	{
		printf("Matrix *A is of size %ix%i while Matrix *B is of size %ix%i.\n",
				A_rows,A_cols,B_rows,B_cols);
	}
	else if((A_cols == B_rows)  && ((C->rows != A_rows) || (C->cols != B_cols)))
	{
		printf("Output Matrix *is of size %ix%i while Matrix *A and B have sizes %ix%i and %ix%i.\n",
		   C->rows,C->cols,A_rows,A_cols, B_rows,B_cols);
	}
}

int checkMatrixOperation(Matrix *A, Matrix *B, Matrix *C, cublasOperation_t T1, cublasOperation_t T2, int blnMatrixProduct)
{
  if(blnMatrixProduct == 0)
  {
    if(blnFaultySizes(A, B, C) == 1)
    {
    	printFaultySizeError(A, B, C);
    	return 1;
    }

  }
  else
  {
    if(blnFaultyMatrixProductSizes(A, B, C, T1, T2) == 1)
    {
      printFaultyMatrixProductSizeError(A, B, C, T1, T2);
      return 1;
    }
  }

  return 0;
}

Matrix *softmax(Matrix *A)
{
	Matrix *out = empty(A->rows,A->cols);
	softmax(A, out);
	return out;
}



Matrix *subMatrixVector(Matrix *A, Matrix *v)
{
	Matrix *out = empty(A->rows,A->cols);
	subMatrixVector(A, v, out);

	return out;
}

void subMatrixVector(Matrix *A, Matrix *v, Matrix *out)
{
	int blocks = (A->size/THREADS_PER_BLOCKS) + 1;
	kSubMatrixVector<<<blocks,THREADS_PER_BLOCKS>>>(A->data, v->data, out->data, A->rows, A->size);
}

Matrix *addMatrixVector(Matrix *A, Matrix *v)
{
	Matrix *out = empty(A->rows,A->cols);
	addMatrixVector(A, v, out);

	return out;
}

void addMatrixVector(Matrix *A, Matrix *v, Matrix *out)
{
	if(A->cols != v->cols)
		printf("Error dimensions do not match: %i columns for matrix vs. %i for vector",A->cols,v->cols);

	assert(A->cols == v->cols);
	int blocks = (A->size/THREADS_PER_BLOCKS) + 1;
	kAddMatrixVector<<<blocks,THREADS_PER_BLOCKS>>>(A->data, v->data, out->data, A->rows, A->size);
}

void softmax(Matrix *A, Matrix *out)
{
    kSoftMax<<<1, A->rows > THREADS_PER_BLOCKS ? THREADS_PER_BLOCKS : A->rows>>>(A->data, out->data, A->rows, A->cols);

    cudaThreadSynchronize();

}

void sparseRdmWeight(Matrix *rdm, Matrix *idx, Matrix *out, int connections)
{
	int blocks = (idx->size/THREADS_PER_BLOCKS) + 1;
    kCreateSparseRdmWeight<<<blocks,THREADS_PER_BLOCKS>>>(rdm->data,idx->data, out->data, out->rows, out->cols, connections);
}


Matrix *create_t_matrix(Matrix *labels, int max_label)
{
	Matrix *out = zeros(labels->rows, max_label);
	create_t_matrix(labels, out);
	return out;
}

void create_t_matrix(Matrix *labels, Matrix *out)
{
	int blocks = (labels->size/THREADS_PER_BLOCKS) + 1;
	kCreate_t_matrix<<<blocks,THREADS_PER_BLOCKS>>>(labels->data, out->data, out->rows, labels->size);
}

Matrix *argmax(Matrix *A)
{
	//note: column major argmax
	Matrix *out = empty(A->rows,1);
	argmax(A, out);
	return out;
}
void argmax(Matrix* A, Matrix* out)
{
	kArgmax<<<1,A->rows > THREADS_PER_BLOCKS ? THREADS_PER_BLOCKS : A->rows>>>(A->data, out->data, A->rows, A->cols);

	cudaThreadSynchronize();

}

Matrix *equal(Matrix *A, Matrix *B)
{
	Matrix *out = empty(A->rows,A->cols);
	equal(A, B, out);

	return out;
}

void equal(Matrix *A, Matrix *B, Matrix *out)
{
	int blocks = (A->size/THREADS_PER_BLOCKS) + 1;
	kEqual<<<blocks,THREADS_PER_BLOCKS>>>(A->data, B->data, out->data, A->size);
}

float sum(Matrix *A)
{
	thrust::device_ptr<float> ptr(A->data);
	return thrust::reduce(ptr, ptr+A->size);
}

int getNonZeroElements(Matrix *A)
{
	Matrix *out = empty(1,1);
	int blocks = (A->size/THREADS_PER_BLOCKS) + 1;
	kGetNonZeroElements<<<blocks,THREADS_PER_BLOCKS>>>(A->data, out->data, A->size);
	Matrix *host = to_host(out);
	float out_value = host->data[0];
	cudaFree(out);
	free(host->data);
	free(host);

	return (int)out_value;
}

void dropout(Matrix *A, Matrix *rdm, float dropout_rate)
{
	int blocks = (A->size/THREADS_PER_BLOCKS) + 1;
	kDropout<<<blocks, THREADS_PER_BLOCKS>>>(A->data, rdm->data, dropout_rate, rdm->size);
}

void RMSprop(Matrix *RMS, Matrix *grad, float RMS_multiplier, float learning_rate, int batch_size)
{

	int blocks = (RMS->size/THREADS_PER_BLOCKS) + 1;
	kRMSprop<<<blocks,THREADS_PER_BLOCKS>>>(RMS->data, grad->data, RMS_multiplier, learning_rate, batch_size, RMS->size);
}

void RMSprop_with_nesterov_weight_update(Matrix *RMS, Matrix *grad, Matrix *w, Matrix *m, float RMS_multiplier, float learning_rate, int batch_size)
{

	int blocks = (RMS->size/THREADS_PER_BLOCKS) + 1;
	kRMSprop_with_nesterov_weight_update<<<blocks,THREADS_PER_BLOCKS>>>(RMS->data, grad->data, w->data, m->data, RMS_multiplier, learning_rate, batch_size, RMS->size);
}

Matrix *rectified_linear(Matrix *A)
{
	Matrix *out = empty(A->rows,A->cols);
	rectified_linear(A,out);
	return out;
}
void rectified_linear(Matrix *A, Matrix *out)
{
	int blocks = (out->size/THREADS_PER_BLOCKS) + 1;
	kRectifiedLinear<<<blocks,THREADS_PER_BLOCKS>>>(A->data, out->data, out->size);
}

Matrix *rectified_linear_derivative(Matrix *A)
{
	Matrix *out = empty(A->rows,A->cols);
	rectified_linear_derivative(A,out);
	return out;
}
void rectified_linear_derivative(Matrix *A, Matrix *out)
{
	int blocks = (out->size/THREADS_PER_BLOCKS) + 1;
	kRectifiedLinear_Derivative<<<blocks,THREADS_PER_BLOCKS>>>(A->data, out->data, out->size);
}

Matrix *double_rectified_linear_derivative(Matrix *A)
{
	Matrix *out = empty(A->rows,A->cols);
	double_rectified_linear_derivative(A,out);
	return out;
}
void double_rectified_linear_derivative(Matrix *A, Matrix *out)
{
	int blocks = (out->size/THREADS_PER_BLOCKS) + 1;
	kDoubleRectifiedLinear_Derivative<<<blocks,THREADS_PER_BLOCKS>>>(A->data, out->data, out->size);
}

Matrix *hardTanH_derivative(Matrix *A)
{
	Matrix *out = empty(A->rows,A->cols);
	hardTanH_derivative(A,out);
	return out;
}
void hardTanH_derivative(Matrix *A, Matrix *out)
{
	int blocks = (out->size/THREADS_PER_BLOCKS) + 1;
	kHardTanH_Derivative<<<blocks,THREADS_PER_BLOCKS>>>(A->data, out->data, out->size);
}

Matrix *maxColumnwise(Matrix *A)
{
	Matrix *out = empty(A->cols,1);
	maxColumnwise(A,out);
	return out;
}
void maxColumnwise(Matrix *A, Matrix *out)
{
	int shared_mem_size = 32 * sizeof(float);
	int w1 = floor(sqrt(A->cols));
	int w2 = A->cols / w1 + (A->cols % w1 == 0 ? 0 : 1);
	dim3 gridDim(w1, w2, 1);

	kMaxColumnwise<<<gridDim,32, shared_mem_size>>>(A->data, out->data, A->cols, A->rows);
	cudaThreadSynchronize();
}


Matrix **maxout(Matrix *A, int maxout_level)
{
	assert(A->cols % maxout_level == 0);
	assert(maxout_level <= 64);
	Matrix *out = empty(A->rows,A->cols/maxout_level);
	Matrix *outargmax = empty(A->rows,A->cols/maxout_level);
	maxout(A, out, outargmax, maxout_level);

	Matrix **out_array = (Matrix**)malloc(sizeof(Matrix*)*2);
	out_array[0] = out;
	out_array[1] = outargmax;

	return out_array;
}

void maxout(Matrix *A, Matrix *out, Matrix *outargmax, int maxout_level)
{
	int batch_size = A->rows;
	int grid_row_size = batch_size*out->cols < 65535 ? batch_size : floor(65535.0f/out->cols);
	if (grid_row_size*out->cols > 65535)
	{
		printf("column size to large for a maxout level of %i! Increase maxout level or decrease column width!", maxout_level);
		assert(false);
	}
	dim3 gridDim(grid_row_size, out->cols, 1);
	kMaxout<<<gridDim,32>>>(A->data, out->data, outargmax->data, maxout_level, A->cols, A->rows);
	cudaThreadSynchronize();
}


Matrix *squared_error(Matrix *A, Matrix *targets)
{
	Matrix *out = empty(A->rows,A->cols);
	squared_error(A,targets,out);
	return out;
}
void squared_error(Matrix *A, Matrix *targets, Matrix *out)
{
	int blocks = (out->size/THREADS_PER_BLOCKS) + 1;
	kSquaredError<<<blocks,THREADS_PER_BLOCKS>>>(A->data, targets->data, out->data, out->size);
}


void expand_to_maxout_grad(Matrix *error, Matrix *idx, Matrix *grad)
{
	int blocks = (grad->size/THREADS_PER_BLOCKS) + 1;
	assert(grad->cols % error->cols == 0);
	int maxout_level = grad->cols/error->cols;
	kExpandToMaxoutGrad<<<blocks,THREADS_PER_BLOCKS>>>(error->data, idx->data, grad->data, error->size, error->rows, maxout_level);
}




void sparse_dot(Matrix *A, Matrix *B, Matrix *out)
{
	int m = A->rows,
	        k = B->cols,
	        n = A->rows;

	    unsigned int grid_x = m / COPY_BLOCK_SIZE;
	    if (m % COPY_BLOCK_SIZE)
	        grid_x++;

	    unsigned int grid_y = n / COPY_BLOCK_SIZE;
	    if (n % COPY_BLOCK_SIZE)
	        grid_y++;

	    dim3 grid(grid_y , grid_x , 1);
	    dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);

	    kSparseDot<<<grid, threads>>>(m, n, k, A->data,
	        A->ptr_rows ,
	        A->idx_cols,
	        B->data, out->data, 0.0f, 1.0f);

	    cudaDeviceSynchronize();

}

void construct_vocab_matrix(Matrix *vocab_idx, Matrix *vocab_idx_y, Matrix *batch_X, Matrix *batch_y, Matrix *vocab, Matrix *rdm_idx)
{
	assert(vocab->rows <= 1024);
	dim3 grid(vocab_idx->rows,vocab_idx->cols,1);
	kConstructVocabMatrix<<<grid,vocab->rows>>>(vocab_idx->data, vocab_idx_y->data, vocab->data, rdm_idx->data, batch_X->data, batch_y->data);
}

void update_vocab_with_gradient(Matrix *grad, Matrix *vocab_idx, Matrix *vocab, float learning_rate)
{
	assert(vocab->rows <= 1024);
	dim3 grid(vocab_idx->rows,vocab_idx->cols,1);
	kUpdateVocabWithGradient<<<grid,vocab->rows>>>(grad->data, vocab_idx->data, vocab->data, learning_rate);

	cudaThreadSynchronize();
}


