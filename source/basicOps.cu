#include <stdio.h>
#include <basicOps.cuh>
#include <clusterKernels.cuh>
#include <assert.h>
#include <util.cuh>
#include <cublas_v2.h>
#include <vector>

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
  Matrix *row_major = A;
	 if(is_row_major == 0)
		 row_major = to_row_major(A);
  float *cpu_data;
  cpu_data = (float*)malloc(row_major->bytes);
  cudaMemcpy(cpu_data,row_major->data,row_major->bytes,cudaMemcpyDefault);
  Matrix *out = (Matrix*)malloc(sizeof(Matrix));
  out->rows = row_major->rows;
  out->cols = row_major->cols;
  out->bytes = row_major->bytes;
  out->size = row_major->size;
  out->data = cpu_data;
  out->isDistributed = 0;
  out->cols_distributed = 0;

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
	kCreateRdmSqrtWeight_Logistic<<<block_size,THREADS_PER_BLOCKS>>>(uniform_rdm->data, uniform_rdm->rows, uniform_rdm->cols);
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
  
  int block_size = (out->size/THREADS_PER_BLOCKS) + 1;
  kFill_with<<<block_size,THREADS_PER_BLOCKS>>>(out->data, fill_value, out->size);
 
  return out;
}

Matrix *add(Matrix *A, Matrix *B)
{
  Matrix *out = empty(A->rows,A->cols);
  add(A, B, out);
  checkMatrixOperation(A, B, out, 0);

  return out;
}

void add(Matrix *A, Matrix *B, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kAdd<<<block_size,THREADS_PER_BLOCKS>>>(A->data, B->data, out->data, A->size);
}

Matrix *sub(Matrix *A, Matrix *B)
{
  Matrix *out = empty(A->rows,A->cols);
  sub(A, B, out);
  checkMatrixOperation(A, B, out, 0);

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
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kSub<<<block_size,THREADS_PER_BLOCKS>>>(A->data, B->data, out->data, A->size);
}

Matrix *mul(Matrix *A, Matrix *B)
{
  Matrix *out = empty(A->rows,A->cols);
  mul(A, B, out);
  checkMatrixOperation(A, B, out, 0);

  return out;
}

void mul(Matrix *A, Matrix *B, Matrix *out)
{
  int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
  kMul<<<block_size,THREADS_PER_BLOCKS>>>(A->data, B->data, out->data, A->size);
}

Matrix *div(Matrix *A, Matrix *B)
{
  Matrix *out = empty(A->rows,A->cols);
  
  div(A, B, out);
  checkMatrixOperation(A, B, out, 0);

  return out;
}

void div(Matrix *A, Matrix *B, Matrix *out)
{
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

int blnFaultyMatrixProductSizes(Matrix *A, Matrix *B, Matrix *C)
{
   if((A->cols == B->rows) &&
      (A->rows == C->rows) &&
      (B->cols == C->cols))
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
  printf("Error: Faulty Matrix *sizes:\n");
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
void printFaultyMatrixProductSizeError(Matrix *A, Matrix *B, Matrix *C)
{
    printf("Error: Faulty Matrix *sizes:\n");  
  if(A->cols != B->rows)
  {
    printf("Matrix *A is of size %ix%i while Matrix *B is of size %ix%i.\n",
           A->rows,A->cols,B->rows,B->cols);
  }
  else if((A->cols == B->rows)  &&          
  	  ((C->rows != A->rows) || (C->cols != B->cols)))
  {
    printf("Output Matrix *is of size %ix%i while Matrix *A and B have sizes %ix%i and %ix%i.\n",
           C->rows,C->cols,A->rows,A->cols, B->rows,B->cols);
  }
}

int checkMatrixOperation(Matrix *A, Matrix *B, Matrix *C, int blnMatrixProduct)
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
    if(blnFaultyMatrixProductSizes(A, B, C) == 1)
    {
      printFaultyMatrixProductSizeError(A, B, C);
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

Matrix *sum(Matrix *v)
{

	Matrix *out = empty(1,1);
	int blocks = (v->size/THREADS_PER_BLOCKS) + 1;
	kSum<<<blocks,THREADS_PER_BLOCKS>>>(v->data, out->data, v->size);

	return out;
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






