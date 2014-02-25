#include <stdio.h>
#include <basicOps.cuh>
#include <clusterKernels.cuh>
#include <assert.h>

Matrix to_gpu(Matrix A){ return to_gpu(A, 0); }
Matrix to_gpu(Matrix A, int is_col_major)
{
  float * gpu_data;
  cudaMalloc((void**)&gpu_data,A.bytes);
  cudaMemcpy(gpu_data,A.data,A.bytes,cudaMemcpyDefault);
  Matrix out = {{A.shape[0],A.shape[1]},A.bytes,A.size,gpu_data};

  if(is_col_major == 0)
	  out = to_col_major(out);

  return out;
}

Matrix to_host(Matrix A){ return to_host(A, 0); }
Matrix to_host(Matrix A, int is_row_major)
{
  Matrix row_major = A;
	 if(is_row_major == 0)
		 row_major = to_row_major(A);
  float *cpu_data;
  cpu_data = (float*)malloc(row_major.bytes);
  cudaMemcpy(cpu_data,row_major.data,row_major.bytes,cudaMemcpyDefault);
  Matrix out = {{row_major.shape[0],row_major.shape[1]},row_major.bytes,row_major.size,cpu_data};



  return out;
}


static inline void T(Matrix A, Matrix out, int rows, int cols)
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
  kTranspose<<< grid, threads >>>(A.data, out.data, rows, cols);

}

Matrix to_col_major(Matrix A)
{
  Matrix out = empty(A.shape[0],A.shape[1]);
  T(A, out, A.shape[1],A.shape[0]);
  //cudaFree(A.data);
  return out;
}

Matrix to_row_major(Matrix A)
{
  Matrix out = empty(A.shape[0],A.shape[1]);
  T(A, out, A.shape[0],A.shape[1]);
  //cudaFree(A.data);
  return out;
}

Matrix T(Matrix A)
{
  Matrix out = empty(A.shape[1],A.shape[0]);
  T(A, out, A.shape[0],A.shape[1]);

  out.shape[0] = A.shape[1];
  out.shape[1] = A.shape[0];
  return out;
}




Matrix slice_rows(Matrix A, int start, int end)
{
  //align memory in contiguous array

  Matrix out = empty((end - start) + 1, A.shape[1]);
  int block_size = (out.size/1024) + 1;
  slice_rows<<<block_size,1024>>>(A.data, out.data, out.size, A.shape[0], start, end);

  return out;
}

Matrix slice_cols(Matrix A, int start, int end)
{
  Matrix out = empty(A.shape[0], end - start + 1);
  int block_size = (out.size/1024) + 1;
  slice_cols<<<block_size,1024>>>(A.data, out.data, start, A.shape[0], out.size);

  return out;
}

Matrix zeros(int rows, int cols)
{
  return fill_matrix(rows, cols, 0.0f);
}

Matrix ones(int rows, int cols)
{
  return fill_matrix(rows, cols, 1.0f);
}

Matrix empty(int rows, int cols)
{
  float *gpu_data;
  int size = rows*cols;
  size_t bytes = rows*cols*sizeof(float);
  cudaMalloc((void**)&gpu_data, bytes);
  
  Matrix A = {{rows, cols}, bytes, size, gpu_data};

  return A;
}


Matrix fill_matrix(int rows, int cols, float fill_value)
{
  if(rows < 1 || cols < 1)
  {
    printf("Error: Dimensions must be greater than zero!\n");
    assert(0);  
  }
 

  float *gpu_data;
  int size = rows*cols;
  size_t bytes = rows*cols*sizeof(float);
  cudaMalloc((void**)&gpu_data, bytes);
  
  int block_size = (size/1024) + 1;
  kFill_with<<<block_size,1024>>>(gpu_data, fill_value, size);

  Matrix m = {{rows, cols}, bytes, size, gpu_data};
 
  return m;
}

Matrix add(Matrix A, Matrix B)
{
  Matrix out = zeros(A.shape[0],A.shape[1]);
  add(A, B, out);
  checkMatrixOperation(A, B, out, 0);

  return out;
}

void add(Matrix A, Matrix B, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kAdd<<<block_size,1024>>>(A.data, B.data, out.data, A.size);
}

Matrix sub(Matrix A, Matrix B)
{
  Matrix out = zeros(A.shape[0],A.shape[1]);
  sub(A, B, out);
  checkMatrixOperation(A, B, out, 0);

  return out;
}

Matrix vStack(Matrix A, Matrix B)
{

  Matrix out;
  if(A.shape[1] == B.shape[1])
  {
	  out = empty(A.shape[0] + B.shape[0],A.shape[1]);
  }
  else
  {
	  out = empty(1,1);
	  printf("Wrong merge sizes!");
	  assert(0);
  }
  int block_size = (out.size/512) + 1;
  vStack<<<block_size,512>>>(A.data, B.data, out.data, out.size, A.shape[0], A.shape[0] + B.shape[0],A.shape[1]);

  return out;
}

void vStack(Matrix A, Matrix B, Matrix out)
{
  if(A.shape[1] != B.shape[1])
  {
	  printf("Wrong merge sizes!");
	  assert(0);
  }

  int block_size = (out.size/512) + 1;
  vStack<<<block_size,512>>>(A.data, B.data, out.data, out.size, A.shape[0], A.shape[0] + B.shape[0],A.shape[1]);
}

Matrix hStack(Matrix A, Matrix B)
{

  Matrix out;
  if(A.shape[0] == B.shape[0])
  {
	  out = empty(A.shape[0],A.shape[1] + B.shape[1]);
  }
  else
  {
	  out = empty(1,1);
	  printf("Wrong merge sizes!");
	  assert(0);
  }
  int block_size = (out.size/512) + 1;
  hStack<<<block_size,512>>>(A.data, B.data, out.data, out.size, A.size);

  return out;
}

void hStack(Matrix A, Matrix B, Matrix out)
{
  if(A.shape[0] != B.shape[0])
  {
	  printf("Wrong merge sizes!");
	  assert(0);
  }

  int block_size = (out.size/512) + 1;
  hStack<<<block_size,512>>>(A.data, B.data, out.data, out.size, A.size);
}

void sub(Matrix A, Matrix B, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kSub<<<block_size,1024>>>(A.data, B.data, out.data, A.size);
}

Matrix mul(Matrix A, Matrix B)
{
  Matrix out = zeros(A.shape[0],A.shape[1]);
  mul(A, B, out);
  checkMatrixOperation(A, B, out, 0);

  return out;
}

void mul(Matrix A, Matrix B, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kMul<<<block_size,1024>>>(A.data, B.data, out.data, A.size);
}

Matrix div(Matrix A, Matrix B)
{
  Matrix out = zeros(A.shape[0],A.shape[1]);
  
  div(A, B, out);
  checkMatrixOperation(A, B, out, 0);

  return out;
}

void div(Matrix A, Matrix B, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kDiv<<<block_size,1024>>>(A.data, B.data, out.data, A.size);
}





Matrix scalarMul(Matrix A, float a)
{
  Matrix out = zeros(A.shape[0],A.shape[1]);
  scalarMul(A, a, out);

  return out;
}

void scalarMul(Matrix A, float a, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kScalarMul<<<block_size,1024>>>(A.data, a, out.data, A.size);
}

Matrix gpuExp(Matrix A)
{
  Matrix out = zeros(A.shape[0],A.shape[1]);
  gpuExp(A, out);

  return out;
}

void gpuExp(Matrix A, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kExp<<<block_size,1024>>>(A.data, out.data, A.size);
}

Matrix gpuLog(Matrix A)
{
  Matrix out = zeros(A.shape[0],A.shape[1]);
  gpuLog(A, out);

  return out;
}

void gpuLog(Matrix A, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kLog<<<block_size,1024>>>(A.data, out.data, A.size);
}

Matrix gpuSqrt(Matrix A)
{
  Matrix out = zeros(A.shape[0],A.shape[1]);
  gpuSqrt(A, out);

  return out;
}

void gpuSqrt(Matrix A, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kSqrt<<<block_size,1024>>>(A.data, out.data, A.size);
}

Matrix square(Matrix A)
{
  Matrix out = zeros(A.shape[0],A.shape[1]);
  square(A, out);

  return out;
}

void square(Matrix A, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kSquare<<<block_size,1024>>>(A.data, out.data, A.size);
}

int blnFaultySizes(Matrix A, Matrix B, Matrix C)
{
  if((A.shape[0] == B.shape[0]) &&
     (A.shape[1] == B.shape[1]) &&
     (C.shape[0] == A.shape[0]) &&
     (C.shape[1] == A.shape[1]))
  {
    return 0;
  }
  else
  {
    return 1;
  }
}

int blnFaultyMatrixProductSizes(Matrix A, Matrix B, Matrix C)
{
   if((A.shape[1] == B.shape[0]) &&
      (A.shape[0] == C.shape[0]) &&
      (B.shape[1] == C.shape[1]))
  {
    return 0;
  }
  else
  {
    return 1;
  }
}

void printFaultySizeError(Matrix A, Matrix B, Matrix C)
{
  printf("Error: Faulty matrix sizes:\n");
  if(A.shape[0] != B.shape[0] || A.shape[1] != B.shape[1])
  {
    printf("Matrix A is of size %ix%i while matrix B is of size %ix%i.\n",
           A.shape[0],A.shape[1],B.shape[0],B.shape[1]);
    assert(0);
  }
  else if((A.shape[0] == B.shape[0])  && (A.shape[1] == B.shape[1]) &&          
  	  ((C.shape[0] != A.shape[0]) || (C.shape[1] != A.shape[1])))
  {
    printf("Output matrix is of size %ix%i while the other matrices are of size %ix%i.\n",
           C.shape[0],C.shape[1],B.shape[0],B.shape[1]);
    assert(0);
  }
}
void printFaultyMatrixProductSizeError(Matrix A, Matrix B, Matrix C)
{
    printf("Error: Faulty matrix sizes:\n");  
  if(A.shape[1] != B.shape[0])
  {
    printf("Matrix A is of size %ix%i while matrix B is of size %ix%i.\n",
           A.shape[0],A.shape[1],B.shape[0],B.shape[1]);
  }
  else if((A.shape[1] == B.shape[0])  &&          
  	  ((C.shape[0] != A.shape[0]) || (C.shape[1] != B.shape[1])))
  {
    printf("Output matrix is of size %ix%i while Matrix A and B have sizes %ix%i and %ix%i.\n",
           C.shape[0],C.shape[1],A.shape[0],A.shape[1], B.shape[0],B.shape[1]);
  }
}

int checkMatrixOperation(Matrix A, Matrix B, Matrix C, int blnMatrixProduct)
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

Matrix softmax(Matrix A)
{
	Matrix out = empty(A.shape[0],A.shape[1]);
	softmax(A, out);
	return out;
}

void softmax(Matrix A, Matrix out)
{
    unsigned int cols = A.shape[1],
                 rows = A.shape[0];



    //if (out.size[0] != h || target->size[1] != w)
    //    return ERROR_INCOMPATIBLE_DIMENSIONS;

    int shared_mem_size = 32 * sizeof(float) ;

    int w1 = floor(sqrt(rows));
    int w2 = rows / w1 + (rows % w1 == 0 ? 0 : 1);
    dim3 gridDim(w1, w2, 1);
    kSoftMax<<<gridDim, 32, shared_mem_size>>>(A.data, out.data, rows, cols);

    cudaThreadSynchronize();

}






