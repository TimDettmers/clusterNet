#include <stdio.h>
#include <basicOps.cuh>
#include <clusterKernels.cuh>
#include <assert.h>

Matrix to_gpu(Matrix A)
{
  float * gpu_data;
  cudaMalloc((void**)&gpu_data,A.bytes);
  cudaMemcpy(gpu_data,A.data,A.bytes,cudaMemcpyDefault);
  Matrix gpu_matrix = {{A.shape[0],A.shape[1]},A.bytes,A.size,gpu_data};

  return gpu_matrix;
}

Matrix zeros(int rows, int cols)
{
  return fill_matrix(rows, cols, 0.0f);
}

Matrix ones(int rows, int cols)
{
  return fill_matrix(rows, cols, 1.0f);
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



Matrix to_host(Matrix A)
{
  float *cpu_data;
  cpu_data = (float*)malloc(A.bytes);
  cudaMemcpy(cpu_data,A.data,A.bytes,cudaMemcpyDefault);
  Matrix host_matrix = {{A.shape[0],A.shape[1]},A.bytes,A.size,cpu_data};
  return host_matrix;
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
    assert(0);
  }
  else if((A.shape[1] == B.shape[0])  &&          
  	  ((C.shape[0] != A.shape[0]) || (C.shape[1] != B.shape[1])))
  {
    printf("Output matrix is of size %ix%i while Matrix A and B have sizes %ix%i and %ix%i.\n",
           C.shape[0],C.shape[1],A.shape[0],A.shape[1], B.shape[0],B.shape[1]);
    assert(0);
  }
}

void checkMatrixOperation(Matrix A, Matrix B, Matrix C, int blnMatrixProduct)
{
  if(blnMatrixProduct == 0)
  {
    if(blnFaultySizes(A, B, C) == 1)
      printFaultySizeError(A, B, C);    
  }
  else
  {
    if(blnFaultyMatrixProductSizes(A, B, C) == 1)
      printFaultyMatrixProductSizeError(A, B, C);
  }
}

Matrix T(Matrix A)
{
  Matrix out = zeros(A.shape[0],A.shape[1]);
  T(A, out);

  return out;
}

void T(Matrix A, Matrix out)
{
  // setup execution parameters
  int grid_x = A.shape[0] / COPY_BLOCK_SIZE;
  if (A.shape[0] % COPY_BLOCK_SIZE)
    grid_x++;

  int grid_y = A.shape[1] / COPY_BLOCK_SIZE;
  if (A.shape[1] % COPY_BLOCK_SIZE)
    grid_y++;

  dim3 grid(grid_x, grid_y, 1);
  dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);
  kTranspose<<< grid, threads >>>(A.data, out.data, A.shape[0], A.shape[1]);
}


