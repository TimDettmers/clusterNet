#include <stdio.h>
#include <basicOps.cuh>
#include <clusterKernels.cuh>

Matrix allocate(Matrix m)
{
  float * gpu_data;
  cudaMalloc((void**)&gpu_data,m.bytes);
  cudaMemcpy(gpu_data,m.data,m.bytes,cudaMemcpyDefault);
  Matrix gpu_matrix = {{m.shape[0],m.shape[1]},m.bytes,m.size,gpu_data};

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
  return out;
}

void div(Matrix A, Matrix B, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kDiv<<<block_size,1024>>>(A.data, B.data, out.data, A.size);
}



Matrix to_host(Matrix m)
{
  float *cpu_data;
  cpu_data = (float*)malloc(m.bytes);
  cudaMemcpy(cpu_data,m.data,m.bytes,cudaMemcpyDefault);
  Matrix host_matrix = {{m.shape[0],m.shape[1]},m.bytes,m.size,cpu_data};
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

