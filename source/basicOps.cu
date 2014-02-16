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
  Matrix out = zeros(A.shape[0],B.shape[0]);
  return add(A, B, out);
}

Matrix add(Matrix A, Matrix B, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kAdd<<<block_size,1024>>>(A.data, B.data, out.data, A.size);
  
  return out;
}

Matrix sub(Matrix A, Matrix B)
{
  Matrix out = zeros(A.shape[0],B.shape[0]);
  return sub(A, B, out);
}

Matrix sub(Matrix A, Matrix B, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kSub<<<block_size,1024>>>(A.data, B.data, out.data, A.size);
  
  return out;
}

Matrix mul(Matrix A, Matrix B)
{
  Matrix out = zeros(A.shape[0],B.shape[0]);
  return mul(A, B, out);
}

Matrix mul(Matrix A, Matrix B, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kMul<<<block_size,1024>>>(A.data, B.data, out.data, A.size);
  
  return out;
}

Matrix div(Matrix A, Matrix B)
{
  Matrix out = zeros(A.shape[0],B.shape[0]);
  return div(A, B, out);
}

Matrix div(Matrix A, Matrix B, Matrix out)
{
  int block_size = (A.size/1024) + 1;
  kDiv<<<block_size,1024>>>(A.data, B.data, out.data, A.size);
  
  return out;
}

Matrix to_host(Matrix m)
{
  float *cpu_data;
  cpu_data = (float*)malloc(m.bytes);
  cudaMemcpy(cpu_data,m.data,m.bytes,cudaMemcpyDefault);
  Matrix host_matrix = {{m.shape[0],m.shape[1]},m.bytes,m.size,cpu_data};
  return host_matrix;
}
