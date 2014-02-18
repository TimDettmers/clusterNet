#include <basicOps.cuh>
#include <cudaLibraryOps.cuh>
#include <clusterKernels.cuh>
#include <cublas_v2.h>
#include <stdio.h>
#include "curand.h"
#include "curand_kernel.h"

Matrix dot(Matrix A, Matrix B)
{
  Matrix out = zeros(A.shape[0],B.shape[1]);
  dot(A, B, out);
  
  checkMatrixOperation(A, B, out, 1);

  return out;
}

void dot(Matrix A, Matrix B, Matrix out)
{	
  cublasStatus_t status;
	
  const float alpha = 1.0f;
  const float beta = 0.0f;

  //cublas
  cublasHandle_t h;
  cublasCreate(&h);      
    
  status = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 
                A.shape[0], B.shape[1], A.shape[1],
                &alpha, A.data, A.shape[0],
                B.data, B.shape[0],
                &beta, out.data, out.shape[0]);   
                
  if(status != CUBLAS_STATUS_SUCCESS)
    printf("CUBLAS ERROR!");
}

Matrix rand(int rows, int cols){ return rand(rows, cols, time(0)); }
Matrix rand(int rows, int cols, int seed)
{	
    curandState *devStates;
    int size = rows * cols;	
    const int bytes = sizeof(float)*size;
    float *rdm_data;
    cudaMalloc((void**) &rdm_data, bytes);
    cudaMalloc((void**) &devStates, sizeof(curandState)*size);
    setup_kernel<<<1 + (size/1024),1024>>>(devStates, seed);
    generate_uniform_kernel<<<1 + (size/(1024*256)),1024>>>(devStates, size, rdm_data);
        
    Matrix ret = {{rows, cols}, bytes, size, rdm_data};
    
    return ret;
}

Matrix randn(int rows, int cols){ return randn(rows, cols, time(0)); }
Matrix randn(int rows, int cols, int seed)
{	
    curandState *devStates;
    int size = rows * cols;	
    const int bytes = sizeof(float)*size;
    float *rdm_data;
    cudaMalloc((void**) &rdm_data, bytes);
    cudaMalloc((void**) &devStates, sizeof(curandState)*size);
    setup_kernel<<<1 + (size/1024),1024>>>(devStates, seed);
    generate_normal_kernel<<<1 + (size/(1024*256)),1024>>>(devStates, size, rdm_data);
        
    Matrix ret = {{rows, cols}, bytes, size, rdm_data};
    
    return ret;
}
