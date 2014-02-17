#include <basicOps.cuh>
#include <cudaLibraryOps.cuh>
#include <cublas_v2.h>
#include <stdio.h>

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
