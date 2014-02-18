#include <basicOps.cuh>
#include <cudaLibraryOps.cuh>
#include <clusterKernels.cuh>
#include <cublas_v2.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <util.cuh>



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

curandGenerator_t random_init(){ return random_init(time(0)); }
curandGenerator_t random_init(int seed)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);    
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandSetGeneratorOffset(gen, 100);

    return gen;
}

Matrix rand(curandGenerator_t gen, int rows, int cols)
{ 
    Matrix out = empty(rows, cols);
    rand(gen, rows, cols, out);

    return out;
}
void rand(curandGenerator_t gen, int rows, int cols, Matrix out){ curandGenerateUniform(gen, out.data, rows*cols); }

Matrix randn(curandGenerator_t gen, int rows, int cols){ return randn(gen, rows, cols, 0, 1); }
Matrix randn(curandGenerator_t gen, int rows, int cols, float mean, float std)
{
    Matrix out = empty(rows,cols);  
    randn(gen, rows, cols, mean, std, out);
    
    return out;
}
void randn(curandGenerator_t gen, int rows, int cols, float mean, float std, Matrix out){ curandGenerateNormal(gen, out.data, rows*cols, 0.0f, 1.0f); }

