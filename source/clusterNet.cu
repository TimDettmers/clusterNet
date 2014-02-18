#include <stdio.h>
#include <clusterNet.cuh>
#include <cublas_v2.h>
#include <util.cuh>
#include <basicOps.cuh>
#include <cudaLibraryOps.cuh>



int main(int argc, char *argv[])
{  
  curandGenerator_t gen = random_init();
  Matrix x = randn(gen, 10000,10000); 

}
