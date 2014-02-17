#include <stdio.h>
#include <clusterNet.cuh>
#include <cublas_v2.h>
#include <util.cuh>
#include <basicOps.cuh>



int main(int argc, char *argv[])
{
  Matrix h_y = read_csv("/home/tim/Downloads/mnist_full_y.csv");
  Matrix h_X = read_csv("/home/tim/Downloads/mnist_full_X.csv");

  Matrix d_y = to_gpu(h_y);
  Matrix d_X = to_gpu(h_X);

  printf("Done.\n");

  

}
