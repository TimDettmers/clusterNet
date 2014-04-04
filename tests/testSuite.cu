#include <stdio.h>
#include <basicOps_test.cuh>
#include <clusterNet_test.cuh>
#include <batchAllocator_test.cuh>
#include <miniMNIST_test.cuh>
#include <clusterNet.h>
#include <util_test.cuh>


int main(int argc, char *argv[])
{
  ClusterNet gpus = ClusterNet(argc,argv,132456);
  run_basicOps_test();
  run_clusterNet_test(gpus);
  run_batchAllocator_test(gpus);
  run_miniMNIST_test(gpus);
  //run_util_test();


  printf("----------------------\n");
  printf("All tests passed successfully!\n");
  printf("----------------------\n");

  gpus.shutdown_MPI();

}
