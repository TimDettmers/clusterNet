#include <stdio.h>
#include <basicOps_test.cuh>
#include <clusterNet_test.cuh>
#include <batchAllocator_test.cuh>

int main(int argc, char *argv[])
{
  run_basicOps_test(argc, argv);
  run_clusterNet_test(argc, argv);
  run_batchAllocator_test(argc, argv);

  printf("----------------------\n");
  printf("All tests passed successfully!\n");
  printf("----------------------\n");



}
