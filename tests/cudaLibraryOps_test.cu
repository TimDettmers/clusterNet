#include <basicOps.cuh>
#include <cudaLibraryOps.cuh>
#include <assert.h>
#include <stdio.h>


int run_cudaLibraryOps_test(int argc, char *argv[])
{

  //dot test
  //column major order
  //      0 2    3             17 0
  // m1 = 0 0.83 59.1387  m2 =  3 4
  //                            0 0
  float m1_data[6] = {0,0,2,0.83,3,59.1387};
  float m2_data[6] = {17,3,0,0,4,0};
  size_t m1_bytes = 2*3*sizeof(float);
  Matrix m1_cpu = {{2,3},m1_bytes,6,m1_data};
  Matrix m2_cpu = {{3,2},m1_bytes,6,m2_data};
  Matrix m1 = to_gpu(m1_cpu);
  Matrix m2 = to_gpu(m2_cpu);
  Matrix m3 = dot(m1,m2);
  Matrix out = zeros(2,2);

  Matrix m_host = to_host(m3);
  assert(m_host.shape[0]==2);
  assert(m_host.shape[1]==2);
  assert(m_host.data[0]==6.0f);	
  assert(m_host.data[1]==2.49f);
  assert(m_host.data[2]==8.0f);
  assert(m_host.data[3]==3.32f);
  assert(m_host.size==4);
  assert(m_host.bytes==4*sizeof(float));

  dot(m1,m2,out);
  m_host = to_host(out);  
  assert(m_host.shape[0]==2);
  assert(m_host.shape[1]==2);
  assert(m_host.data[0]==6.0f);	
  assert(m_host.data[1]==2.49f);
  assert(m_host.data[2]==8.0f);
  assert(m_host.data[3]==3.32f);
  assert(m_host.size==4);
  assert(m_host.bytes==4*sizeof(float));

  return 0;
}
