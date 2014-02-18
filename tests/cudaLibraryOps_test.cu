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

  //test uniform random
  Matrix r1 = rand(100,100);
  m_host = to_host(r1);
  int upper = 0;
  int lower = 0;
  for(int i = 0; i < r1.size; i++)
  {
    assert(m_host.data[i] >= 0.0f);
    assert(m_host.data[i] <= 1.0f);
    if(m_host.data[i] > 0.5f)
       upper++;
    else
       lower++;
  }
  //there should be more than 45% which is > 0.5
  assert(upper > (r1.size)*0.45f);
  assert(lower > (r1.size)*0.45f);
  assert(m_host.shape[0]==100);
  assert(m_host.shape[1]==100);
  assert(m_host.size==100*100);
  assert(m_host.bytes==r1.size*sizeof(float));

  //test uniform random with seed
  r1 = rand(10,10,1234);
  Matrix r2 = rand(10,10,1234);
  Matrix h1 = to_host(r1);
  Matrix h2 = to_host(r2);
  for(int i = 0; i < 100; i++)
  {
    assert(h1.data[i] == h2.data[i]);
  }

  //test normal random
  r1 = randn(100,100);
  m_host = to_host(r1);
  upper = 0;
  lower = 0;
  int zeros = 0;
  for(int i = 0; i < r1.size; i++)
  {
    if(m_host.data[i] > 1.96f)
       upper++;

    if(m_host.data[i] < -1.96f)
       lower++;

    if(m_host.data[i] == 0)
       zeros++;
  }
  //a z-score of greater than 1.96 should only occur with 2.5% probability
  assert(upper < r1.size*0.05);
  assert(lower < r1.size*0.05);
  //if there are more than 5% zeros then there is something fishy
  assert(zeros < r1.size*0.05); 
  assert(m_host.shape[0]==100);
  assert(m_host.shape[1]==100);
  assert(m_host.size==100*100);
  assert(m_host.bytes==r1.size*sizeof(float));

  //test normal random with seed
  r1 = randn(10,10,1234);
  r2 = randn(10,10,1234);
  h1 = to_host(r1);
  h2 = to_host(r2);
  for(int i = 0; i < 100; i++)
  {
    assert(h1.data[i] == h2.data[i]);
  }

  return 0;
}
