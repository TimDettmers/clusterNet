#include <basicOps.cuh>
#include <cudaLibraryOps.cuh>
#include <assert.h>
#include <stdio.h>
#include <util.cuh>


int run_cudaLibraryOps_test(int argc, char *argv[])
{

  //dot test
  //      0 2    3             17 0
  // m1 = 0 0.83 59.1387  m2 =  3 4
  //                            0 0

  //row major data
  float m1_data[6] = {0,2,3,0,0.83,59.1387};
  float m2_data[6] = {17,0,3,4,0,0};
  size_t m1_bytes = 2*3*sizeof(float);
  Matrix m1_cpu = {{2,3},m1_bytes,6,m1_data};
  Matrix m2_cpu = {{3,2},m1_bytes,6,m2_data};
  Matrix m1 = to_gpu(m1_cpu);
  Matrix m2 = to_gpu(m2_cpu);

  Matrix m_host = to_host(m1);
	
  
  Matrix m3 = dot(m1,m2);
  Matrix out = zeros(2,2);

  m_host = to_host(m3);
  assert(test_eq(m_host.data[0], 6.0f,"Dot data."));	
  assert(test_eq(m_host.data[1], 8.0f,"Dot data."));
  assert(test_eq(m_host.data[2], 2.49f,"Dot data."));
  assert(test_eq(m_host.data[3], 3.32f,"Dot data."));	
  assert(test_matrix(m_host,2,2));

  dot(m1,m2,out);
  m_host = to_host(out);  
  assert(test_eq(m_host.data[0], 6.0f,"Dot data."));	
  assert(test_eq(m_host.data[1], 8.0f,"Dot data."));
  assert(test_eq(m_host.data[2], 2.49f,"Dot data."));
  assert(test_eq(m_host.data[3], 3.32f,"Dot data."));	
  assert(test_matrix(m_host,2,2));

  //test uniform random
  curandGenerator_t gen1 = random_init();
  curandGenerator_t gen2 = random_init(1234);
  Matrix r1 = rand(gen1, 100,100);
  m_host = to_host(r1);
  int upper = 0;
  int lower = 0;
  int zeros = 0;
  for(int i = 0; i < r1.size; i++)
  {
    assert(m_host.data[i] >= 0.0f);
    assert(m_host.data[i] <= 1.0f);
    if(m_host.data[i] > 0.5f)
       upper++;
    else
       lower++;

    if(m_host.data[i] == 0)
       zeros++;
  }
  //there should be more than 47% which is > 0.5
  assert(upper > (r1.size)*0.47f);
  assert(lower > (r1.size)*0.47f);
  assert(m_host.shape[0]==100);
  assert(m_host.shape[1]==100);
  assert(m_host.size==100*100);
  assert(m_host.bytes==r1.size*sizeof(float));

  //test uniform random with seed
  gen1 = random_init(1234);
  r1 = rand(gen1, 10,10);
  Matrix r2 = rand(gen2, 10,10);
  Matrix h1 = to_host(r1);
  Matrix h2 = to_host(r2);
  for(int i = 0; i < 100; i++)
  {
    assert(h1.data[i] == h2.data[i]);
  }

  //test normal random
  gen1 = random_init();
  r1 = randn(gen1, 100,100);
  m_host = to_host(r1);
  upper = 0;
  lower = 0;
  int middle = 0;
  zeros = 0;
  for(int i = 0; i < r1.size; i++)
  {
    if(m_host.data[i] > 1.96f)
       upper++;

    if(m_host.data[i] < -1.96f)
       lower++;

    if(m_host.data[i] == 0)
       zeros++;

    if((m_host.data[i] < 1) && (m_host.data[i] > -1))
       middle++;
  }
  //a z-score of greater than 1.96 should only occur with 2.5% probability
  assert(upper < r1.size*0.04);
  assert(lower < r1.size*0.04);
  //the 68% of the data should be within one standard deviation
  assert((middle > r1.size*0.65) && (middle < r1.size*0.70));
  //if there are more than 1% zeros then there is something fishy
  assert(zeros < r1.size*0.01); 
  assert(m_host.shape[0]==100);
  assert(m_host.shape[1]==100);
  assert(m_host.size==100*100);
  assert(m_host.bytes==r1.size*sizeof(float));
  
  //test normal random with seed
  gen1 = random_init(1234);
  gen2 = random_init(1234);
  r1 = randn(gen1, 10,10);
  r2 = randn(gen2, 10,10);
  h1 = to_host(r1);
  h2 = to_host(r2);
  for(int i = 0; i < 100; i++)
  {
    assert(h1.data[i] == h2.data[i]);
  }

  return 0;
}
