#include <stdio.h>
#include <mpi.h>
#include <assert.h>
#include <basicOps.cuh>
#include <math.h>

int main(int argc, char *argv[])
{
  Matrix m1 = ones(5,6);
  Matrix m2 = ones(5,6);  
  Matrix m3 = zeros(5,6);
  Matrix out = zeros(5,6);
  
  //test to_host
  Matrix m_host = to_host(m1);
  assert(m_host.shape[0]==m1.shape[0]);
  assert(m_host.shape[1]==m1.shape[1]);
  assert(m_host.size==m1.size);
  assert(m_host.bytes==m1.bytes);

  //test fill_with
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==1.0f);
  }

  //test add
  m3 = add(m1,m2);
  m_host = to_host(m3);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==2.0f);
  } 

  //test mul
  m3 = mul(m3,m3);
  m_host = to_host(m3);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==4.0f);
  } 

  //test sub
  m3 = sub(m3,m1);
  m_host = to_host(m3);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==3.0f);
  } 

  //test div
  m2 = add(m1,m2); //2
  m3 = div(m3,m2);
  m_host = to_host(m3);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==1.5f);
  } 

  //test add with given output matrix  
  add(m3,m2,out);
  m_host = to_host(out);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==3.5f);
  }

  //test sub with given output matrix  
  sub(m3,m2,out);
  m_host = to_host(out);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==-0.5f);
  }

  //test mul with given output matrix  
  mul(m3,m2,out);
  m_host = to_host(out);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==3.0f);
  }

  //test div with given output matrix  
  div(m3,m2,out);
  m_host = to_host(out);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==0.75f);
  }
  
  //test exp
  m_host = to_host(gpuExp(zeros(5,6)));
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==1.0f);
  }

  //test scalar mul
  m_host = to_host(scalarMul(ones(5,6),1.83));
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==1.83f);
  }

  //test square
  m_host = to_host(gpuSqrt(scalarMul(ones(5,6),4)));
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==2.0f);
  }

  //test log
  m_host = to_host(gpuLog(scalarMul(ones(5,6),2.0)));
  for(int i = 0; i< 30; i++)
  {   
    assert(m_host.data[i]==log(2.0f));
  }

  //test square
  m_host = to_host(square(scalarMul(ones(5,6),2)));
  for(int i = 0; i< 30; i++)
  {
    assert(m_host.data[i]==4.0f);
  }

  printf("Basic kernel test successful.\n");

  return 0;
}


