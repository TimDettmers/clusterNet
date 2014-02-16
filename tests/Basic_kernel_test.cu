#include <stdio.h>
#include <mpi.h>
#include <assert.h>
#include <basicOps.cuh>

int main(int argc, char *argv[])
{
  Matrix m1 = ones(5,5);
  Matrix m2 = ones(5,5);  
  Matrix m3 = zeros(5,5);
  Matrix out = zeros(5,5);
  
  //test to_host
  Matrix m_host = to_host(m1);
  assert(m_host.shape[0]==m1.shape[0]);
  assert(m_host.shape[1]==m1.shape[1]);
  assert(m_host.size==m1.size);
  assert(m_host.bytes==m1.bytes);

  //test fill_with
  for(int i = 0; i< 25; i++)
  {
    assert(m_host.data[i]==1.0f);
  }

  //test add
  m3 = add(m1,m2);
  m_host = to_host(m3);
  for(int i = 0; i< 25; i++)
  {
    assert(m_host.data[i]==2.0f);
  } 

  //test mul
  m3 = mul(m3,m3);
  m_host = to_host(m3);
  for(int i = 0; i< 25; i++)
  {
    assert(m_host.data[i]==4.0f);
  } 

  //test sub
  m3 = sub(m3,m1);
  m_host = to_host(m3);
  for(int i = 0; i< 25; i++)
  {
    assert(m_host.data[i]==3.0f);
  } 

  //test div
  m2 = add(m1,m2); //2
  m3 = div(m3,m2);
  m_host = to_host(m3);
  for(int i = 0; i< 25; i++)
  {
    assert(m_host.data[i]==1.5f);
  } 

  //test add with given output matrix  
  m_host = to_host(add(m3,m2,out));
  Matrix m_host2 = to_host(out);
  for(int i = 0; i< 25; i++)
  {
    assert(m_host.data[i]==3.5f);
    assert(m_host.data[i]==m_host2.data[i]);
  }

  //test sub with given output matrix  
  m_host = to_host(sub(m3,m2,out));
  m_host2 = to_host(out);
  for(int i = 0; i< 25; i++)
  {
    assert(m_host.data[i]==-0.5f);
    assert(m_host.data[i]==m_host2.data[i]);
  }

  //test mul with given output matrix  
  m_host = to_host(mul(m3,m2,out));
  m_host2 = to_host(out);
  for(int i = 0; i< 25; i++)
  {
    assert(m_host.data[i]==3.0f);
    assert(m_host.data[i]==m_host2.data[i]);
  }

  //test div with given output matrix  
  m_host = to_host(div(m3,m2,out));
  m_host2 = to_host(out);
  for(int i = 0; i< 25; i++)
  {
    assert(m_host.data[i]==0.75f);
    assert(m_host.data[i]==m_host2.data[i]);
  }
  

  printf("Basic kernel test successful.\n");

  return 0;
}


