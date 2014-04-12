#include <stdio.h>
#include <mpi.h>
#include <assert.h>
//#include <basicOps.cuh>
#include <math.h>
#include <util.cuh>
#include <clusterNet.h>

int run_basicOps_test()
{

  ClusterNet gpu = ClusterNet();

  Matrix *m1 = ones(5,6);
  Matrix *m2 = ones(5,6);
  Matrix *m3 = zeros(5,6);
  Matrix *out = zeros(5,6);
  
  //to_col_major test
  //      0 2    3             
  // m1 = 0 0.83 59.1387  
  //                           
  float m1_data[6] = {0,2,3,0,0.83,59.1387};
  size_t m1_bytes = 2*3*sizeof(float);
  Matrix *m1_cpu = (Matrix*)malloc(sizeof(Matrix));
  m1_cpu->rows = 2;
  m1_cpu->cols = 3;
  m1_cpu->bytes = m1_bytes;
  m1_cpu->size = 6;
  m1_cpu->data = m1_data;

  m1 = to_gpu(m1_cpu,1);
  //to_col_major test
  m1 = to_col_major(m1);
  float *test;
  test = (float*)malloc(m1->bytes);
  cudaMemcpy(test,m1->data,m1->bytes,cudaMemcpyDefault);

  assert(test_eq(test[0], 0.0f,"To col major data."));
  assert(test_eq(test[1], 0.0f,"To col major data."));
  assert(test_eq(test[2], 2.0f,"To col major data."));
  assert(test_eq(test[3], 0.83f,"To col major data."));
  assert(test_eq(test[4], 3.0f,"To col major data."));
  assert(test_eq(test[5], 59.1387f,"To col major data."));



   m1 = to_row_major(m1);
   cudaMemcpy(test,m1->data,m1->bytes,cudaMemcpyDefault);

   assert(test_eq(test[0], 0.0f,"To row major data."));
   assert(test_eq(test[1], 2.0f,"To row major data."));
   assert(test_eq(test[2], 3.0f,"To row major data."));
   assert(test_eq(test[3], 0.0f,"To row major data."));
   assert(test_eq(test[4], 0.83f,"To row major data."));
   assert(test_eq(test[5], 59.1387f,"To row major data."));

   assert(test_eq(getNonZeroElements(m1),4 ,"Get non-zero elements."));


  //test to_host
  //data is converted to column major and then back to row major
  Matrix *m_host = to_host(to_gpu(m1_cpu));
  assert(m_host->rows==m1->rows);
  assert(m_host->cols==m1->cols);
  assert(m_host->size==m1->size);
  assert(m_host->bytes==m1->bytes);
  for(int i = 0; i< 5; i++)
  {
    assert(m_host->data[i]==m1_cpu->data[i]);
  }


  //test fill_with
  m1 = ones(5,6);
  m_host = to_host(m1);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==1.0f);
  }

  //test add
  m3 = add(m1,m2);
  m_host = to_host(m3);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==2.0f);
  } 

  //test to_gpu
  m_host =  to_host(add(to_gpu(m_host),to_gpu(m_host)));
  for(int i = 0; i< 30; i++)
  {
    assert(test_eq(m_host->data[i],4.0f,"To gpu data"));
  } 

  //test mul
  m3 = mul(m3,m3);
  m_host = to_host(m3);
  for(int i = 0; i< 30; i++)
  {
    assert(test_eq(m_host->data[i],4.0f,"Multiplication data"));
  } 

  //test sub
  m3 = sub(m3,m1);
  m_host = to_host(m3);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==3.0f);
  } 

  //test div
  m2 = add(m1,m2); //2
  m3 = div(m3,m2);
  m_host = to_host(m3);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==1.5f);
  } 

  //test add with given output Matrix *
  add(m3,m2,out);
  m_host = to_host(out);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==3.5f);
  }

  //test sub with given output Matrix *
  sub(m3,m2,out);
  m_host = to_host(out);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==-0.5f);
  }

  //test mul with given output Matrix *
  mul(m3,m2,out);
  m_host = to_host(out);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==3.0f);
  }

  //test div with given output Matrix *
  div(m3,m2,out);
  m_host = to_host(out);
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==0.75f);
  }
  
  //test exp
  m_host = to_host(gpuExp(zeros(5,6)));
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==1.0f);
  }

  //test scalar mul
  m_host = to_host(scalarMul(ones(5,6),1.83));
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==1.83f);
  }

  //test sqrt
  m_host = to_host(gpuSqrt(scalarMul(ones(5,6),4)));
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==2.0f);
  }

  //test log
  m_host = to_host(gpuLog(scalarMul(ones(5,6),2.0)));
  for(int i = 0; i< 30; i++)
  {   
    assert(m_host->data[i]==log(2.0f));
  }

  //test square
  m_host = to_host(square(scalarMul(ones(5,6),2)));
  for(int i = 0; i< 30; i++)
  {
    assert(m_host->data[i]==4.0f);
  }

  //test blnFaultySizes
  assert(blnFaultySizes(ones(1,3),ones(2,3),ones(2,3))==1);
  assert(blnFaultySizes(ones(1,3),ones(1,3),ones(2,3))==1);
  assert(blnFaultySizes(ones(1,3),ones(1,3),ones(1,3))==0);
  assert(blnFaultySizes(ones(3,3),ones(3,3),ones(3,3))==0);
  //test blnFaultyMatrixSizes
  assert(blnFaultyMatrixProductSizes(ones(1,3),ones(1,3),ones(3,3))==1);
  assert(blnFaultyMatrixProductSizes(ones(3,1),ones(1,3),ones(2,2))==1);
  assert(blnFaultyMatrixProductSizes(ones(3,1),ones(1,3),ones(3,3))==0);

  //transpose test
  //column major order
  //      0 2    3
  // m1 = 0 0.83 59.1387
  //
  //test to_gpu with is_col_major = 1
  m_host = to_host(T(to_gpu(m1_cpu)));
  assert(test_eq(m_host->data[0],0.0f,"Transpose data."));
  assert(m_host->data[1]==0.0f);
  assert(m_host->data[2]==2.0f);
  assert(m_host->data[3]==0.83f);
  assert(m_host->data[4]==3.0f);
  assert(m_host->data[5]==59.1387f);
  assert(test_matrix(m_host,3,2));

  //to host and to gpu test
  //      0 2    3
  // m1 = 0 0.83 59.1387
  //
  //to gpu and to host should cancel each other out
  m_host = to_host(to_gpu(m1_cpu));
  assert(m_host->data[0]==0.0f);
  assert(m_host->data[1]==2.0f);
  assert(m_host->data[2]==3.0f);
  assert(m_host->data[3]==0.0f);
  assert(m_host->data[4]==0.83f);
  assert(m_host->data[5]==59.1387f);
  assert(test_matrix(m_host,2,3));

  //to_gpu for col major data test
  //col major data
  float m2_data[6] = {0,0,2,0.83,3,59.1387};
  size_t m2_bytes = 2*3*sizeof(float);
  Matrix *m2_cpu = (Matrix*)malloc(sizeof(Matrix));
  m2_cpu->rows = 2;
  m2_cpu->cols = 3;
  m2_cpu->bytes = m2_bytes;
  m2_cpu->size = 6;
  m2_cpu->data = m2_data;
  m_host = to_host(to_gpu(m2_cpu,1));
  //should be in row major now
  assert(m_host->data[0]==0.0f);
  assert(m_host->data[1]==2.0f);
  assert(m_host->data[2]==3.0f);
  assert(m_host->data[3]==0.0f);
  assert(m_host->data[4]==0.83f);
  assert(m_host->data[5]==59.1387f);
  assert(test_matrix(m_host,2,3));


  //slice rows
  m1 = gpu.rand(10,10);
  m2 = to_host(slice_rows(m1, 2,5));
  m1 = to_host(m1);
  assert(test_matrix(m2,4,10));
  int idx = 0;
  for(int i = 20; i < 60; i++)
  {        
    assert(test_eq(m1->data[i], m2->data[idx], idx, i , "Row slice data"));
    idx++;
  }  

  //slice cols
  m1 = gpu.rand(10,10);
  m2 = to_host(slice_cols(m1, 2,5));
  m1 = to_host(m1);
  idx = 0;
  assert(test_matrix(m2,10,4));


  for(int i = 2; i < 100;i++)
  {
    if(((i % 10) < 6) &&
       ((i % 10) > 1))
    {  
      assert(test_eq(m1->data[i], m2->data[idx], idx, i , "Col slice data"));
      idx++;
    }
  }

  //softmax test
  m1 = softmax(ones(2056,10));
  m_host = to_host(m1);
  assert(test_matrix(m_host,2056,10));
  for(int i = 0; i < m_host->size; i++)
  {
	  assert(test_eq(m_host->data[i],0.1,"Softmax equal test"));
  }

  m1 = softmax(gpu.rand(2222,17));
  m_host = to_host(m1);
  assert(test_matrix(m_host,2222,17));
  float sum_value = 0;
  for(int i = 0; i < m_host->size; i++)
  {
	  sum_value += m_host->data[i];
	  if((i > 0) &&  (((i+1) % 17) == 0))
	  {
		  ASSERT((sum_value > 0.99) && (sum_value < 1.01), "Softmax row sum equal one");
		  sum_value = 0.0f;
	  }
  }


  m1 = zeros(10,10);
  m2 = ones(10,1);
  //sub matrix vector test: A - v
  m_host= to_host(subMatrixVector(m1,m2));
  assert(test_matrix(m_host,10,10));
  for(int i = 0; i < m_host->size; i++)
  {
	  assert(test_eq(m_host->data[i],-1.0f, "Matrix - vector, equal data test"));
  }
  m3 = gpu.rand(13,17);
  Matrix *m4 = gpu.rand(1,17);
  m_host = to_host(addMatrixVector(m3,m4));
  m3 = to_host(m3);
  m4 = to_host(m4);
  assert(test_matrix(m_host,13,17));
  for(int row = 0; row < m_host->rows; row++)
  {
	  for(int col = 0; col < m_host->cols; col++)
		  assert(test_eq(m_host->data[(row*m_host->cols) + col], m3->data[(row*m_host->cols) + col] + m4->data[col], "Matrix + vector, equal data test"));
  }

  //      0 2    3
  // m1 = 0 0.83 59.1387
  //
  //argmax test
  //col_value = A[(i*cols) + idx];
  m1 = argmax(to_gpu(m1_cpu));
  m_host = to_host(m1);
  assert(test_matrix(m_host,2,1));
  assert(test_eq(m_host->data[0],2.0f, "Argmax test"));
  assert(test_eq(m_host->data[1],2.0f, "Argmax test"));
  m1 = gpu.rand(2056,10);
  m_host = to_host(argmax(m1));
  int counts[10] = {0,0,0,0,0,
		  	  	  	0,0,0,0,0};
  assert(test_matrix(m_host,2056,1));
  for(int i = 0; i < m_host->size; i++)
  {
	  counts[(int)m_host->data[i]]++;
  }
  for(int i = 0; i < 10; i++)
  {
	  //expectation is 205.6 each;
	  ASSERT((counts[i] > 140) && (counts[i] < 280), "Argmax value test");
  }

  //create t matrix test
  m1 = scalarMul(ones(10,1),4);
  m1 = create_t_matrix(m1,7);
  m_host = to_host(m1);
  assert(test_matrix(m_host,10,7));
  for(int i = 0; i < m_host->size; i++)
  {
	  if((i % m1->cols) == 4)
	  {
		  assert(test_eq(m_host->data[i],1.0f, "Create t matrix data"));
	  }
	  else
	  {
		  assert(test_eq(m_host->data[i],0.0f, "Create t matrix data"));
	  }
  }

  //equal test
  gpu = ClusterNet(12345);
  ClusterNet gpu2 = ClusterNet(12345);
  m2 = gpu.rand(10,7);
  m1 = gpu2.rand(10,7);
  m_host = to_host(equal(m1,m2));
  assert(test_matrix(m_host,10,7));
  for(int i = 0; i < m_host->size; i++)
  {
	  assert(test_eq(m_host->data[i],1.0f, "Matrix matrix Equal data test"));
  }
  m1 = gpu2.rand(10,7);
  m_host = to_host(equal(m1,m2));
  assert(test_matrix(m_host,10,7));
  for(int i = 0; i < m_host->size; i++)
  {
	  assert(test_eq(m_host->data[i],0.0f, "Matrix matrix Equal data test"));
  }


  //test sum
  m1 = ones(10,1);
  m2 = ones(1,10);

  ASSERT(sum(m1) == 10.0f, "Vector sum test");
  ASSERT(sum(m2)  == 10.0f, "Vector sum test");
  m1 = ones(10,10);
  ASSERT(sum(m1)  == 100.0f, "Vector sum test");
  ASSERT(sum(scalarMul(m2,1.73)) > 17.29f, "Vector sum test");
  ASSERT(sum(scalarMul(m2,1.73)) < 17.31f, "Vector sum test");

  //logistic test
  m1 = zeros(2,2);
  m1 = to_host(logistic(m1));
  assert(test_matrix(m1,2,2));
  for(int i = 0; i < m1->size; i++)
  {
	  ASSERT(m1->data[i] == 0.5f,"Logistic data test.");
  }
  m1 = gpu.randn(100,100);
  m1 = to_host(logistic(m1));
  assert(test_matrix(m1,100,100));
  for(int i = 0; i < m1->size; i++)
  {
	  ASSERT((m1->data[i] > 0.0f) && (m1->data[i] < 1.0f),"Logistic data test.");
  }

  //logistic grad test
  m1 = ones(2,2);
  m1 = to_host(logisticGrad(m1));
  assert(test_matrix(m1,2,2));
  for(int i = 0; i < m1->size; i++)
  {
	  ASSERT(m1->data[i] == 0.0f,"Logistic data test.");
  }
  m1 = gpu.randn(100,100);
  m_host = to_host(m1);
  m1 = to_host(logisticGrad(m1));
  assert(test_matrix(m1,100,100));
  for(int i = 0; i < m1->size; i++)
  {
	  ASSERT(m_host->data[i]*(1-m_host->data[i]) == m1->data[i],"Logistic data test.");
  }

  //arange test
  m1 = arange(10,7);
  m_host = to_host(m1);
  assert(test_matrix(m_host,10,7));
  for(int i = 0; i < m1->size; i++)
  {
	  assert(test_eq(m_host->data[i],(float)i, "Arange data test."));
  }

  m1 = arange(101,10,7);
  m_host = to_host(m1);
  assert(test_matrix(m_host,10,7));
  for(int i = 0; i < m1->size; i++)
  {
	  assert(test_eq(m_host->data[i],(float)(i + 101), "Arange data test."));
  }

  //cutoff to probability test
  m_host = to_host(doubleRectifiedLinear(gpu.randn(123,357,0,10)));
  assert(test_matrix(m_host,123,357));
  for(int i = 0; i < m_host->size; i++)
	  ASSERT((m_host->data[i] <=1.0f) && (m_host->data[i] >=0.0f),"cutoff to probability test.");


  m1 = empty_sparse(17,83,0.01783,0.0);
  int elements = ceil(17*83*0.01783) + 1;
  ASSERT(m1->rows == 17, "empty sparse rows");
  ASSERT(m1->cols == 83, "empty sparse cols");
  ASSERT(m1->size == elements, "empty sparse size");
  ASSERT(m1->isSparse == 1, "empty sparse");
  ASSERT(m1->idx_bytes == sizeof(float)*elements, "empty sparse bytes");
  ASSERT(m1->bytes == sizeof(float)*elements, "empty sparse bytes");
  ASSERT(m1->ptr_bytes == sizeof(float)*(m1->rows + 1), "empty sparse bytes");

  m1 = empty_sparse(17,83,500);
  elements = 500;
  ASSERT(m1->rows == 17, "empty sparse rows");
  ASSERT(m1->cols == 83, "empty sparse cols");
  ASSERT(m1->size == elements, "empty sparse size");
  ASSERT(m1->isSparse == 1, "empty sparse");
  ASSERT(m1->idx_bytes == sizeof(float)*elements, "empty sparse bytes");
  ASSERT(m1->bytes == sizeof(float)*elements, "empty sparse bytes");
  ASSERT(m1->ptr_bytes == sizeof(float)*(m1->rows + 1), "empty sparse bytes");

  m1 = empty_pinned_sparse(171,837,0.01783,0.001110);
  elements = ceil(171*837*(0.01783+0.001110)) + 1;
  ASSERT(m1->rows == 171, "empty sparse rows");
  ASSERT(m1->cols == 837, "empty sparse cols");
  ASSERT(m1->size == elements, "empty sparse size");
  ASSERT(m1->isSparse == 1, "empty sparse");
  ASSERT(m1->idx_bytes == sizeof(float)*elements, "empty sparse bytes");
  ASSERT(m1->bytes == sizeof(float)*elements, "empty sparse bytes");
  ASSERT(m1->ptr_bytes == sizeof(float)*(m1->rows + 1), "empty sparse bytes");

  for(int i = 0; i < m1->size; i++)
  {
	  ASSERT(m1->data[i] == 0.0f,"empty sparse data");
	  ASSERT(m1->idx_cols[i] == 0.0f,"empty sparse data");
  }




  return 0;
}



