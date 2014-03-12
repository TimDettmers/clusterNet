#include <basicOps.cuh>
#include <clusterNet.h>
#include <batchAllocator.h>
#include <assert.h>
#include <stdio.h>
#include <util.cuh>
#include <batchAllocator.h>


int run_clusterNet_test(int argc, char *argv[])
{
	ClusterNet gpu = ClusterNet();
	ClusterNet ticktock_test = ClusterNet();
	ticktock_test.tick("ClusterNet test ran in");

	//dot test
	//      0 2    3             17 0
	// m1 = 0 0.83 59.1387  m2 =  3 4
	//                            0 0

	//row major data
	float m1_data[6] = {0,2,3,0,0.83,59.1387};
	float m2_data[6] = {17,0,3,4,0,0};
	size_t m1_bytes = 2*3*sizeof(float);
	Matrix *m1_cpu = (Matrix*)malloc(sizeof(Matrix));
	m1_cpu->rows = 2;
	m1_cpu->cols = 3;
	m1_cpu->bytes = m1_bytes;
	m1_cpu->size = 6;
	m1_cpu->data = m1_data;
	Matrix *m2_cpu = (Matrix*)malloc(sizeof(Matrix));
	m2_cpu->rows = 3;
	m2_cpu->cols = 2;
	m2_cpu->bytes = m1_bytes;
	m2_cpu->size = 6;
	m2_cpu->data = m2_data;
	Matrix *m1 = to_gpu(m1_cpu);
	Matrix *m2 = to_gpu(m2_cpu);

	Matrix *m_host = to_host(m1);


	Matrix *m3 = gpu.dot(m1,m2);
	Matrix *out = zeros(2,2);
	m_host = to_host(m3);

	assert(test_eq(m_host->data[0], 6.0f,"Dot data."));
	assert(test_eq(m_host->data[1], 8.0f,"Dot data."));
	assert(test_eq(m_host->data[2], 2.49f,"Dot data."));
	assert(test_eq(m_host->data[3], 3.32f,"Dot data."));
	assert(test_matrix(m_host,2,2));

	gpu.dot(m1,m2,out);
	m_host = to_host(out);
	assert(test_eq(m_host->data[0], 6.0f,"Dot data."));
	assert(test_eq(m_host->data[1], 8.0f,"Dot data."));
	assert(test_eq(m_host->data[2], 2.49f,"Dot data."));
	assert(test_eq(m_host->data[3], 3.32f,"Dot data."));
	assert(test_matrix(m_host,2,2));

	//Tdot test

	out = zeros(3,3);
	gpu.Tdot(m1,m1,out);
	m_host = to_host(out);
	assert(test_eq(m_host->data[0], 0.0f,"Dot data."));
	assert(test_eq(m_host->data[1], 0.0f,"Dot data."));
	assert(test_eq(m_host->data[2], 0.0f,"Dot data."));
	assert(test_eq(m_host->data[3], 0.0f,"Dot data."));
	assert(test_eq(m_host->data[4], 4.6889f,"Dot data."));
	assert(test_eq(m_host->data[5], 55.085117f,"Dot data."));
	assert(test_eq(m_host->data[6], 0.0f,"Dot data."));
	assert(test_eq(m_host->data[7], 55.085117f,"Dot data."));
	assert(test_eq(m_host->data[8], 3506.385742f,"Dot data."));
	assert(test_matrix(m_host,3,3));

	out = zeros(2,2);
	gpu.Tdot(m2,m2,out);
	m_host = to_host(out);
	assert(test_eq(m_host->data[0], 298.0f,"Dot data."));
	assert(test_eq(m_host->data[1], 12.0f,"Dot data."));
	assert(test_eq(m_host->data[2], 12.0f,"Dot data."));
	assert(test_eq(m_host->data[3], 16.0f,"Dot data."));
	assert(test_matrix(m_host,2,2));
	//dot T test
	gpu.dotT(m1,m1,out);
	m_host = to_host(out);
	assert(test_eq(m_host->data[0], 13.0f,"Dot data."));
	assert(test_eq(m_host->data[1], 179.0761f,"Dot data."));
	assert(test_eq(m_host->data[2], 179.0761f,"Dot data."));
	assert(test_eq(m_host->data[3], 3498.074463f,"Dot data."));
	assert(test_matrix(m_host,2,2));

	//test uniform random
	Matrix *r1 = gpu.rand(100,100);
	m_host = to_host(r1);
	int upper = 0;
	int lower = 0;
	int zeros = 0;
	for(int i = 0; i < r1->size; i++)
	{
	assert(m_host->data[i] >= 0.0f);
	assert(m_host->data[i] <= 1.0f);
	if(m_host->data[i] > 0.5f)
	   upper++;
	else
	   lower++;

	if(m_host->data[i] == 0)
	   zeros++;
	}
	//there should be more than 47% which is > 0.5
	assert(upper > (r1->size)*0.47f);
	assert(lower > (r1->size)*0.47f);
	assert(m_host->rows==100);
	assert(m_host->cols==100);
	assert(m_host->size==100*100);
	assert(m_host->bytes==r1->size*sizeof(float));

	//test same seeds
	ClusterNet gpu2 = ClusterNet(1234);
	gpu = ClusterNet(1234);
	r1 = gpu.rand(10,10);
	Matrix *r2 = gpu2.rand(10,10);
	Matrix *h1 = to_host(r1);
	Matrix *h2 = to_host(r2);
	for(int i = 0; i < 100; i++)
	{
	assert(h1->data[i] == h2->data[i]);
	}
	//test different seeds
	gpu2 = ClusterNet(1235);
	gpu = ClusterNet(1234);
	r1 = gpu.rand(10,10);
	r2 = gpu2.rand(10,10);
	h1 = to_host(r1);
	h2 = to_host(r2);
	for(int i = 0; i < 100; i++)
	{
	assert(h1->data[i] != h2->data[i]);
	}

	//test normal random
	r1 = gpu.randn(100,100);
	m_host = to_host(r1);
	upper = 0;
	lower = 0;
	int middle = 0;
	zeros = 0;
	for(int i = 0; i < r1->size; i++)
	{
	if(m_host->data[i] > 1.96f)
	   upper++;

	if(m_host->data[i] < -1.96f)
	   lower++;

	if(m_host->data[i] == 0)
	   zeros++;

	if((m_host->data[i] < 1) && (m_host->data[i] > -1))
	   middle++;
	}
	//a z-score of greater than 1.96 should only occur with 2.5% probability
	assert(upper < r1->size*0.04);
	assert(lower < r1->size*0.04);
	//the 68% of the data should be within one standard deviation
	assert((middle > r1->size*0.65) && (middle < r1->size*0.70));
	//if there are more than 1% zeros then there is something fishy
	assert(zeros < r1->size*0.01);
	assert(m_host->rows==100);
	assert(m_host->cols==100);
	assert(m_host->size==100*100);
	assert(m_host->bytes==r1->size*sizeof(float));

	//dotMPI_batchSlice test
	gpu = ClusterNet(argc, argv, 12345);
	m1 = gpu.rand(200,400);
	m2 = gpu.rand(400,800);
	m3 = gpu.dot(m1,m2);
	Matrix *m4 = gpu.dotMPI_batchSlice(m1,m2);
	m3 = to_host(m3);
	m4 = to_host(m4);
	if(gpu.MYRANK == 0)
	{

	  for (int i = 0; i < m3->size; ++i)
	  {
		  assert(test_eq(m3->data[i],m4->data[i],i,i,"dotMPI Test"));
	  }

	  assert(test_matrix(m3,200,800));
	  assert(test_matrix(m4,200,800));
	}

	//dotMPI_unitSlice test
	m1 = gpu.rand(200,400);
	m2 = gpu.rand(400,800);
	m3 = gpu.dot(m1,m2);
	m4 = gpu.dotMPI(m1,m2);
	m3 = to_host(m3);
	m4 = to_host(m4);
	if(gpu.MYRANK == 0)
	{

	  for (int i = 0; i < m3->size; ++i)
	  {
		  assert(test_eq(m3->data[i],m4->data[i],i,i,"dotMPI Test"));
	  }

	  assert(test_matrix(m3,200,800));
	  assert(test_matrix(m4,200,800));
	}

	gpu.shutdown();


	//dropout test
	m1 = gpu.rand(1000,1000);
	m_host = to_host(gpu.dropout(m1,0.5));
	assert(test_matrix(m_host,1000,1000));
	int count = 0;
	for(int i = 0; i < m1->size; i++)
	{
	   if(m_host->data[i] == 0.0f)
	   count++;
	}
	ASSERT((count >= 499000) && (count < 501000),"dropout test");
	m1 = gpu.rand(1000,1000);
	m_host = to_host(gpu.dropout(m1,0.2));
	count = 0;
	for(int i = 0; i < m1->size; i++)
	{
	   if(m_host->data[i] == 0.0f)
	   count++;
	}
	ASSERT((count >= 199000) && (count < 201000),"dropout test");
	m1 = gpu.rand(1000,1000);
	m_host = to_host(gpu.dropout(m1,0.73));
	count = 0;
	for(int i = 0; i < m1->size; i++)
	{
	   if(m_host->data[i] == 0.0f)
	   count++;
	}
	ASSERT((count >= 729000) && (count < 731000),"dropout test");



	//rdmsqrtweight test
	m1 = gpu.uniformSqrtWeight(784,777);
	m_host = to_host(m1);
	assert(test_matrix(m_host,784,777));
	count = 0;
	for(int i = 0; i < m1->size; i++)
	{
	  ASSERT((m_host->data[i] > -4.0f*sqrt(6.0f/(784.0+777.0))) && (m_host->data[i] < 4.0f*sqrt(6.0f/(784.0+777.0))),"RdmSqrtWeight test");
	  if(m_host->data[i] == 0)
		  count++;
	}

	ASSERT(count < 10,"RdmSqrtWeight test");

	//rand_int test
	m1 = gpu.rand_int(784,777,2,10);
	m_host = to_host(m1);
	assert(test_matrix(m_host,784,777));
	count = 0;
	for(int i = 0; i < m1->size; i++)
	{
	  ASSERT((m_host->data[i] >= 2) && (m_host->data[i] <= 10),"rand_int test");
	  if(m_host->data[i] == 0)
		  count++;
	}

	m1 = gpu.rand_int(100,100,782965,78254609);
	m_host = to_host(m1);
	assert(test_matrix(m_host,100,100));
	count = 0;
	for(int i = 0; i < m1->size; i++)
	{
	  ASSERT((m_host->data[i] >= 782965) && (m_host->data[i] <= 78254609),"rand_int test");
	  if(m_host->data[i] == 0)
		  count++;
	}
	ASSERT(count == 0,"rand_int test");

	m_host = to_host(gpu.rand_int(1000,1000,0,9));
	int counts[10] = {0,0,0,0,0,
					  0,0,0,0,0};
	assert(test_matrix(m_host,1000,1000));
	for(int i = 0; i < m_host->size; i++)
	{
		counts[(int)m_host->data[i]]++;
	}
	for(int i = 0; i < 10; i++)
	{
		//expectation is 100000 each
		ASSERT((counts[i] > 95000) && (counts[i] < 105000), "rand_int test");
	}

	//rdmsparseweight test
	m1 = gpu.sparseInitWeight(784,812);
	m_host = to_host(m1,1);
	assert(test_matrix(m_host,784,812));
	count = 0;
	for(int i = 0; i < m_host->size; i++)
	{
		if(m_host->data[i] != 0.0f)
		{
			count++;
		}
	}
	//average should be bigger than 14
	ASSERT(count/812.0f > 14.0f,"sparse weight test");
	m1 = gpu.sparseInitWeight(532,2000,73);
	m_host = to_host(m1);
	assert(test_matrix(m_host,532,2000));
	count = 0;
	for(int i = 0; i < m_host->size; i++)
	{
		if(m_host->data[i] != 0.0f)
		{
			count++;
		}
	}
	//average should be bigger than 65 (there is a high chance of re-rolling the same number)
	ASSERT(count/2000.0f > 65.0f,"sparse weight test");


	//This should just pass without error
	ticktock_test.tock("ClusterNet test ran in");

	return 0;
}


