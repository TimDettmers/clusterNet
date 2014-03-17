#include <basicOps.cuh>
#include <clusterNet.h>
#include <assert.h>
#include <stdio.h>
#include <util.cuh>
#include <batchAllocator.h>
#include <basicOps.cuh>

using std::cout;
using std::endl;

int run_batchAllocator_test(ClusterNet gpus)
{
	Matrix *m1;
	Matrix *m2;
	Matrix *m_host;
	Matrix *m_host2;
	Matrix *m_host_dist;
	Matrix *m_host2_dist;

	//batch allocator test
	m1 = to_host(arange(10000,784));
	m2 = to_host(arange(10000,1));
	BatchAllocator b = BatchAllocator();
	b.init(m1,m2,0.20,128,256);
	assert(test_matrix(b.CURRENT_BATCH,128,784));
	assert(test_matrix(b.CURRENT_BATCH_Y,128,1));
	assert(test_matrix(b.CURRENT_BATCH_CV,256,784));
	assert(test_matrix(b.CURRENT_BATCH_CV_Y,256,1));

	BatchAllocator b_dist = BatchAllocator();
	b_dist.init(m1,m2,0.2,128,256,gpus,Distributed_weights);
	assert(test_matrix(b_dist.CURRENT_BATCH,128,784));
	assert(test_matrix(b_dist.CURRENT_BATCH_Y,128,1));
	assert(test_matrix(b_dist.CURRENT_BATCH_CV,256,784));
	assert(test_matrix(b_dist.CURRENT_BATCH_CV_Y,256,1));

	int value = 0;
	int value_y = 0;
	for(int epoch = 0; epoch < 2; epoch++)
	{
	  value = 0;
	  value_y = 0;
	  for(int batchno = 0; batchno < b.TOTAL_ITERATIONS; batchno++)
	  {
		  m_host = to_host(b.CURRENT_BATCH);
		  m_host2 = to_host(b.CURRENT_BATCH_Y);
		  b.allocate_next_batch_async();
		  m_host_dist = to_host(b_dist.CURRENT_BATCH);
		  m_host2_dist = to_host(b_dist.CURRENT_BATCH_Y);
		  b_dist.allocate_next_batch_async();

		  for(int i = 0; i <  b.CURRENT_BATCH->rows*784; i++)
		  {
			  assert(test_eq(m_host->data[i],(float)value,i,i,"Batch test"));
			  assert(test_eq(m_host_dist->data[i],(float)value,i,i,"Batch test"));
			  value++;
		  }

		  b_dist.broadcast_batch_to_PCI();

		  for(int i = 0; i <  b.CURRENT_BATCH->rows; i++)
		  {
			  assert(test_eq(m_host2->data[i],(float)value_y,i,i,"Batch test"));
			  assert(test_eq(m_host2_dist->data[i],(float)value_y,i,i,"Batch test"));
			  value_y++;
		  }

		  b.replace_current_batch_with_next();
		  b_dist.replace_current_batch_with_next();
	  }

	  assert(test_eq(value,6272000,"Batch test train 128"));
	  assert(test_eq(value_y,8000,"Batch test train 128"));

	  for(int batchno = 0; batchno < b.TOTAL_ITERATIONS_CV; batchno++)
	  {
		  m_host = to_host(b.CURRENT_BATCH_CV);
		  m_host2 = to_host(b.CURRENT_BATCH_CV_Y);
		  b.allocate_next_cv_batch_async();

		  for(int i = 0; i <  b.CURRENT_BATCH_CV->rows*784; i++)
		  {
			  assert(test_eq(m_host->data[i],(float)value,"Batch test"));
			  value++;
		  }

		  for(int i = 0; i <  b.CURRENT_BATCH_CV->rows; i++)
		  {
			  assert(test_eq(m_host2->data[i],(float)value_y,"Batch test"));
			  value_y++;
		  }

		  b.replace_current_cv_batch_with_next();
	  }
	}


	   assert(test_eq(value,7840000,"Batch test"));
	   assert(test_eq(value_y,10000,"Batch test"));

	   m1 = to_host(arange(70000,784));
	   m2 = to_host(arange(70000,10));
	   b = BatchAllocator();
	   b.init(m1,m2,0.20,128,512);
	   assert(test_matrix(b.CURRENT_BATCH,128,784));
	   assert(test_matrix(b.CURRENT_BATCH_Y,128,10));
	   assert(test_matrix(b.CURRENT_BATCH_CV,512,784));
	   assert(test_matrix(b.CURRENT_BATCH_CV_Y,512,10));

	   for(int epoch = 0; epoch < 2; epoch++)
	   {
		   value = 0;
		   value_y = 0;
		   for(int batchno = 0; batchno < b.TOTAL_BATCHES; batchno++)
		   {
			  m_host = to_host(b.CURRENT_BATCH);
			  m_host2 = to_host(b.CURRENT_BATCH_Y);
			  b.allocate_next_batch_async();

			  for(int i = 0; i < b.CURRENT_BATCH->rows*784; i++)
			  {
				  assert(test_eq(m_host->data[i],(float)value,"Batch test"));
				  value++;
			  }

			  for(int i = 0; i < b.CURRENT_BATCH->rows*10; i++)
			  {
				  assert(test_eq(m_host2->data[i],(float)value_y,"Batch test"));
				  value_y++;
			  }

			  b.replace_current_batch_with_next();
		   }

		   assert(test_eq(value,43904000,"Batch test"));
		   assert(test_eq(value_y,560000,"Batch test"));


		for(int batchno = 0; batchno < b.TOTAL_BATCHES_CV; batchno++)
		{
		  m_host = to_host(b.CURRENT_BATCH_CV);
		  m_host2 = to_host(b.CURRENT_BATCH_CV_Y);
		  b.allocate_next_cv_batch_async();

		  for(int i = 0; i < b.CURRENT_BATCH_CV->rows*784; i++)
		  {
			  assert(test_eq(m_host->data[i],(float)value,"Batch test"));
			  value++;
		  }

		  for(int i = 0; i < b.CURRENT_BATCH_CV->rows*10; i++)
		  {
			  assert(test_eq(m_host2->data[i],(float)value_y,"Batch test"));
			  value_y++;
		  }

		  b.replace_current_cv_batch_with_next();
		}

		assert(test_eq(value,54880000,"Batch test"));
		assert(test_eq(value_y,700000,"Batch test"));
	  }


  return 0;
}



