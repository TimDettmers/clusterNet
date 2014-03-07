#include <basicOps.cuh>
#include <clusterNet.h>
#include <assert.h>
#include <stdio.h>
#include <util.cuh>
#include <batchAllocator.h>
#include <basicOps.cuh>


int run_batchAllocator_test(int argc, char *argv[])
{
  Matrix *m1;
  Matrix *m2;
  Matrix *m_host;

  //batch allocator test
  m1 = to_host(arange(10000,784));
  m2 = to_host(arange(10000,1));
  BatchAllocator b = BatchAllocator();
  b.init(m1,m2,0.20,128,256);
  assert(test_matrix(b.CURRENT_BATCH,128,784));
  assert(test_matrix(b.CURRENT_BATCH_Y,128,1));
  assert(test_matrix(b.CURRENT_BATCH_CV,256,784));
  assert(test_matrix(b.CURRENT_BATCH_CV_Y,256,1));
  Matrix *m_host2;
  int value = 0;
  int value_y = 0;
  for(int epoch = 0; epoch < 2; epoch++)
  {
	  value = 0;
  	  value_y = 0;
	  for(int batchno = 0; batchno < b.TOTAL_BATCHES; batchno++)
	  {
		  m_host = to_host(b.CURRENT_BATCH);
		  m_host2 = to_host(b.CURRENT_BATCH_Y);
		  b.allocate_next_batch_async();


		  //std::cout << "bachtsize X: " << gpu.CURRENT_BATCHSIZE << std::endl;

		  for(int i = 0; i <  b.CURRENT_BATCH->rows*784; i++)
		  {
			  assert(test_eq(m_host->data[i],(float)value,i,i,"Batch test"));
			  value++;
		  }

		  for(int i = 0; i <  b.CURRENT_BATCH->rows; i++)
		  {
			  assert(test_eq(m_host2->data[i],(float)value_y,i,i,"Batch test"));
			  value_y++;
		  }

		  b.replace_current_batch_with_next();
	  }

	  assert(test_eq(value,6272000,"Batch test train 128"));
	  assert(test_eq(value_y,8000,"Batch test train 128"));

	  for(int batchno = 0; batchno < b.TOTAL_BATCHES_CV; batchno++)
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



