#include <basicOps.cuh>
#include <clusterNet.h>
#include <assert.h>
#include <stdio.h>
#include <util.cuh>


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
  if(gpu.m_rank == 0)
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
  m4 = gpu.dotMPI_unitSlice(m1,m2);
  m3 = to_host(m3);
  m4 = to_host(m4);
  if(gpu.m_rank == 0)
  {

	  for (int i = 0; i < m3->size; ++i)
	  {
		  assert(test_eq(m3->data[i],m4->data[i],i,i,"dotMPI Test"));
	  }

	  assert(test_matrix(m3,200,800));
	  assert(test_matrix(m4,200,800));
  }

  gpu.shutdown_MPI();

  //These should just pass without error
  ticktock_test.tock("ClusterNet test ran in");

  //batch allocator test
  m1 = to_host(arange(10000,784));
  m2 = to_host(arange(10000,1));
  gpu.init_batch_allocator(m1,m2,0.20,128,256);
  assert(test_matrix(gpu.m_current_batch_X,128,784));
  assert(test_matrix(gpu.m_current_batch_y,128,1));
  assert(test_matrix(gpu.m_current_batch_cv_X,256,784));
  assert(test_matrix(gpu.m_current_batch_cv_y,256,1));
  Matrix *m_host2;
  int value = 0;
  int value_y = 0;
  for(int epoch = 0; epoch < 2; epoch++)
  {
	  value = 0;
  	  value_y = 0;
	  for(int batchno = 0; batchno < gpu.TOTAL_BATCHES; batchno++)
	  {
		  m_host = to_host(gpu.m_current_batch_X);
		  m_host2 = to_host(gpu.m_current_batch_y);
		  gpu.allocate_next_batch_async();


		  //std::cout << "bachtsize X: " << gpu.CURRENT_BATCHSIZE << std::endl;

		  for(int i = 0; i <  gpu.m_current_batch_X->rows*784; i++)
		  {
			  assert(test_eq(m_host->data[i],(float)value,i,i,"Batch test"));
			  value++;
		  }

		  for(int i = 0; i <  gpu.m_current_batch_X->rows; i++)
		  {
			  assert(test_eq(m_host2->data[i],(float)value_y,i,i,"Batch test"));
			  value_y++;
		  }

		  gpu.replace_current_batch_with_next();
	  }

	  assert(test_eq(value,6272000,"Batch test train 128"));
	  assert(test_eq(value_y,8000,"Batch test train 128"));

	  for(int batchno = 0; batchno < gpu.TOTAL_BATCHES_CV; batchno++)
	  {
		  m_host = to_host(gpu.m_current_batch_cv_X);
		  m_host2 = to_host(gpu.m_current_batch_cv_y);
		  gpu.allocate_next_cv_batch_async();

		  for(int i = 0; i <  gpu.m_current_batch_cv_X->rows*784; i++)
		  {
			  assert(test_eq(m_host->data[i],(float)value,"Batch test"));
			  value++;
		  }

		  for(int i = 0; i <  gpu.m_current_batch_cv_X->rows; i++)
		  {
			  assert(test_eq(m_host2->data[i],(float)value_y,"Batch test"));
			  value_y++;
		  }

		  gpu.replace_current_cv_batch_with_next();
	  }
  }


	   gpu = ClusterNet(12326557);
	   assert(test_eq(value,7840000,"Batch test"));
	   assert(test_eq(value_y,10000,"Batch test"));

	   m1 = to_host(arange(70000,784));
	   m2 = to_host(arange(70000,10));
	   gpu.init_batch_allocator(m1,m2,0.20,128,512);
	   assert(test_matrix(gpu.m_current_batch_X,128,784));
	   assert(test_matrix(gpu.m_current_batch_y,128,10));
	   assert(test_matrix(gpu.m_current_batch_cv_X,512,784));
	   assert(test_matrix(gpu.m_current_batch_cv_y,512,10));

	   for(int epoch = 0; epoch < 2; epoch++)
	   {
		   value = 0;
		   value_y = 0;
		   for(int batchno = 0; batchno < gpu.TOTAL_BATCHES; batchno++)
		   {
			  m_host = to_host(gpu.m_current_batch_X);
			  m_host2 = to_host(gpu.m_current_batch_y);
			  gpu.allocate_next_batch_async();

			  for(int i = 0; i < gpu.m_current_batch_X->rows*784; i++)
			  {
				  assert(test_eq(m_host->data[i],(float)value,"Batch test"));
				  value++;
			  }

			  for(int i = 0; i < gpu.m_current_batch_X->rows*10; i++)
			  {
				  assert(test_eq(m_host2->data[i],(float)value_y,"Batch test"));
				  value_y++;
			  }

			  gpu.replace_current_batch_with_next();
		   }

		   assert(test_eq(value,43904000,"Batch test"));
		   assert(test_eq(value_y,560000,"Batch test"));


		for(int batchno = 0; batchno < gpu.TOTAL_BATCHES_CV; batchno++)
		{
		  m_host = to_host(gpu.m_current_batch_cv_X);
		  m_host2 = to_host(gpu.m_current_batch_cv_y);
		  gpu.allocate_next_cv_batch_async();

		  for(int i = 0; i < gpu.m_current_batch_cv_X->rows*784; i++)
		  {
			  assert(test_eq(m_host->data[i],(float)value,"Batch test"));
			  value++;
		  }

		  for(int i = 0; i < gpu.m_current_batch_cv_X->rows*10; i++)
		  {
			  assert(test_eq(m_host2->data[i],(float)value_y,"Batch test"));
			  value_y++;
		  }

		  gpu.replace_current_cv_batch_with_next();
		}

		assert(test_eq(value,54880000,"Batch test"));
		assert(test_eq(value_y,700000,"Batch test"));
	  }



	   //dropout test
	   m1 = gpu.rand(1000,1000);
	   m_host = to_host(gpu.dropout(m1,0.5));
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



  return 0;
}



