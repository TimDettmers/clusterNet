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
	  for(int batchno = 0; batchno < b.TOTAL_BATCHES; batchno++)
	  {
		  assert(b.CURRENT_BATCH->rows == 128 || b.CURRENT_BATCH->rows == 8000%128);
		  assert(b_dist.CURRENT_BATCH->rows == 128 || b_dist.CURRENT_BATCH->rows == 8000%128);

		  m_host = to_host(b.CURRENT_BATCH);
		  m_host2 = to_host(b.CURRENT_BATCH_Y);
		  b.broadcast_batch_to_processes();
		  m_host_dist = to_host(b_dist.CURRENT_BATCH);
		  m_host2_dist = to_host(b_dist.CURRENT_BATCH_Y);
		  b_dist.broadcast_batch_to_processes();


		  for(int i = 0; i <  b.CURRENT_BATCH->rows*784; i++)
		  {
			  assert(test_eq(m_host->data[i],(float)value,i,i,"Batch test"));
			  assert(test_eq(m_host_dist->data[i],(float)value,i,i,"Batch test"));
			  value++;
		  }

		  b.allocate_next_batch_async();
		  b_dist.allocate_next_batch_async();

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

	  for(int batchno = 0; batchno < b.TOTAL_BATCHES_CV; batchno++)
	  {
		  assert(b.CURRENT_BATCH_CV->rows == 256 || b.CURRENT_BATCH_CV->rows == 2000%256);
		  assert(b_dist.CURRENT_BATCH_CV->rows == 256 || b_dist.CURRENT_BATCH_CV->rows == 2000%256);
		  m_host = to_host(b.CURRENT_BATCH_CV);
		  m_host2 = to_host(b.CURRENT_BATCH_CV_Y);
		  b.broadcast_batch_cv_to_processes();
		  m_host_dist = to_host(b_dist.CURRENT_BATCH_CV);
		  m_host2_dist = to_host(b_dist.CURRENT_BATCH_CV_Y);
		  b_dist.broadcast_batch_cv_to_processes();

		  for(int i = 0; i <  b.CURRENT_BATCH_CV->rows*784; i++)
		  {
			  assert(test_eq(m_host->data[i],(float)value,"Batch test"));
			  assert(test_eq(m_host_dist->data[i],(float)value,"Batch test"));
			  value++;
		  }

		  b.allocate_next_cv_batch_async();
		  b_dist.allocate_next_cv_batch_async();

		  for(int i = 0; i <  b.CURRENT_BATCH_CV->rows; i++)
		  {
			  assert(test_eq(m_host2->data[i],(float)value_y,"Batch test"));
			  assert(test_eq(m_host2_dist->data[i],(float)value_y,"Batch test"));
			  value_y++;
		  }

		  b.replace_current_cv_batch_with_next();
		  b_dist.replace_current_cv_batch_with_next();
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

	    b_dist = BatchAllocator();
		b_dist.init(m1,m2,0.2,128,512,gpus,Distributed_weights);
		assert(test_matrix(b_dist.CURRENT_BATCH,128,784));
		assert(test_matrix(b_dist.CURRENT_BATCH_Y,128,10));
		assert(test_matrix(b_dist.CURRENT_BATCH_CV,512,784));
		assert(test_matrix(b_dist.CURRENT_BATCH_CV_Y,512,10));

	   for(int epoch = 0; epoch < 2; epoch++)
	   {
		   value = 0;
		   value_y = 0;
		   for(int batchno = 0; batchno < b.TOTAL_BATCHES; batchno++)
		   {
			  assert(b.CURRENT_BATCH->rows == 128 || b.CURRENT_BATCH->rows == 56000%128);
			  assert(b_dist.CURRENT_BATCH->rows == 128 || b_dist.CURRENT_BATCH->rows == 56000%128);
			  m_host = to_host(b.CURRENT_BATCH);
			  m_host2 = to_host(b.CURRENT_BATCH_Y);
			  b.broadcast_batch_to_processes();
			  m_host_dist = to_host(b_dist.CURRENT_BATCH);
			  m_host2_dist = to_host(b_dist.CURRENT_BATCH_Y);
			  b_dist.broadcast_batch_to_processes();

			  for(int i = 0; i < b.CURRENT_BATCH->rows*784; i++)
			  {
				  assert(test_eq(m_host->data[i],(float)value,"Batch test"));
				  assert(test_eq(m_host_dist->data[i],(float)value,"Batch test"));
				  value++;
			  }

			  b.allocate_next_batch_async();
			  b_dist.allocate_next_batch_async();

			  for(int i = 0; i < b.CURRENT_BATCH->rows*10; i++)
			  {
				  assert(test_eq(m_host2->data[i],(float)value_y,"Batch test"));
				  assert(test_eq(m_host2_dist->data[i],(float)value_y,"Batch test"));
				  value_y++;
			  }

			  b.replace_current_batch_with_next();
			  b_dist.replace_current_batch_with_next();
		   }

		   assert(test_eq(value,43904000,"Batch test"));
		   assert(test_eq(value_y,560000,"Batch test"));


		for(int batchno = 0; batchno < b.TOTAL_BATCHES_CV; batchno++)
		{
		  assert(b.CURRENT_BATCH_CV->rows == 512 || b.CURRENT_BATCH_CV->rows == 14000%512);
		  assert(b_dist.CURRENT_BATCH_CV->rows == 512 || b_dist.CURRENT_BATCH_CV->rows == 14000%512);
		  m_host = to_host(b.CURRENT_BATCH_CV);
		  m_host2 = to_host(b.CURRENT_BATCH_CV_Y);
		  b.broadcast_batch_cv_to_processes();
		  m_host_dist = to_host(b_dist.CURRENT_BATCH_CV);
		  m_host2_dist = to_host(b_dist.CURRENT_BATCH_CV_Y);
		  b_dist.broadcast_batch_cv_to_processes();

		  for(int i = 0; i < b.CURRENT_BATCH_CV->rows*784; i++)
		  {
			  assert(test_eq(m_host->data[i],(float)value,"Batch test"));
			  assert(test_eq(m_host_dist->data[i],(float)value,"Batch test"));
			  value++;
		  }

		  b.allocate_next_cv_batch_async();
		  b_dist.allocate_next_cv_batch_async();

		  for(int i = 0; i < b.CURRENT_BATCH_CV->rows*10; i++)
		  {
			  assert(test_eq(m_host2->data[i],(float)value_y,"Batch test"));
			  assert(test_eq(m_host2_dist->data[i],(float)value_y,"Batch test"));
			  value_y++;
		  }

		  b.replace_current_cv_batch_with_next();
		  b_dist.replace_current_cv_batch_with_next();
		}

		assert(test_eq(value,54880000,"Batch test"));
		assert(test_eq(value_y,700000,"Batch test"));
	  }


	char buff[1024] = {0};
	ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
	std::string path = std::string(buff);
	replace(path,"/build/testSuite.out","/tests/");


	Matrix *X;
	Matrix *y;


	if(gpus.MYGPUID == 0)
	{
		X = read_sparse_hdf5((path + "crowdflower_X_test.hdf5").c_str());
		y = read_sparse_hdf5((path + "crowdflower_y_test.hdf5").c_str());
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(gpus.MYGPUID == 1)
	{
		X = read_sparse_hdf5((path + "crowdflower_X_test.hdf5").c_str());
		y = read_sparse_hdf5((path + "crowdflower_y_test.hdf5").c_str());
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(gpus.MYGPUID == 2)
	{
		X = read_sparse_hdf5((path + "crowdflower_X_test.hdf5").c_str());
		y = read_sparse_hdf5((path + "crowdflower_y_test.hdf5").c_str());
	}

	b = BatchAllocator();
	b.init(X,y,0.20,128,256,gpus, Distributed_weights_sparse);
	assert(test_eq(b.CURRENT_BATCH->rows,128,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH->cols,9000,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_Y->rows,128,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_Y->cols,24,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV->rows,256,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV->cols,9000,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV_Y->rows,256,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV_Y->cols,24,"sparse distributed batch allocator test"));

	int index = 0;
	int index_rows = 0;

	for(int i = 0; i < b.TOTAL_BATCHES; i++)
	{

		b.broadcast_batch_to_processes();
		Matrix *s1 = to_host(b.CURRENT_BATCH);
		Matrix *B = ones(b.CURRENT_BATCH->cols,20);
		Matrix *out = zeros(b.CURRENT_BATCH->rows, B->cols);

		for(int j = 0; j < b.CURRENT_BATCH->size; j++)
		{
			assert(test_eq(X->data[index],s1->data[j],"sparse batch allocator data test"));
			assert(test_eq(X->idx_cols[index],s1->idx_cols[j],"sparse batch allocator data test"));
			index++;
		}

		assert(test_eq(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows],b.CURRENT_BATCH->size,"test sparse batch size"));
		assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows])*sizeof(float),(int)b.CURRENT_BATCH->bytes,"test sparse batch bytes"));
		assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows])*sizeof(int),(int)b.CURRENT_BATCH->idx_bytes,"test sparse batch bytes"));
		assert(test_eq((int)(b.CURRENT_BATCH->rows +1)*sizeof(int),(int)b.CURRENT_BATCH->ptr_bytes,"test sparse batch bytes"));
		for(int j = 0; j < b.CURRENT_BATCH->rows+1; j++)
		{
			assert(test_eq(X->ptr_rows[index_rows],s1->ptr_rows[j],"sparse batch allocator data test"));
			index_rows++;
		}
		index_rows--;


		gpus.dot_sparse(b.CURRENT_BATCH, B, out);
		cout << "myrank: " << gpus.MYRANK << " " << sum(out) << endl;
		MPI_Barrier(MPI_COMM_WORLD);
		ASSERT(sum(out) > -50000 && sum(out) < 50000, "sparse batching sparse dot output test");

		b.allocate_next_batch_async();
		b.replace_current_batch_with_next();


		cudaFree(s1->data);
		cudaFree(s1->idx_cols);
		cudaFree(s1->ptr_rows);
		free(s1);
		cudaFree(B->data);
		cudaFree(out->data);
		free(out);
		free(B);
	}

	if(gpus.MYGPUID != 0)
	{
		X = empty_sparse(1,1,1);
		y = empty_sparse(1,1,1);
	}

	b = BatchAllocator();
	b.init(X,y,0.20,128,256,gpus, Distributed_weights_sparse);
	assert(test_eq(b.CURRENT_BATCH->rows,128,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH->cols,9000,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_Y->rows,128,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_Y->cols,24,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV->rows,256,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV->cols,9000,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV_Y->rows,256,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV_Y->cols,24,"sparse distributed batch allocator test"));


	index_rows = 0;
	index = 0;
	for(int i = 0; i < b.TOTAL_BATCHES; i++)
	{
		Matrix *B = ones(b.CURRENT_BATCH->cols,20);
		Matrix *out = zeros(b.CURRENT_BATCH->rows, B->cols);

		b.broadcast_batch_to_processes();
		if(gpus.MYGPUID == 0)
		{
			Matrix *s1 = to_host(b.CURRENT_BATCH);

			for(int j = 0; j < b.CURRENT_BATCH->size; j++)
			{
				assert(test_eq(X->data[index],s1->data[j],"sparse batch allocator data test"));
				assert(test_eq(X->idx_cols[index],s1->idx_cols[j],"sparse batch allocator data test"));
				index++;
			}

			assert(test_eq(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows],b.CURRENT_BATCH->size,"test sparse batch size"));
			assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows])*sizeof(float),(int)b.CURRENT_BATCH->bytes,"test sparse batch bytes"));
			assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows])*sizeof(int),(int)b.CURRENT_BATCH->idx_bytes,"test sparse batch bytes"));
			assert(test_eq((int)(b.CURRENT_BATCH->rows +1)*sizeof(int),(int)b.CURRENT_BATCH->ptr_bytes,"test sparse batch bytes"));
			for(int j = 0; j < b.CURRENT_BATCH->rows+1; j++)
			{
				assert(test_eq(X->ptr_rows[index_rows],s1->ptr_rows[j],"sparse batch allocator data test"));
				index_rows++;
			}
			index_rows--;

			cudaFree(s1->data);
			cudaFree(s1->idx_cols);
			cudaFree(s1->ptr_rows);
			free(s1);

		}
		gpus.dot_sparse(b.CURRENT_BATCH, B, out);
		cout << "myrank: " << gpus.MYRANK << " " << sum(out) << endl;
		MPI_Barrier(MPI_COMM_WORLD);
		ASSERT(sum(out) > -50000 && sum(out) < 50000, "sparse batching sparse dot output test");

		b.allocate_next_batch_async();
		b.replace_current_batch_with_next();

		cudaFree(B->data);
		cudaFree(out->data);
		free(out);
		free(B);
	}


  return 0;
}




