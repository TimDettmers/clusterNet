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
	int index_y = 0;
	int index_rows = 0;
	int row_ptr_offset = 0;
	int row_ptr_offset_y = 0;

	for(int epoch = 0; epoch < 3; epoch++)
	{
		index_rows = 0;
		index = 0;
		index_y = 0;
		row_ptr_offset = 0;
		row_ptr_offset_y = 0;
		for(int i = 0; i < b.TOTAL_BATCHES; i++)
		{

			Matrix *s1 = to_host(b.CURRENT_BATCH);
			Matrix *s2 = to_host(b.CURRENT_BATCH_Y);
			Matrix *B = ones(b.CURRENT_BATCH->cols,20);
			Matrix *out = zeros(b.CURRENT_BATCH->rows, B->cols);

			b.broadcast_batch_to_processes();

			for(int j = 0; j < b.CURRENT_BATCH->size; j++)
			{
				assert(test_eq(X->data[index],s1->data[j],"sparse batch allocator data test"));
				assert(test_eq(X->idx_cols[index],s1->idx_cols[j],"sparse batch allocator data test"));
				index++;
			}

			for(int j = 0; j < b.CURRENT_BATCH_Y->size; j++)
			{
				assert(test_eq(y->data[index_y],s2->data[j],"sparse batch allocator data test"));
				assert(test_eq(y->idx_cols[index_y],s2->idx_cols[j],"sparse batch allocator data test"));
				index_y++;
			}

			assert(test_eq(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows],b.CURRENT_BATCH->size,"test sparse batch size"));
			assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows])*sizeof(float),(int)b.CURRENT_BATCH->bytes,"test sparse batch bytes"));
			assert(test_eq((int)b.CURRENT_BATCH->idx_bytes,(int)b.CURRENT_BATCH->bytes,"test sparse batch bytes"));
			assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows])*sizeof(int),(int)b.CURRENT_BATCH->idx_bytes,"test sparse batch bytes"));
			assert(test_eq((int)(b.CURRENT_BATCH->rows +1)*sizeof(int),(int)b.CURRENT_BATCH->ptr_bytes,"test sparse batch bytes"));

			assert(test_eq(y->ptr_rows[index_rows + b.CURRENT_BATCH_Y->rows] - y->ptr_rows[index_rows],b.CURRENT_BATCH_Y->size,"test sparse batch size"));
			assert(test_eq((int)(y->ptr_rows[index_rows + b.CURRENT_BATCH_Y->rows] - y->ptr_rows[index_rows])*sizeof(float),(int)b.CURRENT_BATCH_Y->bytes,"test sparse batch bytes"));
			assert(test_eq((int)b.CURRENT_BATCH_Y->idx_bytes,(int)b.CURRENT_BATCH_Y->bytes,"test sparse batch bytes"));
			assert(test_eq((int)(y->ptr_rows[index_rows + b.CURRENT_BATCH_Y->rows] - y->ptr_rows[index_rows])*sizeof(int),(int)b.CURRENT_BATCH_Y->idx_bytes,"test sparse batch bytes"));
			assert(test_eq((int)(b.CURRENT_BATCH_Y->rows +1)*sizeof(int),(int)b.CURRENT_BATCH_Y->ptr_bytes,"test sparse batch bytes"));

			for(int j = 0; j < b.CURRENT_BATCH_Y->rows+1; j++)
			{
				assert(test_eq(X->ptr_rows[index_rows],s1->ptr_rows[j] + row_ptr_offset,"sparse batch allocator data test"));
				assert(test_eq(y->ptr_rows[index_rows],s2->ptr_rows[j]+ row_ptr_offset_y,"sparse batch allocator data test"));
				index_rows++;
			}
			index_rows--;
			row_ptr_offset += b.CURRENT_BATCH->size;
			row_ptr_offset_y += b.CURRENT_BATCH_Y->size;

			gpus.dot_sparse(b.CURRENT_BATCH, B, out);
			ASSERT(sum(out) > -15000 && sum(out) < 15000, "sparse batching sparse dot output test");


			if((i +1) == b.TOTAL_BATCHES)
				assert(test_eq(b.CURRENT_BATCH->rows,((int)ceil((X->rows*0.8))) % b.BATCH_SIZE,"after all sparse batches test: partial batch size"));

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

		assert(test_eq(index_rows+1,((int)ceil((X->rows*0.8))) +1,"after all sparse batches test: rows idx."));
		assert(test_eq(index_y,y->ptr_rows[((int)ceil((y->rows*0.8))) ],"after all sparse batches test: data idx y"));
		assert(test_eq(index,X->ptr_rows[((int)ceil((y->rows*0.8)))],"after all sparse batches test: data idx X"));

		for(int i = 0; i < b.TOTAL_BATCHES_CV; i++)
		{

			Matrix *s1 = to_host(b.CURRENT_BATCH_CV);
			Matrix *s2 = to_host(b.CURRENT_BATCH_CV_Y);
			Matrix *B = ones(b.CURRENT_BATCH_CV->cols,20);
			Matrix *out = zeros(b.CURRENT_BATCH_CV->rows, B->cols);

			b.broadcast_batch_cv_to_processes();

			for(int j = 0; j < b.CURRENT_BATCH_CV->size; j++)
			{
				assert(test_eq(X->data[index],s1->data[j],"sparse batch allocator data test"));
				assert(test_eq(X->idx_cols[index],s1->idx_cols[j],"sparse batch allocator data test"));
				index++;
			}


			for(int j = 0; j < b.CURRENT_BATCH_CV_Y->size; j++)
			{
				assert(test_eq(y->data[index_y],s2->data[j],"sparse batch allocator data test"));
				assert(test_eq(y->idx_cols[index_y],s2->idx_cols[j],"sparse batch allocator data test"));
				index_y++;
			}

			assert(test_eq(X->ptr_rows[index_rows + b.CURRENT_BATCH_CV->rows] - X->ptr_rows[index_rows],b.CURRENT_BATCH_CV->size,"test sparse batch size"));
			assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH_CV->rows] - X->ptr_rows[index_rows])*sizeof(float),(int)b.CURRENT_BATCH_CV->bytes,"test sparse batch bytes"));
			assert(test_eq((int)b.CURRENT_BATCH_CV->idx_bytes,(int)b.CURRENT_BATCH_CV->bytes,"test sparse batch bytes"));
			assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH_CV->rows] - X->ptr_rows[index_rows])*sizeof(int),(int)b.CURRENT_BATCH_CV->idx_bytes,"test sparse batch bytes"));
			assert(test_eq((int)(b.CURRENT_BATCH_CV->rows +1)*sizeof(int),(int)b.CURRENT_BATCH_CV->ptr_bytes,"test sparse batch bytes"));

			assert(test_eq(y->ptr_rows[index_rows + b.CURRENT_BATCH_CV_Y->rows] - y->ptr_rows[index_rows],b.CURRENT_BATCH_CV_Y->size,"test sparse batch size"));
			assert(test_eq((int)(y->ptr_rows[index_rows + b.CURRENT_BATCH_CV_Y->rows] - y->ptr_rows[index_rows])*sizeof(float),(int)b.CURRENT_BATCH_CV_Y->bytes,"test sparse batch bytes"));
			assert(test_eq((int)b.CURRENT_BATCH_CV_Y->idx_bytes,(int)b.CURRENT_BATCH_CV_Y->bytes,"test sparse batch bytes"));
			assert(test_eq((int)(y->ptr_rows[index_rows + b.CURRENT_BATCH_CV_Y->rows] - y->ptr_rows[index_rows])*sizeof(int),(int)b.CURRENT_BATCH_CV_Y->idx_bytes,"test sparse batch bytes"));
			assert(test_eq((int)(b.CURRENT_BATCH_CV_Y->rows +1)*sizeof(int),(int)b.CURRENT_BATCH_CV_Y->ptr_bytes,"test sparse batch bytes"));

			for(int j = 0; j < b.CURRENT_BATCH_CV_Y->rows+1; j++)
			{
				assert(test_eq(X->ptr_rows[index_rows],s1->ptr_rows[j] + row_ptr_offset,"sparse batch allocator data test"));
				assert(test_eq(y->ptr_rows[index_rows],s2->ptr_rows[j]+ row_ptr_offset_y,"sparse batch allocator data test"));
				index_rows++;
			}
			index_rows--;
			row_ptr_offset += b.CURRENT_BATCH_CV->size;
			row_ptr_offset_y += b.CURRENT_BATCH_CV_Y->size;

			gpus.dot_sparse(b.CURRENT_BATCH_CV, B, out);
			ASSERT(sum(out) > -25000 && sum(out) < 25000, "sparse batching sparse dot output test");


			if((i +1) == b.TOTAL_BATCHES_CV)
				assert(test_eq(b.CURRENT_BATCH_CV->rows,(X->rows - (int)ceil((X->rows*0.8))) % b.BATCH_SIZE_CV,"after all sparse batches test: partial batch size"));


			b.allocate_next_cv_batch_async();
			b.replace_current_cv_batch_with_next();


			cudaFree(s1->data);
			cudaFree(s1->idx_cols);
			cudaFree(s1->ptr_rows);
			free(s1);
			cudaFree(B->data);
			cudaFree(out->data);
			free(out);
			free(B);
		}

		assert(test_eq(index_rows+1,X->rows  +1,"after all sparse batches test: rows idx."));
		assert(test_eq(index_y,y->ptr_rows[y->rows ],"after all sparse batches test: data idx y"));
		assert(test_eq(index,X->ptr_rows[y->rows],"after all sparse batches test: data idx X"));
	}

	if(gpus.MYGPUID != 0)
	{
		X = empty_sparse(1,1,1);
		y = empty_sparse(1,1,1);
	}

	b = BatchAllocator();
	b.init(X,y,0.20,33,77,gpus, Distributed_weights_sparse);
	assert(test_eq(b.CURRENT_BATCH->rows,33,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH->cols,9000,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_Y->rows,33,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_Y->cols,24,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV->rows,77,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV->cols,9000,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV_Y->rows,77,"sparse distributed batch allocator test"));
	assert(test_eq(b.CURRENT_BATCH_CV_Y->cols,24,"sparse distributed batch allocator test"));


	for(int epoch = 0; epoch < 3; epoch++)
	{
		index_rows = 0;
		index = 0;
		index_y = 0;
		row_ptr_offset = 0;
		row_ptr_offset_y = 0;
		for(int i = 0; i < b.TOTAL_BATCHES; i++)
		{
			Matrix *B = ones(b.CURRENT_BATCH->rows,20);
			Matrix *out = zeros(b.CURRENT_BATCH->cols, B->cols);

			b.broadcast_batch_to_processes();
			if(gpus.MYGPUID == 0)
			{
				Matrix *s1 = to_host(b.CURRENT_BATCH);
				Matrix *s2 = to_host(b.CURRENT_BATCH_Y);

				Matrix *m3 = gpus.sparse_to_dense(b.CURRENT_BATCH);
				Matrix *s3 = to_host((gpus.dense_to_sparse(m3)));


				//cout << sum(m3) << " vs " << sum(s3) << endl;


				for(int i = 0; i < s3->size; i++)
				{
					cout << s1->idx_cols[i] << " vs " << s3->idx_cols[i] << endl;
					//assert(test_eq(s1->data[i],s3->data[i],"dense to sparse and back equality."));
					//assert(test_eq(s1->idx_cols[i],s3->idx_cols[i],"dense to sparse and back equality."));
				}

				cout << "size: " << s1->size << endl;
				for(int i = 0; i < s3->rows+1; i++)
					cout << s1->ptr_rows[i] << " vs " << s3->ptr_rows[i] << endl;
					//assert(test_eq(s1->ptr_rows[i],s3->ptr_rows[i],"dense to sparse and back equality."));

				for(int j = 0; j < b.CURRENT_BATCH->size; j++)
				{
					assert(test_eq(X->data[index],s1->data[j],"sparse batch allocator data test"));
					assert(test_eq(X->idx_cols[index],s1->idx_cols[j],"sparse batch allocator data test"));
					index++;
				}

				for(int j = 0; j < b.CURRENT_BATCH_Y->size; j++)
				{
					assert(test_eq(y->data[index_y],s2->data[j],"sparse batch allocator data test"));
					assert(test_eq(y->idx_cols[index_y],s2->idx_cols[j],"sparse batch allocator data test"));
					index_y++;
				}

				assert(test_eq(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows],b.CURRENT_BATCH->size,"test sparse batch size"));
				assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows])*sizeof(float),(int)b.CURRENT_BATCH->bytes,"test sparse batch bytes"));
				assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH->rows] - X->ptr_rows[index_rows])*sizeof(int),(int)b.CURRENT_BATCH->idx_bytes,"test sparse batch bytes"));
				assert(test_eq((int)(b.CURRENT_BATCH->rows +1)*sizeof(int),(int)b.CURRENT_BATCH->ptr_bytes,"test sparse batch bytes"));
				for(int j = 0; j < b.CURRENT_BATCH->rows+1; j++)
				{
					assert(test_eq(X->ptr_rows[index_rows],s1->ptr_rows[j] + row_ptr_offset,"sparse batch allocator data test"));
					assert(test_eq(y->ptr_rows[index_rows],s2->ptr_rows[j]+ row_ptr_offset_y,"sparse batch allocator data test"));
					index_rows++;
				}
				index_rows--;
				row_ptr_offset += b.CURRENT_BATCH->size;
				row_ptr_offset_y += b.CURRENT_BATCH_Y->size;

				cudaFree(s1->data);
				cudaFree(s1->idx_cols);
				cudaFree(s1->ptr_rows);
				free(s1);

				cudaFree(s2->data);
				cudaFree(s2->idx_cols);
				cudaFree(s2->ptr_rows);
				free(s2);
			}

			cout << "pre Tdot" << endl;
			gpus.Tdot_sparse(b.CURRENT_BATCH, B, out);
			cout << "post Tdot" << endl;
			cout << sum(out) << endl;
			//ASSERT(sum(out) > -3000 && sum(out) < 3000, "sparse batching sparse dot output test");

			b.allocate_next_batch_async();
			b.replace_current_batch_with_next();

			cudaFree(B->data);
			cudaFree(out->data);
			free(out);
			free(B);
		}

		for(int i = 0; i < b.TOTAL_BATCHES_CV; i++)
		{
			Matrix *B = ones(b.CURRENT_BATCH_CV->cols,20);
			Matrix *out = zeros(b.CURRENT_BATCH_CV->rows, B->cols);

			b.broadcast_batch_cv_to_processes();

			if(gpus.MYGPUID == 0)
			{
				Matrix *s1 = to_host(b.CURRENT_BATCH_CV);
				Matrix *s2 = to_host(b.CURRENT_BATCH_CV_Y);

				for(int j = 0; j < b.CURRENT_BATCH_CV->size; j++)
				{
					assert(test_eq(X->data[index],s1->data[j],"sparse batch allocator data test"));
					assert(test_eq(X->idx_cols[index],s1->idx_cols[j],"sparse batch allocator data test"));
					index++;
				}


				for(int j = 0; j < b.CURRENT_BATCH_CV_Y->size; j++)
				{
					assert(test_eq(y->data[index_y],s2->data[j],"sparse batch allocator data test"));
					assert(test_eq(y->idx_cols[index_y],s2->idx_cols[j],"sparse batch allocator data test"));
					index_y++;
				}

				assert(test_eq(X->ptr_rows[index_rows + b.CURRENT_BATCH_CV->rows] - X->ptr_rows[index_rows],b.CURRENT_BATCH_CV->size,"test sparse batch size"));
				assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH_CV->rows] - X->ptr_rows[index_rows])*sizeof(float),(int)b.CURRENT_BATCH_CV->bytes,"test sparse batch bytes"));
				assert(test_eq((int)b.CURRENT_BATCH_CV->idx_bytes,(int)b.CURRENT_BATCH_CV->bytes,"test sparse batch bytes"));
				assert(test_eq((int)(X->ptr_rows[index_rows + b.CURRENT_BATCH_CV->rows] - X->ptr_rows[index_rows])*sizeof(int),(int)b.CURRENT_BATCH_CV->idx_bytes,"test sparse batch bytes"));
				assert(test_eq((int)(b.CURRENT_BATCH_CV->rows +1)*sizeof(int),(int)b.CURRENT_BATCH_CV->ptr_bytes,"test sparse batch bytes"));

				assert(test_eq(y->ptr_rows[index_rows + b.CURRENT_BATCH_CV_Y->rows] - y->ptr_rows[index_rows],b.CURRENT_BATCH_CV_Y->size,"test sparse batch size"));
				assert(test_eq((int)(y->ptr_rows[index_rows + b.CURRENT_BATCH_CV_Y->rows] - y->ptr_rows[index_rows])*sizeof(float),(int)b.CURRENT_BATCH_CV_Y->bytes,"test sparse batch bytes"));
				assert(test_eq((int)b.CURRENT_BATCH_CV_Y->idx_bytes,(int)b.CURRENT_BATCH_CV_Y->bytes,"test sparse batch bytes"));
				assert(test_eq((int)(y->ptr_rows[index_rows + b.CURRENT_BATCH_CV_Y->rows] - y->ptr_rows[index_rows])*sizeof(int),(int)b.CURRENT_BATCH_CV_Y->idx_bytes,"test sparse batch bytes"));
				assert(test_eq((int)(b.CURRENT_BATCH_CV_Y->rows +1)*sizeof(int),(int)b.CURRENT_BATCH_CV_Y->ptr_bytes,"test sparse batch bytes"));

				for(int j = 0; j < b.CURRENT_BATCH_CV_Y->rows+1; j++)
				{
					assert(test_eq(X->ptr_rows[index_rows],s1->ptr_rows[j] + row_ptr_offset,"sparse batch allocator data test"));
					assert(test_eq(y->ptr_rows[index_rows],s2->ptr_rows[j]+ row_ptr_offset_y,"sparse batch allocator data test"));
					index_rows++;
				}
				index_rows--;
				row_ptr_offset += b.CURRENT_BATCH_CV->size;
				row_ptr_offset_y += b.CURRENT_BATCH_CV_Y->size;


				if((i +1) == b.TOTAL_BATCHES_CV)
					assert(test_eq(b.CURRENT_BATCH_CV->rows,(X->rows - (int)ceil((X->rows*0.8))) % b.BATCH_SIZE_CV,"after all sparse batches test: partial batch size"));

				cudaFree(s1->data);
				cudaFree(s1->idx_cols);
				cudaFree(s1->ptr_rows);
				free(s1);
			}


			gpus.dot_sparse(b.CURRENT_BATCH_CV, B, out);
			ASSERT(sum(out) > -8000 && sum(out) < 8000, "sparse batching sparse dot output test");




			b.allocate_next_cv_batch_async();
			b.replace_current_cv_batch_with_next();


			cudaFree(B->data);
			cudaFree(out->data);
			free(out);
			free(B);
		}
	}



  return 0;
}




