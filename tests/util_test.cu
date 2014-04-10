#include <util.cuh>
#include <stdio.h>
#include <assert.h>
#include <string>
#include <iostream>

using std::cout;
using std::endl;

void run_util_test()
{
	char buff[1024];
	ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
	std::string path = std::string(buff);
	replace(path,"/build/testSuite.out","/tests/");

	Matrix *X = read_hdf5((path + "/numpy_arange_as_h5py.hdf5").c_str());
	for(int i = 0;i < X->size; i++)
		assert(test_eq(X->data[i],(float)i,"HDF5 read for h5py data."));

	X = read_sparse_hdf5((path + "/scipy_sparse_arange_as_h5py.hdf5").c_str());
	for(int i = 0;i < X->size; i++)
		assert(test_eq(X->data[i],(float)(i+1),"HDF5 read sparse for h5py data."));

	int col_count = 1;
	for(int i = 0;i < 105; i++)
	{
		assert(test_eq(X->idx_cols[i],col_count,"HDF5 read sparse for h5py data."));
		col_count++;
		if(col_count == 50)
			col_count = 0;

	}

	int row_ptr = 0;
	for(int i = 0;i < X->rows-1; i++)
	{
		assert(test_eq(X->ptr_rows[i],row_ptr,"HDF5 read sparse for h5py data."));
		row_ptr += i == 0 ? 49 : 50;
	}

	ASSERT(determine_max_sparsity(X,X->rows) == (float)((X->rows*X->cols)-1)/(float)(X->rows*X->cols),"max sparsity test");

	Matrix *out = empty_pinned(X->rows,X->cols);
	slice_sparse_to_dense(X,out,0,X->rows);
	for(int i = 0;i < out->size; i++)
		assert(test_eq(out->data[i],(float)i,"slice sparse to dense test."));





}
