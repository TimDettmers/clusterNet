#include <util.cuh>
#include <stdio.h>
#include <assert.h>
#include <string>


void run_util_test()
{
	char buff[1024];
	ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
	std::string path = std::string(buff);
	replace(path,"/build/testSuite.out","/tests/");

	Matrix *X = read_hdf5((path + "/numpy_arange_as_h5py.hdf5").c_str());
	for(int i = 0;i < X->size; i++)
		assert(test_eq(X->data[i],(float)i,"HDF5 read for h5py data."));
}
