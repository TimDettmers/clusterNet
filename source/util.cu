#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <list>
#include <vector>
#include <util.cuh>
#include <basicOps.cuh>
#include <hdf5.h>
#include <iomanip>
#include <ios>
#include <assert.h>

using std::string;
using std::vector;
using std::cout;
using std::endl;

Matrix *read_csv (const char* filename)
{
  std::ifstream  dStream(filename);
  int columns = 0;
  int rows = 0;
  vector<float> X;

    string line;
    while(std::getline(dStream,line))
    {
        std::stringstream  lineStream(line);
        string        cell;
        while(std::getline(lineStream,cell,','))
        {
			X.push_back(::atof(cell.c_str()));

			if(rows == 0)
				columns++;
        }
	rows++;
    }

  float *data;
  size_t bytes = columns*rows*sizeof(float);
  cudaHostAlloc(&data, bytes, cudaHostAllocPortable);
  memcpy(data,&X[0], columns*rows*sizeof(float));

  Matrix *out = (Matrix*)malloc(sizeof(Matrix));
  out->rows = rows;
  out->cols = columns;
  out->bytes = bytes;
  out->size = columns*rows;
  out->data = data;
  out->isDistributed = 0;
  out->cols_distributed = 0;
  out->isSparse = 0;

  return out;
}

void write_csv(const char* filename, Matrix *X, const char* header, Matrix *ids)
{
	std::ofstream myfile;
	myfile.open(filename,std::ios::trunc);
	myfile << header << "\r\n";
	for(int row = 0; row< X->rows; row++)
	  {
		  for(int col = 0; col < X->cols; col++)
		  {
			  if(col > 0)
				  myfile << ",";
			  else
				  myfile << (int)ids->data[row] << ",";

			  myfile << std::fixed << X->data[(row*X->cols)+col];
		  }
		  myfile << "\r\n";
	  }
	myfile.close();
}

void write_csv(const char* filename, Matrix *X)
{
	std::ofstream myfile;
	myfile.open(filename,std::ios::trunc);
	for(int row = 0; row< X->rows; row++)
	  {
		  for(int col = 0; col < X->cols; col++)
		  {
			  if(col > 0)
				  myfile << ",";

			  myfile << std::fixed << X->data[(row*X->cols)+col];
		  }
		  myfile << "\r\n";
	  }
	myfile.close();
}

Matrix *read_hdf5(const char *filepath){ return read_hdf5(filepath,"/Default"); }
Matrix *read_hdf5(const char *filepath, const char *tag)
{
	   hid_t       file_id, dataset_id;

	   file_id = H5Fopen(filepath, H5F_ACC_RDWR, H5P_DEFAULT);
	   dataset_id = H5Dopen2(file_id, tag, H5P_DEFAULT);

	   hid_t dspace = H5Dget_space(dataset_id);
	   hsize_t dims[2];
	   H5Sget_simple_extent_dims(dspace, dims, NULL);
	   size_t bytes = sizeof(float)*dims[0]*dims[1];

	   float *data;
	   cudaHostAlloc(&data, bytes, cudaHostAllocPortable);

	   H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	   H5Dclose(dataset_id);
	   H5Fclose(file_id);

	   Matrix *out = (Matrix*)malloc(sizeof(Matrix));
	   out->rows = (int)dims[0];
	   out->cols= (int)dims[1];
	   out->bytes = bytes;
	   out->data = data;
	   out->size = (int)(dims[0]*dims[1]);
	   out->isDistributed = 0;
	   out->cols_distributed = 0;
	   out->isSparse = 0;

	   return out;
}

Matrix *read_sparse_hdf5(const char *filepath)
{
	hid_t       file_id, dataset_id_idx, dataset_id_ptr, dataset_id_data, dataset_id_shape, dspace;
	hsize_t dims[2];
	size_t bytes;
	file_id = H5Fopen(filepath, H5F_ACC_RDWR, H5P_DEFAULT);
	Matrix *out = (Matrix*)malloc(sizeof(Matrix));

	dataset_id_idx = H5Dopen2(file_id, "/indices", H5P_DEFAULT);
	dspace = H5Dget_space(dataset_id_idx);
	H5Sget_simple_extent_dims(dspace, dims, NULL);
	bytes = sizeof(int)*dims[0];
	int *idx;
	cudaHostAlloc(&idx, bytes, cudaHostAllocPortable);
	H5Dread(dataset_id_idx, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, idx);
	H5Dclose(dataset_id_idx);

	out->idx_bytes = sizeof(int)*dims[0];
	out->idx_cols = idx;


	dataset_id_ptr = H5Dopen2(file_id, "/indptr", H5P_DEFAULT);
	dspace = H5Dget_space(dataset_id_ptr);
	H5Sget_simple_extent_dims(dspace, dims, NULL);
	bytes = sizeof(int)*dims[0];
	int *ptr;
	cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable);
	H5Dread(dataset_id_ptr, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);
	H5Dclose(dataset_id_ptr);

	out->ptr_bytes = sizeof(int)*dims[0];
	out->ptr_rows = ptr;


	dataset_id_data = H5Dopen2(file_id, "/data", H5P_DEFAULT);
	dspace = H5Dget_space(dataset_id_data);
	H5Sget_simple_extent_dims(dspace, dims, NULL);
	bytes = sizeof(float)*dims[0];
	float *data;
	cudaHostAlloc(&data, bytes, cudaHostAllocPortable);
	H5Dread(dataset_id_data, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	H5Dclose(dataset_id_data);

	out->bytes = sizeof(float)*dims[0];
	out->size = (int)dims[0];

	dataset_id_shape = H5Dopen2(file_id, "/shape", H5P_DEFAULT);
	dspace = H5Dget_space(dataset_id_shape);
	H5Sget_simple_extent_dims(dspace, dims, NULL);
	bytes = sizeof(long)*dims[0];
	long shape[2];
	H5Dread(dataset_id_shape, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, shape);
	H5Dclose(dataset_id_shape);

	H5Fclose(file_id);


	out->rows = (int)shape[0];
	out->cols= (int)shape[1];
	out->data = data;
	out->isDistributed = 0;
	out->isSparse = 1;




	return out;
}

void write_hdf5(const char * filepath, Matrix *A)
{
	   hid_t       file_id, dataset_id, dataspace_id;
	   hsize_t     dims[2];

	   file_id = H5Fcreate(filepath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	   dims[0] = A->rows;
	   dims[1] = A->cols;
	   dataspace_id = H5Screate_simple(2, dims, NULL);
	   dataset_id = H5Dcreate2(file_id, "/Default", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	   H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, A->data);
	   H5Dclose(dataset_id);
	   H5Fclose(file_id);
}

cudaEvent_t* tick()
{
    cudaEvent_t* startstop;
    startstop = (cudaEvent_t*)malloc(2*sizeof(cudaEvent_t));
    cudaEventCreate(&startstop[0]);
    cudaEventCreate(&startstop[1]);
    cudaEventRecord(startstop[0], 0);

    return startstop;
}

void tock(cudaEvent_t* startstop){ tock(startstop, "Time for the kernel(s): "); }
void tock(cudaEvent_t* startstop, std::string text)
{
	float time;
	cudaEventRecord(startstop[1], 0);
	cudaEventSynchronize(startstop[1]);
	cudaEventElapsedTime(&time, startstop[0], startstop[1]);
	printf((text + ": %f ms.\n").c_str(), time);
}
void tock(std::string text, float tocks)
{
	printf((text + ": %f ms.\n").c_str(), tocks);
}
float tock(cudaEvent_t* startstop, float tocks)
{
	float time;
	cudaEventRecord(startstop[1], 0);
	cudaEventSynchronize(startstop[1]);
	cudaEventElapsedTime(&time, startstop[0], startstop[1]);

	return time+tocks;
}



int test_eq(float f1, float f2, char* message)
{
  if(f1 == f2){ return 1;}
  else{ printf("%s: %f != %f\n", message, f1, f2); }
  return 0;
}

int test_eq(float f1, float f2, int idx1, int idx2, char* message)
{
  if(f1 == f2){ return 1;}
  else{ printf("%s: %f != %f for index %i and %i.\n", message, f1, f2, idx1, idx2); }
  return 0;
}

int test_eq(int i1, int i2, char* message)
{
  if(i1 == i2){ return 1;}
  else{ printf("%s: %i != %i\n", message, i1, i2); }
  return 0;
}

int test_eq(int i1, int i2, int idx1, int idx2, char* message)
{
  if(i1 == i2){ return 1;}
  else{ printf("%s: %i != %i for index %i and %i.\n", message, i1, i2, idx1, idx2); }
  return 0;
}

int test_matrix(Matrix *A, int rows, int cols)
{
  if((A->rows == rows) &&
     (A->cols == cols) &&
     (A->size == cols*rows) &&
     (A->bytes == cols*rows*sizeof(float)))
      {return 1;}
  else
  {
    test_eq(A->rows,rows,"Matrix rows");
    test_eq(A->cols,cols,"Matrix cols");
    test_eq(A->size,cols*rows,"Matrix size");
    test_eq((int)(A->bytes),(int)(cols*rows*sizeof(float)),"Matrix bytes");
  }

  return 0;
}

void print_matrix(Matrix *A, int end_rows, int end_cols)
{
	if(A->isSparse != 1)
	{
		for(int row = 0; row< end_rows; row++)
		  {
			  printf("[");
			  for(int col =0; col < end_cols; col++)
			  {
				  printf("%f ",A->data[(row*A->cols)+col]);
			  }
			  printf("]\n");
		  }
		  printf("\n");
	}
	else
	{
		printf("[");
		for(int i = end_rows; i < end_cols; i++)
			printf("%f ",A->data[i]);

		printf("]\n");
	}
}

void print_matrix(Matrix *A, int start_row, int end_row, int start_col, int end_col)
{
	assert(A->isSparse == 0);

	for(int row = start_row; row< end_row; row++)
	{
		printf("[");
		for(int col =start_col; col < end_col; col++)
		{
			printf("%f ",A->data[(row*A->cols)+col]);
		}
		printf("]\n");
	}
	printf("\n");

}

void printmat(Matrix *A)
{
  Matrix * m = to_host(A);
  if(A->isSparse == 0)
	  print_matrix(m,A->rows,A->cols);
  else
	  print_matrix(m,0,A->size);
  free(m->data);
  free(m);

}

void printdim(Matrix *A)
{
	cout << A->rows << "x" << A->cols << endl;
}

void printsum(Matrix *A)
{
	cout << sum(A) << endl;
}

void printhostmat(Matrix *A)
{
  if(A->isSparse == 0)
	  print_matrix(A,A->rows,A->cols);
  else
	  print_matrix(A,0,A->size);
}

void printmat(Matrix *A, int end_rows, int end_cols)
{
  Matrix * m = to_host(A);
  print_matrix(m, end_rows, end_cols);
  free(m->data);
  free(m);

}

void printmat(Matrix *A, int start_row, int end_row, int start_col, int end_col)
{
  Matrix * m = to_host(A);
  print_matrix(m, start_row, end_row, start_col, end_col);
  free(m->data);
  free(m);

}

bool replace(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

void slice_sparse_to_dense(Matrix *X, Matrix *out, int start, int length)
{
	int idx_from = 0;
	int idx_to = 0;
	int idx = 0;

	for(int i = 0; i < out->size; i++)
		out->data[i] = 0.0f;

	for(int row = 0; row < length; row++)
	{
		idx_from = X->ptr_rows[start + row];
		idx_to = X->ptr_rows[start + row + 1];

		for(int i = idx_from; i < idx_to; i++)
		{
			idx = X->idx_cols[i];
			out->data[(row*out->cols) + idx] = X->data[i];
		}
	}



}

float determine_max_sparsity(Matrix *X, int batch_size)
{

	float max_sparsity = 0.0;

	Matrix *dense_batch = empty_cpu(batch_size,X->cols);
	int batches = (X->rows / batch_size);
	float batch_elements = batch_size*X->cols;

	float nonzero_count = 0.0f;
	for(int i = 0; i < batches; i++)
	{
		nonzero_count = (X->ptr_rows[(i+1)*batch_size] - X->ptr_rows[i*batch_size]);

		if(max_sparsity < (nonzero_count / batch_elements))
			max_sparsity = (nonzero_count / batch_elements);

		nonzero_count = 0.0f;
	}

	return max_sparsity;

}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}




