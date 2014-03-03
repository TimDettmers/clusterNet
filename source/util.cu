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

using std::string;
using std::vector;

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
  data = (float*)malloc(columns*rows*sizeof(float));
  memcpy(data,&X[0], columns*rows*sizeof(float));

  Matrix *out = (Matrix*)malloc(sizeof(Matrix));
  out->rows = rows;
  out->cols = columns;
  out->bytes = columns*rows*sizeof(float);
  out->size = columns*rows;
  out->data = data;

  return out;
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

void print_matrix(Matrix *A)
{
	for(int row = 0; row< A->rows; row++)
	  {
		  printf("[");
		  for(int col =0; col < A->cols; col++)
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
  print_matrix(m);
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


Matrix *read_hdf5(const char * filepath)
{
	   hid_t       file_id, dataset_id;

	   file_id = H5Fopen(filepath, H5F_ACC_RDWR, H5P_DEFAULT);
	   dataset_id = H5Dopen2(file_id, "/Default", H5P_DEFAULT);

	   hid_t dspace = H5Dget_space(dataset_id);
	   hsize_t dims[2];
	   H5Sget_simple_extent_dims(dspace, dims, NULL);
	   size_t bytes = sizeof(float)*dims[0]*dims[1];
	   float *data = (float*)malloc(bytes);

	   H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	   H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

	   //status = H5Dread (dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,  &readBuf[0]);
	   H5Dclose(dataset_id);
	   H5Fclose(file_id);

	   Matrix *out = (Matrix*)malloc(sizeof(Matrix));
	   out->rows = dims[0];
	   out->cols= dims[1];
	   out->bytes = bytes;
	   out->data = data;
	   out->size = dims[0]*dims[1];

	   return out;
}




