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

using std::string;
using std::vector;

Matrix read_csv (char* filename)
{
  std::ifstream  dStream(filename);
  int dimX = 0;
  int dimY = 0;
  vector<float> X;

    string line;
    while(std::getline(dStream,line))
    {
        std::stringstream  lineStream(line);
        string        cell;
        while(std::getline(lineStream,cell,','))
        {
	    X.push_back(::atof(cell.c_str()));
	    
	if(dimY == 0)
	    dimX++;
        }
	dimY++;
    }

  
  float *data;  
  data = (float*)malloc(dimX*dimY*sizeof(float));
  memcpy(data,&X[0], dimX*dimY*sizeof(float));
  Matrix m = {{dimX,dimY},dimX*dimY*sizeof(float),dimX*dimY,data};  

  return m;
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

int test_matrix(Matrix A, int rows, int cols)
{
  if((A.shape[0] == rows) &&
     (A.shape[1] == cols) &&
     (A.size == cols*rows) &&
     (A.bytes == cols*rows*sizeof(float)))
      {return 1;}
  else
  {
    test_eq(A.shape[0],rows,"Matrix rows");
    test_eq(A.shape[1],cols,"Matrix cols");
    test_eq(A.size,cols*rows,"Matrix size");
    test_eq((int)(A.bytes),(int)(cols*rows*sizeof(float)),"Matrix bytes");
  }

  return 0;
}

void print_matrix(Matrix A)
{
  for(int i = 0; i< A.size; i++)
  {
    printf("%f\n",A.data[i]);
  }
}

void print_gpu_matrix(Matrix A)
{
  float *data = (float*)malloc(A.bytes);
  cudaMemcpy(data,A.data,A.bytes,cudaMemcpyDefault);
  for(int i = 0; i< A.size; i++)
  {
    printf("%f\n",data[i]);
  }
  cudaFree(data);
  free(data);
}



