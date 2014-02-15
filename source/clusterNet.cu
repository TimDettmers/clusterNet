#include <stdio.h>
#include <clusterNet.cuh>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <list>
#include <vector>

typedef struct Matrix
{
  const int shape[2];
  const size_t bytes;
  const int size;
  const float *data;
} Matrix;

using namespace std;

Matrix read_csv (char* filename)
{
  ifstream  dStream(filename);
  int dimX = 0;
  int dimY = 0;
  vector<float> X;

    string line;
    while(std::getline(dStream,line))
    {
        stringstream  lineStream(line);
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


Matrix allocate(Matrix m)
{
  float * gpu_data;
  cudaMalloc((void**)&gpu_data,m.bytes);
  cudaMemcpy(gpu_data,m.data,m.bytes,cudaMemcpyDefault);
  Matrix gpu_matrix = {{m.shape[0],m.shape[1]},m.bytes,m.size,gpu_data};

  return gpu_matrix;
}




int main(int argc, char *argv[])
{
  Matrix h_y = read_csv("/home/tim/Downloads/mnist_full_y.csv");
  Matrix h_X = read_csv("/home/tim/Downloads/mnist_full_X.csv");

  Matrix d_y = allocate(h_y);
  Matrix d_X = allocate(h_X);

  

}
