#include <stdio.h>
#include <clusterNet.cuh>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <list>
#include <vector>
#include <cublas_v2.h>
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

Matrix dot(Matrix A, Matrix B)
{
	float *output;
	const int out_rows = A.shape[0];
	const int out_cols = B.shape[1];
	const int ARRAY_BYTES = sizeof(float)*A.shape[0] *B.shape[1];
	cudaMalloc((void**) &output, ARRAY_BYTES);	
	
	cublasStatus_t status;
	
	const float alpha = 1.0f;
	const float beta = 0.0f;

	//cublas
	cublasHandle_t h;
        cublasCreate(&h);
      
    
    status = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 
                A.shape[0], B.shape[1], A.shape[1],
                &alpha, A.data, A.shape[0],
                B.data, B.shape[0],
                &beta, output, out_rows);
    
                
   if(status != CUBLAS_STATUS_SUCCESS)
   		printf("CUBLAS ERROR!");
	
	Matrix ret = {{out_rows,out_cols},ARRAY_BYTES,out_rows*out_cols,output};
	return ret;
}
