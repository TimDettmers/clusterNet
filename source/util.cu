#include <stdio.h>
#include <clusterNet.cuh>
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

void tock(cudaEvent_t* startstop)
{
    float time;
    cudaEventRecord(startstop[1], 0);
    cudaEventSynchronize(startstop[1]);   
    cudaEventElapsedTime(&time, startstop[0], startstop[1]);
    printf ("Time for the kernel: %f ms\n", time);
}

