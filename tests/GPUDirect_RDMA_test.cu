#include <stdio.h>
#include <mpi.h>
#include <assert.h>

__global__ void times_two(float *data, long size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {

       data[i] *= 2;
  }
}

__global__ void times_three(float *data, long size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx; i < size; i += numThreads)
  {
        data[i] *= 3;
  }
}

int main(int argc, char *argv[])
{
    int myrank;
    MPI_Status status;
    float *pdata;
    float data[10]={0,1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9};

    const int size =10;
    const size_t bytes = size*sizeof(float);
    const int grid_size = 1024;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    cudaMalloc((void**) &pdata,bytes);
	    if(myrank == 0)
      {
		    cudaMemcpy(pdata, data, bytes, cudaMemcpyDefault);
	      times_two<<<grid_size, 1024>>>(pdata, size);
		    MPI_Send(pdata, size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
	    }
	    else
	    {
	 	    MPI_Recv(pdata, size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
        times_three<<<grid_size,1024>>>(pdata, size);
      }

        float *output;
        output = (float*)malloc(bytes);
        cudaMemcpy(output,pdata,bytes,cudaMemcpyDefault);

        for(int i = 0; i< 10; i++)
        {
          if(myrank == 0)
          {
            assert(data[i]*2==output[i]);
          }
          else
          {
            assert(data[i]*6==output[i]);
          }
        }


       MPI_Finalize();

    return 0;
}


