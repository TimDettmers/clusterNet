#include <cublas_v2.h>
#include <clusterNet.h>
#include <basicOps.cuh>
#include <util.cuh>
#include <cstdlib>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <pthread.h>


ClusterNet::ClusterNet(){ init((int)(time(0) % 10000)); }
ClusterNet::ClusterNet(int seed){ init(seed);}
ClusterNet::ClusterNet(int argc, char* argv[], int seed){ init(seed); init_MPI(argc, argv); }
void ClusterNet::init(int seed)
{
	/*
	 * times for 1bn rand numbers
	- CURAND_RNG_PSEUDO_DEFAULT 135/144 ms
	 * - CURAND_RNG_PSEUDO_XORWOW 135/144 ms
	 * - CURAND_RNG_PSEUDO_MRG32K3A 270/310 ms
	 * - CURAND_RNG_PSEUDO_MTGP32 230/235 ms
	 * - CURAND_RNG_PSEUDO_PHILOX4_32_10 130/135 ms //different numbers with same seed?
	 * - CURAND_RNG_QUASI_DEFAULT 140/156 ms
	 * - CURAND_RNG_QUASI_SOBOL32 140/156 ms //correlated adjacent values
	 *
	 *
	 * */
	curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_generator, seed);
	curandSetGeneratorOffset(m_generator, 100);
	cublasCreate(&m_handle);
	m_hasMPI = false;


}



void ClusterNet::init_MPI(int argc, char * argv[])
{
	int myrank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int gpus;
	cudaGetDeviceCount(&gpus);
	int mygpu_id;
	int your_gpu_id;
	if(myrank == 0)
	{
		mygpu_id = 0;
		if(gpus > 1)
			your_gpu_id = 1;
		else
			your_gpu_id = 0;

		MPI_Send(&your_gpu_id,1, MPI_INT,1,0,MPI_COMM_WORLD);
	}
	else
	{
		MPI_Recv(&mygpu_id,1,MPI_INT,myrank-1,0,MPI_COMM_WORLD,&m_status);
		if(gpus > mygpu_id+1)
			your_gpu_id = mygpu_id + 1;
		else
			your_gpu_id = 0;
		if(myrank < size-1)
			MPI_Send(&your_gpu_id,1, MPI_INT,myrank+1,0,MPI_COMM_WORLD);
	}

	cudaSetDevice(mygpu_id);
	m_nodes = size;
	m_hasMPI = true;
	m_myrank = myrank;

	std::cout << "there are " << m_nodes << " processes and I work on gpu device " << mygpu_id << std::endl;

}

void ClusterNet::shutdown_MPI()
{
	MPI_Finalize();
}

Matrix *ClusterNet::dot(Matrix *A, Matrix *B)
{
	Matrix *out = zeros(A->rows,B->cols);
	dot(A, B, out);

	return out;
}

Matrix *ClusterNet::Tdot(Matrix *A, Matrix *B)
{
	//if(m_hasMPI){ return dotMPI(A,B);}
	Matrix *out = zeros(A->cols,B->cols);
	Tdot(A, B, out);

	return out;
}

Matrix *ClusterNet::dotT(Matrix *A, Matrix *B)
{
	//if(m_hasMPI){ return dotMPI(A,B);}
	Matrix *out = zeros(A->rows,B->rows);
	dotT(A, B, out);

	return out;
}

void ClusterNet::dotT(Matrix *A, Matrix *B, Matrix *out){ dot(A,B,out,CUBLAS_OP_N, CUBLAS_OP_T); }
void ClusterNet::Tdot(Matrix *A, Matrix *B, Matrix *out){ dot(A,B,out,CUBLAS_OP_T, CUBLAS_OP_N); }
void ClusterNet::dot(Matrix *A, Matrix *B, Matrix *out){ dot(A,B,out,CUBLAS_OP_N, CUBLAS_OP_N); }
void ClusterNet::dot(Matrix *A, Matrix *B, Matrix *out, cublasOperation_t T1, cublasOperation_t T2)
{
	//if(checkMatrixOperation(A, B, out, 1) == 1){ throw "Matrix *size error:\n"; }
	cublasStatus_t status;

	const float alpha = 1.0f;
	const float beta = 0.0f;
	int A_rows = A->rows, B_rows = B->rows, A_cols = A->cols, B_cols = B->cols;
	if(T1 == CUBLAS_OP_T){ A_rows = A->cols; A_cols = A->rows; }
	if(T2 == CUBLAS_OP_T){ B_rows = B->cols; B_cols = B->rows; }

	status = cublasSgemm(m_handle, T1, T2,
				A_rows, B_cols, A_cols,
				&alpha, A->data, A->rows,
				B->data, B->rows,
				&beta, out->data, out->rows);

	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "CUBLAS ERROR: Status " << status << std::endl;
		throw "CUBLAS ERROR";
	}
}

Matrix *ClusterNet::dotMPI_batchSlice(Matrix *A, Matrix *B)
{
	int split_size = A->rows/m_nodes;
	Matrix *out = empty(split_size,B->cols);
	Matrix *out_rev = empty(split_size,B->cols);

	tick("slice batch");
	Matrix *A1 = slice_rows(A, split_size*m_myrank,split_size*(m_myrank+1)-1);
	tick("slice batch");
	tick("dot batch");
	dot(A1,B,out);
	tick("dot batch");
	for(int i = 0; i < m_nodes; i++)
	{
		if(m_myrank == i) { continue; }
		MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request));
		tick("send batch");
		MPI_Isend(out->data, out->size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, request);
		tick("send batch");
	}

	for(int i = 0; i < m_nodes; i++)
	{
		if(m_myrank == i) { continue; }
		tick("receive batch");
		MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request));
		//m_receiveRequests[i].push_back(request);
		MPI_Recv(out_rev->data, out_rev->size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, &m_status);
		tick("receive batch");
		tick("merge batch");
		out = vStack(out,out_rev);
		tick("merge batch");
	}

	waitForAllRequests();

	return out;
}

Matrix *ClusterNet::dotMPI_unitSlice(Matrix *A, Matrix *B)
{
	int split_size = B->cols/m_nodes;
	std::string matrix_size = A->rows + "x" + split_size;
	Matrix *out;
	Matrix *out_rev;
	if(m_matrixCache.count("out " + matrix_size) > 0)
	{
		out = m_matrixCache["out " + matrix_size];
		m_matrixCacheUsage["out " + matrix_size] -= 1;
	}
	else
	{
		out = empty(A->rows,split_size);
		m_matrixCache["out " + matrix_size] = out;
		m_matrixCacheUsage["out " + matrix_size] = 0;
	}
	if(m_matrixCache.count("out_rev " + matrix_size) > 0)
	{
		out_rev = m_matrixCache["out_rev " + matrix_size];
		m_matrixCacheUsage["out_rev " + matrix_size] -= 1;
	}
	else
	{
		out_rev = empty(A->rows,split_size);
		m_matrixCache["out_rev " + matrix_size] = out_rev;
		m_matrixCacheUsage["out_rev " + matrix_size] = 0;
	}

	tick("slice unit");
	Matrix *B1 = slice_cols(B, split_size*m_myrank,split_size*(m_myrank+1)-1);
	tick("slice unit");
	tick("dot unit");
	dot(A,B1,out);
	tick("dot unit");
	for(int i = 0; i < m_nodes; i++)
	{
		if(m_myrank == i) { continue; }
		MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request));
		tick("send unit");
		MPI_Isend(out->data, out->size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, request);
		tick("send unit");
	}

	for(int i = 0; i < m_nodes; i++)
	{
		if(m_myrank == i) { continue; }
		tick("receive unit");
		MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request));
		//m_receiveRequests[i].push_back(request);
		MPI_Recv(out_rev->data, out_rev->size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, &m_status);
		tick("receive unit");
		tick("merge unit");
		out = hStack(out,out_rev);
		tick("merge unit");
	}

	//waitForAllRequests();
	/* TODO: Manage Matrix *cache
	typedef std::map<std::string, std::map<std::string, int>>::iterator it_type;
	for(it_type pair = m_matrixCacheUsage.begin(); iterator != m_matrixCacheUsage.end(); iterator++)
	{

		pair.first
	}
	*/


	return out;
}

//Uniform
Matrix *ClusterNet::rand(int rows, int cols)
{
  Matrix *out = empty(rows,cols);

  rand(rows, cols, out);

    return out;
}
void ClusterNet::rand(int rows, int cols, Matrix *out)
{
	curandGenerateUniform(m_generator, out->data, rows*cols);
	//print_gpu_matrix(*out);
}

//Gaussian
Matrix *ClusterNet::randn(int rows, int cols){ return randn(rows, cols, 0, 1); }
Matrix *ClusterNet::randn(int rows, int cols, float mean, float std)
{
    Matrix *out = empty(rows,cols);
    randn(rows, cols, mean, std, out);

    return out;
}
void ClusterNet::randn(int rows, int cols, float mean, float std, Matrix *out)
{
	curandGenerateNormal(m_generator, out->data, rows*cols, 0.0f, 1.0f);
}

Matrix *ClusterNet::rand_int(int rows, int cols, int low, int high)
{
	Matrix * out = rand(rows, cols);
	::rand_int(out, low, high);

	return out;
}


void ClusterNet::tick(){tick("default"); }
void ClusterNet::tick(std::string name)
{
	if(m_dictTickTock.count(name) > 0)
	{
		if(m_dictTickTockCumulative.count(name) > 0)
		{
			m_dictTickTockCumulative[name] += ::tock(m_dictTickTock[name],0.0f);
			m_dictTickTock.erase(name);
		}
		else
		{
			m_dictTickTockCumulative[name]  = ::tock(m_dictTickTock[name],0.0f);
			m_dictTickTock.erase(name);
		}
	}
	else
	{
		m_dictTickTock[name] = ::tick();
	}
}
void ClusterNet::tock(){tock("default"); }
void ClusterNet::tock(std::string name)
{
	if(m_dictTickTockCumulative.count(name) > 0)
	{
		::tock("<<<Cumulative>>>: " + name , m_dictTickTockCumulative[name]);
	}
	else
	{
		assert(("No tick event was registered for the name" + name, m_dictTickTock.count(name) > 0));
		::tock(m_dictTickTock[name], name);
	}
}

void ClusterNet::waitForAllRequests()
{
	tick("wait...");
	for (std::list<MPI_Request*>::const_iterator request = m_requests.begin(); request != m_requests.end(); ++request)
	{
	    MPI_Wait(*request, &m_status);
	}
	tick("wait...");
}

void ClusterNet::benchmark_dot()
{
	tock("send batch");
	tock("merge batch");
	tock("receive batch");
	tock("slice batch");
	tock("dot batch");

	tock("send unit");
	tock("merge unit");
	tock("receive unit");
	tock("slice unit");
	tock("dot unit");
}

Matrix *ClusterNet::dropout(Matrix *A, float dropout_rate)
{
	Matrix *out = rand(A->rows, A->cols);
	::dropout(A, out, dropout_rate);
	return out;
}

Matrix *ClusterNet::uniformSqrtWeight(int rows, int cols)
{
	Matrix * out = rand(rows, cols);
	::uniformSqrtWeight(out);
	return out;
}


Matrix *ClusterNet::sparseInitWeight(int rows, int cols){ return sparseInitWeight(rows, cols, 15); }
Matrix *ClusterNet::sparseInitWeight(int rows, int cols, int connections)
{

	Matrix *rdm = randn(cols,connections);
	Matrix *idx = rand_int(cols,connections,0,rows-1);
	Matrix *out = zeros(rows, cols);
	sparseRdmWeight(rdm,idx,out,connections);

	return out;

}



