#include <cublas_v2.h>
#include <clusterNet.cuh>
#include <basicOps.cuh>
#include <util.cuh>
#include <cstdlib>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>
#include <assert.h>
#include <algorithm>


ClusterNet::ClusterNet(){ init((int)(time(0) % 10000)); }
ClusterNet::ClusterNet(int seed){ init(seed);}
ClusterNet::ClusterNet(int argc, char* argv[], int seed){ init(seed); init_MPI(argc, argv); }
void ClusterNet::init(int seed)
{
	curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_generator, seed);
	curandSetGeneratorOffset(m_generator, 100);
	cublasCreate(&m_handle);
	m_hasMPI = false;
}

void ClusterNet::init_MPI(int argc, char * argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
	m_nodes = MPI::COMM_WORLD.Get_size();
	m_hasMPI = true;
}

void ClusterNet::shutdown_MPI()
{
	MPI_Finalize();
}

Matrix ClusterNet::dot(Matrix A, Matrix B)
{
	//if(m_hasMPI){ return dotMPI(A,B);}
	Matrix out = zeros(A.shape[0],B.shape[1]);
	if(checkMatrixOperation(A, B, out, 1) == 1){ throw "Matrix size error:\n"; }
	dot(A, B, out);

	return out;
}

Matrix ClusterNet::dotMPI_batchSlice(Matrix A, Matrix B)
{
	int split_size = A.shape[0]/m_nodes;
	Matrix out = empty(split_size,B.shape[1]);
	Matrix out_rev = empty(split_size,B.shape[1]);

	tick("slice batch");
	Matrix A1 = slice_rows(A, split_size*m_rank,split_size*(m_rank+1)-1);
	tick("slice batch");
	tick("dot batch");
	dot(A1,B,out);
	tick("dot batch");
	for(int i = 0; i < m_nodes; i++)
	{
		if(m_rank == i) { continue; }
		MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request));
		tick("send batch");
		MPI_Isend(out.data, out.size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, request);
		tick("send batch");
	}

	for(int i = 0; i < m_nodes; i++)
	{
		if(m_rank == i) { continue; }
		tick("receive batch");
		MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request));
		//m_receiveRequests[i].push_back(request);
		MPI_Recv(out_rev.data, out_rev.size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, &m_status);
		tick("receive batch");
		tick("merge batch");
		out = vStack(out,out_rev);
		tick("merge batch");
	}

	waitForAllRequests();

	return out;
}

Matrix ClusterNet::dotMPI_unitSlice(Matrix A, Matrix B)
{
	int split_size = B.shape[1]/m_nodes;
	std::string matrix_size = A.shape[0] + "x" + split_size;
	Matrix out;
	Matrix out_rev;
	if(m_matrixCache.count("out " + matrix_size) > 0)
	{
		out = m_matrixCache["out " + matrix_size];
		m_matrixCacheUsage["out " + matrix_size] -= 1;
	}
	else
	{
		out = empty(A.shape[0],split_size);
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
		out_rev = empty(A.shape[0],split_size);
		m_matrixCache["out_rev " + matrix_size] = out_rev;
		m_matrixCacheUsage["out_rev " + matrix_size] = 0;
	}

	tick("slice unit");
	Matrix B1 = slice_cols(B, split_size*m_rank,split_size*(m_rank+1)-1);
	tick("slice unit");
	tick("dot unit");
	dot(A,B1,out);
	tick("dot unit");
	for(int i = 0; i < m_nodes; i++)
	{
		if(m_rank == i) { continue; }
		MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request));
		tick("send unit");
		MPI_Isend(out.data, out.size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, request);
		tick("send unit");
	}

	for(int i = 0; i < m_nodes; i++)
	{
		if(m_rank == i) { continue; }
		tick("receive unit");
		MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request));
		//m_receiveRequests[i].push_back(request);
		MPI_Recv(out_rev.data, out_rev.size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, &m_status);
		tick("receive unit");
		tick("merge unit");
		out = hStack(out,out_rev);
		tick("merge unit");
	}

	//waitForAllRequests();
	/* TODO: Manage matrix cache
	typedef std::map<std::string, std::map<std::string, int>>::iterator it_type;
	for(it_type pair = m_matrixCacheUsage.begin(); iterator != m_matrixCacheUsage.end(); iterator++)
	{

		pair.first
	}
	*/


	return out;
}

void ClusterNet::dot(Matrix A, Matrix B, Matrix out)
{
	if(checkMatrixOperation(A, B, out, 1) == 1){ throw "Matrix size error:\n"; }
	cublasStatus_t status;

	const float alpha = 1.0f;
	const float beta = 0.0f;

	status = cublasSgemm(m_handle, CUBLAS_OP_N, CUBLAS_OP_N,
				A.shape[0], B.shape[1], A.shape[1],
				&alpha, A.data, A.shape[0],
				B.data, B.shape[0],
				&beta, out.data, out.shape[0]);

	if(status != CUBLAS_STATUS_SUCCESS)
		std::cout << "CUBLAS ERROR!\n";
}


//Uniform
Matrix ClusterNet::rand(int rows, int cols)
{
  Matrix out = empty(rows,cols);

  rand(rows, cols, out);

    return out;
}
void ClusterNet::rand(int rows, int cols, Matrix out)
{
	curandGenerateUniform(m_generator, out.data, rows*cols);
	//print_gpu_matrix(*out);
}

//Gaussian
Matrix ClusterNet::randn(int rows, int cols){ return randn(rows, cols, 0, 1); }
Matrix ClusterNet::randn(int rows, int cols, float mean, float std)
{
    Matrix out = empty(rows,cols);
    randn(rows, cols, mean, std, out);

    return out;
}
void ClusterNet::randn(int rows, int cols, float mean, float std, Matrix out)
{
	curandGenerateNormal(m_generator, out.data, rows*cols, 0.0f, 1.0f);
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


