#include <cublas_v2.h>
#include <clusterNet.cuh>
#include <basicOps.cuh>
#include <util.cuh>
#include <cstdlib>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>


ClusterNet::ClusterNet(){ init((int)(time(0) % 10000)); }
ClusterNet::ClusterNet(int seed){ init(seed);}
ClusterNet::ClusterNet(int argc, char* argv[], int seed){ init(seed); init_MPI(argc, argv); }
//ClusterNet::~ClusterNet(){ if(m_hasMPI) MPI_Finalize(); }
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
	dot(A, B, out);
	checkMatrixOperation(A, B, out, 1);

	return out;
}

Matrix ClusterNet::dotMPI(Matrix A, Matrix B)
{
	int split_size = A.shape[0]/m_nodes;
	Matrix out = empty(split_size,B.shape[1]);
	Matrix out_rev = empty(split_size,B.shape[1]);

	Matrix A1 = slice_rows(A, split_size*m_rank,split_size*(m_rank+1)-1);
	dot(A1,B,out);
	for(int i = 0; i < m_nodes; i++)
	{
		if(m_rank == i) { continue; }
		MPI_Send(out.data, out.size, MPI_FLOAT, i, 100, MPI_COMM_WORLD);
	}

	for(int i = 0; i < m_nodes; i++)
	{
		if(m_rank == i) { continue; }
		MPI_Recv(out_rev.data, out_rev.size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, &m_status);
		out = merge(out,out_rev);
	}

	return out;
}
void ClusterNet::dotMPI(Matrix A, Matrix B, Matrix out)
{

}

void ClusterNet::dot(Matrix A, Matrix B, Matrix out)
{
	checkMatrixOperation(A, B, out, 1);
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


