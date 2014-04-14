#include <cublas_v2.h>
#include <cusparse_v2.h>
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
#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

using std::cout;
using std::endl;

ClusterNet::ClusterNet()
{
	init((int) (time(0) % 10000));
}
ClusterNet::ClusterNet(int seed)
{
	init(seed);
}
ClusterNet::ClusterNet(int argc, char* argv[], int seed)
{
	init_MPI(argc, argv);
	init(seed + (10000*MYRANK+12345));
}
ClusterNet::ClusterNet(int argc, char* argv[])
{
	init_MPI(argc, argv);
	init((int) (time(0) % (10000*MYRANK+12345)));
}
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
	m_cublasInitialized = false;
	m_cusparseInitialized = false;

	if (!m_hasMPI)
	{
		MYGPUID = 0;
		NODES = 1;
		PCIe_RANKS.push_back(0);
		MYRANK = 0;
	}
}

void ClusterNet::init_MPI(int argc, char * argv[])
{
	int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
	cudaSetDevice(local_rank);
	MYGPUID = local_rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MYRANK);
	MPI_Comm_size(MPI_COMM_WORLD, &MPI_SIZE);

	m_requests = (MPI_Request*) malloc(sizeof(MPI_Request) * MPI_SIZE);
	for (int i = 0; i < MPI_SIZE - 1; i++)
	{
		MPI_Request request;
		m_requests[i] = request;
		MPI_Request sendrequest;
		m_sendrequests.push_back(sendrequest);
	}

	m_destination = MYRANK + 1;
	m_source = MYRANK - 1;
	if (m_destination == MPI_SIZE)
	{
		m_destination = 0;
	}
	if (m_source < 0)
	{
		m_source = MPI_SIZE - 1;
	}

	m_hasMPI = true;
	compute_GPUID_and_Nodes();
	compute_PCIe_ranks();

}

void ClusterNet::compute_GPUID_and_Nodes()
{
	//determine master gpu ranks
	int recv;
	for (int i = 0; i < MPI_SIZE; i++)
	{
		if (i == MYRANK)
		{
			if (MYGPUID == 0)
				MASTER_GPU_RANKS.push_back(i);
			for (int j = 0; j < MPI_SIZE; j++)
			{
				if (i != j)
					MPI_Send(&MYGPUID, 1, MPI_INT, j, 999, MPI_COMM_WORLD );
			}
		} else
		{
			MPI_Recv(&recv, 1, MPI_INT, i, 999, MPI_COMM_WORLD, &m_status);

			if (recv == 0)
				MASTER_GPU_RANKS.push_back(i);
		}
	}

	NODES = MASTER_GPU_RANKS.size();

}

void ClusterNet::compute_PCIe_ranks()
{
	int gpus;
	cudaGetDeviceCount(&gpus);
	for(int i = 0; i < gpus; i++)
		PCIe_RANKS.push_back(MYRANK-MYGPUID + i);
}

void ClusterNet::shutdown_MPI()
{
	cudaDeviceSynchronize();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

Matrix *ClusterNet::dot(Matrix *A, Matrix *B)
{
	Matrix *out;
	if(B->isDistributed == 1)
	{
		out = dotMPI(A,B);
	}
	else
	{
		out = zeros(A->rows, B->cols);
		dot(A, B, out);
	}

	return out;
}

Matrix *ClusterNet::Tdot(Matrix *A, Matrix *B)
{
	Matrix *out;
	if(B->isDistributed == 1)
	{
		out = TdotMPI(A,B);
	}
	else
	{
		out = zeros(A->cols, B->cols);
		Tdot(A, B, out);
	}

	return out;
}

Matrix *ClusterNet::dotT(Matrix *A, Matrix *B)
{
	Matrix *out;
	if(B->isDistributed  == 1)
	{
		out = dotTMPI(A,B);
	}
	else
	{
		out =  zeros(A->rows, B->rows);
		dotT(A, B, out);
	}

	return out;
}

void ClusterNet::dotT(Matrix *A, Matrix *B, Matrix *out)
{
	if(B->isDistributed == 1 || out->isDistributed == 1)
		dotTMPI(A,B,out);
	else
		dot(A, B, out, CUBLAS_OP_N, CUBLAS_OP_T);
}
void ClusterNet::Tdot(Matrix *A, Matrix *B, Matrix *out)
{
	if(B->isDistributed == 1 || out->isDistributed == 1)
		TdotMPI(A,B,out);
	else
		dot(A, B, out, CUBLAS_OP_T, CUBLAS_OP_N);
}
void ClusterNet::dot(Matrix *A, Matrix *B, Matrix *out)
{
	if(B->isDistributed ==1 || out->isDistributed == 1)
		dotMPI(A,B,out);
	else
		dot(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N);
}


Matrix *ClusterNet::dot_sparse(Matrix *A, Matrix *B){ Matrix *out = empty(A->rows, B->cols); dot_sparse(A,B, out, CUBLAS_OP_N, CUBLAS_OP_N); return out; }
Matrix *ClusterNet::Tdot_sparse(Matrix *A, Matrix *B){ Matrix *out = empty(A->rows, B->cols); dot_sparse(A,B, out, CUBLAS_OP_T, CUBLAS_OP_N); return out; }
Matrix *ClusterNet::dotT_sparse(Matrix *A, Matrix *B){ Matrix *out = empty(A->rows, B->cols); dot_sparse(A,B, out, CUBLAS_OP_N, CUBLAS_OP_T); return out; }
void ClusterNet::Tdot_sparse(Matrix *A, Matrix *B, Matrix *out){ dot_sparse(A, B, out, CUBLAS_OP_T, CUBLAS_OP_N); }
void ClusterNet::dotT_sparse(Matrix *A, Matrix *B, Matrix *out){ dot_sparse(A, B, out, CUBLAS_OP_N, CUBLAS_OP_T); }
void ClusterNet::dot_sparse(Matrix *A, Matrix *B, Matrix *out){ dot_sparse(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N); }
void ClusterNet::dot_sparse(Matrix *A, Matrix *B, Matrix *out, cublasOperation_t T1, cublasOperation_t T2)
{
	if(!m_cusparseInitialized)
	{
		m_cusparseInitialized = true;
		cusparseCreate(&m_sparse_handle);
	}

	cusparseStatus_t status;
	cusparseMatDescr_t descriptor_A;
	cusparseCreateMatDescr(&descriptor_A);

	cusparseSetMatType(descriptor_A,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descriptor_A,CUSPARSE_INDEX_BASE_ZERO);

    const float alpha = 1;
    const float beta = 0;
	int B_cols = T2 == CUBLAS_OP_T ? B->rows : B->cols;



	/*
	 cout << "T1: " << T1 << endl;
	 cout << "T2: " << T2 << endl;
	 cout << "A rows: " << A->rows << endl;
	 cout << "A cols: " << A->cols << endl;
	 cout << "B rows: " << B->rows << endl;
	 cout << "B cols: " << B_cols << endl;
	 cout << "out rows: " << out->rows << endl;
	 cout << "out cols: " << out->cols << endl;
	 cout << "sum A: " << sum(A) << endl;
	 cout << "sum B: "  << sum(B) << endl;
	 cout << "sum out: " << sum(out) << endl;
	 */

	//size_t freemem, total;
	//cudaMemGetInfo(&freemem,&total);
	//cout << "pre memory: " << freemem << endl;




	status = cusparseScsrmm2(m_sparse_handle,
		T1 == CUBLAS_OP_N ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE,
		T2 == CUBLAS_OP_N ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE,
				A->rows, B_cols, A->cols,
		A->size, &alpha, descriptor_A,
		A->data, A->ptr_rows, A->idx_cols,
		B->data, B->rows,  &beta,
		out->data, out->rows);


	//cudaMemGetInfo(&freemem,&total);
	//cout << "post memory: " << freemem << endl;



	if (status != CUSPARSE_STATUS_SUCCESS)
	{
		cout << "CUSPARSE ERROR: " << status <<  "!" << endl;
		throw "CUSPARSE ERROR!";
	}



}


void ClusterNet::dot(Matrix *A, Matrix *B, Matrix *out, cublasOperation_t T1, cublasOperation_t T2)
{
	//if(checkMatrixOperation(A, B, out, 1) == 1){ throw "Matrix *size error:\n"; }
	cublasStatus_t status;
	if(!m_cublasInitialized)
	{
		m_cublasInitialized = true;
		cublasCreate_v2(&m_handle);
	}

	const float alpha = 1.0f;
	const float beta = 0.0f;
	int A_rows = A->rows, A_cols = A->cols, B_cols = B->cols;
	if (T1 == CUBLAS_OP_T)
	{
		A_rows = A->cols;
		A_cols = A->rows;
	}
	if (T2 == CUBLAS_OP_T)
		B_cols = B->rows;


	/*
	 cout << "T1: " << T1 << endl;
	 cout << "T2: " << T2 << endl;
	 cout << "A rows: " << A_rows << endl;
	 cout << "A cols: " << A_cols << endl;
	 cout << "B rows: " << B->rows << endl;
	 cout << "B cols: " << B_cols << endl;
	 cout << "out rows: " << out->rows << endl;
	 cout << "out cols: " << out->cols << endl;
	 cout << "sum A: " << sum(A) << endl;
	 cout << "sum B: "  << sum(B) << endl;
	 cout << "sum out: " << sum(out) << endl;
*/


	status = cublasSgemm(m_handle, T1, T2, A_rows, B_cols,
			A_cols, &alpha, A->data, A->rows, B->data, B->rows, &beta,
			out->data, out->rows);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printmat(A,0,1,0,10);
		printmat(A,A->rows-1,A->rows, A->cols-10,A->cols);
		std::cout << "CUBLAS ERROR: Status " << status << std::endl;
		throw "CUBLAS ERROR";

	}
}


Matrix *ClusterNet::dotMPI(Matrix *A, Matrix *B)
{
	Matrix *out = empty(A->rows, B->isDistributed == 0 ? B->cols : B->cols_distributed);
	dotMPI(A, B, out);
	return out;
}

Matrix *ClusterNet::dotTMPI(Matrix *A, Matrix *B)
{
	Matrix *out = empty(A->rows, B->rows);
	dotTMPI(A, B, out);
	return out;
}
Matrix *ClusterNet::TdotMPI(Matrix *A, Matrix *B)
{
	Matrix *out = empty(A->cols, B->isDistributed == 0 ? B->cols : B->cols_distributed);
	TdotMPI(A, B, out);
	return out;
}
void ClusterNet::dotMPI(Matrix *A, Matrix *B, Matrix *out){ dotMPI(A,B,out,false, false); }
void ClusterNet::TdotMPI(Matrix *A, Matrix *B, Matrix *out){ dotMPI(A,B,out,true, false); }
void ClusterNet::dotTMPI(Matrix *A, Matrix *B, Matrix *out){ dotMPI(A,B,out,false, true); }
void ClusterNet::dotMPI(Matrix *A, Matrix *B, Matrix *out, bool applyTranspose_A, bool applyTranspose_B)
{
	int col_split_size = (B->isDistributed == 1 ? B->cols_distributed : B->cols) / MPI_SIZE;
	int remainder = (B->isDistributed == 1 ? B->cols_distributed : B->cols) - (col_split_size*MPI_SIZE);
	std::string strMatrixName = SSTR(A->rows) + "x" + SSTR(A->cols) + " * " +
								SSTR(B->rows) + SSTR((B->isDistributed == 1 ? B->cols_distributed : B->cols)) +
								"T" + SSTR((applyTranspose_B ? 1 : 0));

	if(out->isDistributed == 0)
	{
		if (m_matrixCache.count(strMatrixName) == 0)
		{
			if(!applyTranspose_B)
			{
				Matrix** arrOut = (Matrix**) malloc(sizeof(Matrix*) * MPI_SIZE);
				for (int i = 0; i < MPI_SIZE; i++)
				{
					if (i == MPI_SIZE - 1)
						arrOut[i] = empty(A->rows, col_split_size + remainder);
					else
						arrOut[i] = empty(A->rows, col_split_size);
				}
				m_matrixCache[strMatrixName] = arrOut;
				m_matrixCacheUsage[strMatrixName] = 1;

				float **h_arrA = (float**) malloc(sizeof(float*) * MPI_SIZE);
				for (int i = 0; i < MPI_SIZE; i++)
					h_arrA[i] = m_matrixCache[strMatrixName][i]->data;

				float **d_arrA;
				cudaMalloc((void**) &d_arrA, sizeof(float*) * MPI_SIZE);
				cudaMemcpy(d_arrA, h_arrA, sizeof(float*) * MPI_SIZE,cudaMemcpyDefault);
				m_matrixHStackCache[strMatrixName] = d_arrA;
				free(h_arrA);
			}
			else
			{
				Matrix** arrOut = (Matrix**) malloc(sizeof(Matrix*)*MPI_SIZE);
				for (int i = 0; i < MPI_SIZE; i++)
					arrOut[i] = empty(out->rows, out->cols);

				m_matrixCache[strMatrixName] = arrOut;
			}
		}

		m_matrixCacheUsage[strMatrixName] = 1;

		std::map<std::string, int>::iterator iter;
		std::vector<std::string> toDecrement;
		std::vector<std::string> toDelete;
		//determine the matices to delete and to decrement
		for (iter = m_matrixCacheUsage.begin(); iter != m_matrixCacheUsage.end();++iter)
		{
			if (iter->first != strMatrixName)
			{
				toDecrement.push_back(iter->first);
				if (iter->second < -2000)
					toDelete.push_back(iter->first);
			}
		}

		//decrement matrices
		for (int i = 0; i < toDecrement.size(); i++)
			m_matrixCacheUsage[toDecrement[i]] -= 1;

		//free matrices that were reused not enough
		for (int i = 0; i < toDelete.size(); i++)
		{
			if(m_matrixHStackCache.count(toDelete[i]) > 0)
			{
				for (int j = 0; j < MPI_SIZE; j++)
					cudaFree(m_matrixCache[toDelete[i]][j]->data);

				cudaFree(m_matrixHStackCache[toDelete[i]]);
				m_matrixHStackCache.erase(toDelete[i]);
			}
			else
			{
				cudaFree(m_matrixCache[toDelete[i]][0]->data);
			}
			m_matrixCache.erase(toDelete[i]);
			m_matrixCacheUsage.erase(toDelete[i]);
		}


		toDecrement.clear();
		toDelete.clear();
	}


	Matrix *B1;
	Matrix *A1;
	if (B->isDistributed == 0)
	{
		if (MYRANK == MPI_SIZE - 1)
			B1 = slice_cols(B, col_split_size * MYRANK,	col_split_size * (MYRANK + 1) - 1 + remainder);
		else
			B1 = slice_cols(B, col_split_size * MYRANK,	col_split_size * (MYRANK + 1) - 1);

		if(out->isDistributed == 1)
			dot(A, B1, out, applyTranspose_A ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N);
		else
			dot(A, B1, m_matrixCache[strMatrixName][MYRANK], CUBLAS_OP_N, CUBLAS_OP_N);
	}
	else
	{
		if(!applyTranspose_B)
			dot(A, B, m_matrixCache[strMatrixName][MYRANK], CUBLAS_OP_N, CUBLAS_OP_N);
		else
		{
			if (MYRANK == MPI_SIZE - 1)
				A1 = slice_cols(A, col_split_size * MYRANK,	col_split_size * (MYRANK + 1) - 1 + remainder);
			else
				A1 = slice_cols(A, col_split_size * MYRANK,	col_split_size * (MYRANK + 1) - 1);


			dot(A1,B,m_matrixCache[strMatrixName][MYRANK], CUBLAS_OP_N, CUBLAS_OP_T);
		}
	}

	if(out->isDistributed == 0 && !applyTranspose_B)
	{
		int matrix_idx = MYRANK;
		for (int i = 0; i < MPI_SIZE - 1; i++)
		{
			MPI_Isend(m_matrixCache[strMatrixName][matrix_idx]->data, m_matrixCache[strMatrixName][matrix_idx]->size, MPI_FLOAT, m_destination, 100, MPI_COMM_WORLD, &m_sendrequests[i]);
			matrix_idx = (matrix_idx - 1) < 0 ? MPI_SIZE - 1 : (matrix_idx - 1);
			MPI_Recv(m_matrixCache[strMatrixName][matrix_idx]->data, m_matrixCache[strMatrixName][matrix_idx]->size, MPI_FLOAT, m_source, 100, MPI_COMM_WORLD, &m_status);
		}

		for(int i = 0; i < MPI_SIZE -1;i++ )
			MPI_Wait(&m_sendrequests[i],&m_status);

		if(B->isDistributed == 0)
			cudaFree(B1->data);
		hStackN(m_matrixHStackCache[strMatrixName],	m_matrixCache[strMatrixName][0]->size, out, MPI_SIZE);


	}
	else if(out->isDistributed == 0 && applyTranspose_B)
	{
		int matrix_idx = MYRANK;
		for (int i = 0; i < MPI_SIZE - 1; i++)
		{
			MPI_Isend(m_matrixCache[strMatrixName][matrix_idx]->data,m_matrixCache[strMatrixName][matrix_idx]->size, MPI_FLOAT, m_destination, 100, MPI_COMM_WORLD, &m_sendrequests[i]);
			matrix_idx = (matrix_idx - 1) < 0 ? MPI_SIZE - 1 : (matrix_idx - 1);
			MPI_Recv(m_matrixCache[strMatrixName][matrix_idx]->data, m_matrixCache[strMatrixName][matrix_idx]->size, MPI_FLOAT, m_source, 100, MPI_COMM_WORLD, &m_status);
		}
		for(int i = 0; i < MPI_SIZE -1;i++ )
			MPI_Wait(&m_sendrequests[i],&m_status);

		cudaMemset(out->data,0,out->bytes);
		for(int i= 0; i < MPI_SIZE; i++)
			add(out,m_matrixCache[strMatrixName][i],out);


		cudaFree(A1->data);

	}

	if(out->isDistributed == 1)
		cudaFree(B1->data);


}

//Uniform
Matrix *ClusterNet::rand(int rows, int cols)
{
	Matrix *out = empty(rows, cols);

	rand(rows, cols, out);

	return out;
}
void ClusterNet::rand(int rows, int cols, Matrix *out)
{
	curandGenerateUniform(m_generator, out->data, rows * cols);
}

//Gaussian
Matrix *ClusterNet::randn(int rows, int cols)
{
	return randn(rows, cols, 0, 1);
}
Matrix *ClusterNet::randn(int rows, int cols, float mean, float std)
{
	Matrix *out = empty(rows, cols);
	randn(rows, cols, mean, std, out);

	return out;
}
void ClusterNet::randn(int rows, int cols, float mean, float std, Matrix *out)
{
	curandGenerateNormal(m_generator, out->data, rows * cols, 0.0f, 1.0f);
}

Matrix *ClusterNet::rand_int(int rows, int cols, int low, int high)
{
	Matrix * out = rand(rows, cols);
	::rand_int(out, low, high);

	return out;
}

void ClusterNet::tick()
{
	tick("default");
}
void ClusterNet::tick(std::string name)
{
	if (m_dictTickTock.count(name) > 0)
	{
		if (m_dictTickTockCumulative.count(name) > 0)
		{
			m_dictTickTockCumulative[name] += ::tock(m_dictTickTock[name],
					0.0f);
			m_dictTickTock.erase(name);
		} else
		{
			m_dictTickTockCumulative[name] = ::tock(m_dictTickTock[name], 0.0f);
			m_dictTickTock.erase(name);
		}
	} else
	{
		m_dictTickTock[name] = ::tick();
	}
}
void ClusterNet::tock()
{
	tock("default");
}
void ClusterNet::tock(std::string name)
{
	if (m_dictTickTockCumulative.count(name) > 0)
	{
		::tock("<<<Cumulative>>>: " + name, m_dictTickTockCumulative[name]);
		m_dictTickTockCumulative.erase(name);
	} else
	{
		if (m_dictTickTock.count(name) == 0)
			cout << "Error for name: " << name << endl;
		assert(
				("No tick event was registered for the name" + name, m_dictTickTock.count(
						name) > 0));
		::tock(m_dictTickTock[name], name);
		m_dictTickTock.erase(name);
	}
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

Matrix *ClusterNet::sparseInitWeight(int rows, int cols)
{
	return sparseInitWeight(rows, cols, 15);
}
Matrix *ClusterNet::sparseInitWeight(int rows, int cols, int connections)
{

	Matrix *rdm = randn(cols, connections);
	Matrix *idx = rand_int(cols, connections, 0, rows - 1);
	Matrix *out = zeros(rows, cols);
	sparseRdmWeight(rdm, idx, out, connections);

	cudaDeviceSynchronize();
	cudaFree(rdm->data);
	cudaFree(idx->data);

	return out;

}

Matrix *ClusterNet::distributed_uniformSqrtWeight(int rows, int cols)
{
	assert(m_hasMPI);
	Matrix *W;
	int split_size = cols / MPI_SIZE;
	int remainder = cols - (split_size * MPI_SIZE);
	if (MYRANK < MPI_SIZE - 1)
		W = rand(rows, split_size);
	else
		W = rand(rows, split_size + remainder);

	W->isDistributed = 1;
	W->cols_distributed = cols;
	::uniformSqrtWeight(W,W->rows,W->cols_distributed);

	return W;
}

Matrix *ClusterNet::distributed_zeros(int rows, int cols)
{
	assert(m_hasMPI);
	Matrix *W;
	int split_size = cols / MPI_SIZE;
	int remainder = cols - (split_size * MPI_SIZE);
	if (MYRANK < MPI_SIZE - 1)
		W = zeros(rows, split_size);
	else
		W = zeros(rows, split_size + remainder);

	W->isDistributed = 1;
	W->cols_distributed = cols;

	return W;
}

Matrix *ClusterNet::distributed_ones(int rows, int cols)
{
	assert(m_hasMPI);
	Matrix *W;
	int split_size = cols / MPI_SIZE;
	int remainder = cols - (split_size * MPI_SIZE);
	if (MYRANK < MPI_SIZE - 1)
		W = ones(rows, split_size);
	else
		W = ones(rows, split_size + remainder);

	W->isDistributed = 1;
	W->cols_distributed = cols;

	return W;
}

Matrix *ClusterNet::distributed_sparseInitWeight(int rows, int cols)
{
	assert(m_hasMPI);
	int split_size = cols / MPI_SIZE;
	int remainder = cols - (split_size * MPI_SIZE);
	int col_size =
			MYRANK < MPI_SIZE - 1 ?
					split_size + remainder : split_size;
	int connections = 15;

	Matrix *W = zeros(rows, col_size);
	Matrix *rdm = randn(col_size, connections);
	Matrix *idx = rand_int(col_size, connections, 0, rows - 1);
	Matrix *out = zeros(rows, col_size);

	sparseRdmWeight(rdm, idx, out, connections);
	cudaFree(rdm->data);
	cudaFree(idx->data);

	W->isDistributed = 1;
	W->cols_distributed = cols;

	return W;
}

Matrix *ClusterNet::dense_to_sparse(Matrix *A)
{
	if(!m_cusparseInitialized)
	{
		m_cusparseInitialized = true;
		cusparseCreate(&m_sparse_handle);
	}

	cusparseMatDescr_t descriptor_A;
	cusparseCreateMatDescr(&descriptor_A);
	cusparseSetMatType(descriptor_A,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descriptor_A,CUSPARSE_INDEX_BASE_ZERO);
    int nonzeros = 0;
    int *nonzerosPerRow;

    cudaMalloc((void**)&nonzerosPerRow,sizeof(int)*A->rows);

    cusparseSnnz(m_sparse_handle,CUSPARSE_DIRECTION_ROW,
    			A->rows, A->cols,descriptor_A,
    			A->data,A->rows,nonzerosPerRow,&nonzeros);

    Matrix *out = empty_sparse(A->rows,A->cols,nonzeros);

	cusparseSdense2csr(m_sparse_handle,A->rows,A->cols,
					   descriptor_A,A->data,A->rows,nonzerosPerRow,
					   out->data,out->ptr_rows,out->idx_cols);

	return out;
}

Matrix *ClusterNet::sparse_to_dense(Matrix *A)
{
	if(!m_cusparseInitialized)
	{
		m_cusparseInitialized = true;
		cusparseCreate(&m_sparse_handle);
	}

	assert(A->isSparse == 1);

	cusparseMatDescr_t descriptor_A;
	cusparseCreateMatDescr(&descriptor_A);
	cusparseSetMatType(descriptor_A,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descriptor_A,CUSPARSE_INDEX_BASE_ZERO);

    Matrix *out = zeros(A->rows,A->cols);

    cusparseScsr2dense(m_sparse_handle,A->rows,A->cols,
			   descriptor_A,
			   A->data,A->ptr_rows,A->idx_cols,
			   out->data,out->rows);

    //to_col_major(out,out);

	return out;
}
