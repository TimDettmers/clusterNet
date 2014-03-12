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

using std::cout;
using std::endl;

ClusterNet::ClusterNet() {
	init((int) (time(0) % 10000));
}
ClusterNet::ClusterNet(int seed) {
	init(seed);
}
ClusterNet::ClusterNet(int argc, char* argv[], int seed) {
	init_MPI(argc, argv);
	init(seed);
}
void ClusterNet::init(int seed) {
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
	int gpus;
	cudaGetDeviceCount(&gpus);
	for (int i = 0; i < gpus; i++) {
		cudaSetDevice(i);
		cublasHandle_t handle;
		cublasCreate(&handle);
		m_handles.push_back(handle);
	}
	cudaSetDevice(MYGPUID);

	if (!m_hasMPI) {
		MYGPUID = 0;
		NODES = 1;
		PCIe_RANKS.push_back(0);
		MYRANK = 0;
		m_hasMPI = false;
	}
}

void ClusterNet::init_MPI(int argc, char * argv[]) {

	int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
	cudaSetDevice(local_rank);
	MYGPUID = local_rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MYRANK);
	MPI_Comm_size(MPI_COMM_WORLD, &MPI_SIZE);

	m_requests = (MPI_Request*) malloc(sizeof(MPI_Request) * MPI_SIZE);
	for (int i = 0; i < MPI_SIZE - 1; i++) {
		MPI_Request request;
		m_requests[i] = request;
	}

	m_destination = MYRANK + 1;
	m_source = MYRANK - 1;
	if (m_destination == MPI_SIZE) {
		m_destination = 0;
	}
	if (m_source < 0) {
		m_source = MPI_SIZE - 1;
	}

	m_hasMPI = true;
	compute_GPUID_and_Nodes();
	compute_PCIe_ranks();

}

void ClusterNet::compute_GPUID_and_Nodes() {
	//determine master gpu ranks
	int recv;
	for (int i = 0; i < MPI_SIZE; i++) {
		if (i == MYRANK) {
			if (MYGPUID == 0)
				MASTER_GPU_RANKS.push_back(i);
			for (int j = 0; j < MPI_SIZE; j++) {
				if (i != j)
					MPI_Send(&MYGPUID, 1, MPI_INT, j, 999, MPI_COMM_WORLD );
			}
		} else {
			MPI_Recv(&recv, 1, MPI_INT, i, 999, MPI_COMM_WORLD, &m_status);

			if (recv == 0)
				MASTER_GPU_RANKS.push_back(i);
		}
	}

	NODES = MASTER_GPU_RANKS.size();

}

void ClusterNet::compute_PCIe_ranks() {
	int gpus;
	cudaGetDeviceCount(&gpus);
	if (gpus > 1) {
		int *PCIe_Ranks_buffer = (int*) malloc(sizeof(int) * gpus - 1);
		if (MYGPUID == 0) {
			//device 0 on the PCIe does know how many gpus are on the board
			//and also which gpu has which rank -> spread the information
			for (int i = 0; i < gpus; i++)
				PCIe_Ranks_buffer[i] = MYRANK + i;

			for (int i = 0; i < gpus; i++) {
				if (i > 0 && PCIe_Ranks_buffer[i] < MPI_SIZE)
					MPI_Send(PCIe_Ranks_buffer, gpus, MPI_INT,
							PCIe_Ranks_buffer[i], 17, MPI_COMM_WORLD );
				PCIe_RANKS.push_back(PCIe_Ranks_buffer[i]);
			}
		} else {
			MPI_Recv(PCIe_Ranks_buffer, gpus, MPI_INT, MYRANK - MYGPUID, 17,
					MPI_COMM_WORLD, &m_status);
			for (int i = 0; i < gpus; i++)
				PCIe_RANKS.push_back(PCIe_Ranks_buffer[i]);
		}
		free(PCIe_Ranks_buffer);
	} else {
		//no sends and receives
		PCIe_RANKS.push_back(MYRANK);
	}
}

void ClusterNet::shutdown() {
	MPI_Finalize();
}

Matrix *ClusterNet::dot(Matrix *A, Matrix *B) {
	Matrix *out = zeros(A->rows, B->cols);
	dot(A, B, out);

	return out;
}

Matrix *ClusterNet::Tdot(Matrix *A, Matrix *B) {
	//if(m_hasMPI){ return dotMPI(A,B);}
	Matrix *out = zeros(A->cols, B->cols);
	Tdot(A, B, out);

	return out;
}

Matrix *ClusterNet::dotT(Matrix *A, Matrix *B) {
	//if(m_hasMPI){ return dotMPI(A,B);}
	Matrix *out = zeros(A->rows, B->rows);
	dotT(A, B, out);

	return out;
}

void ClusterNet::dotT(Matrix *A, Matrix *B, Matrix *out) {
	dot(A, B, out, CUBLAS_OP_N, CUBLAS_OP_T);
}
void ClusterNet::Tdot(Matrix *A, Matrix *B, Matrix *out) {
	dot(A, B, out, CUBLAS_OP_T, CUBLAS_OP_N);
}
void ClusterNet::dot(Matrix *A, Matrix *B, Matrix *out) {
	dot(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N);
}
void ClusterNet::dot(Matrix *A, Matrix *B, Matrix *out, cublasOperation_t T1,
		cublasOperation_t T2) {
	//if(checkMatrixOperation(A, B, out, 1) == 1){ throw "Matrix *size error:\n"; }
	cublasStatus_t status;

	const float alpha = 1.0f;
	const float beta = 0.0f;
	int A_rows = A->rows, B_rows = B->rows, A_cols = A->cols, B_cols = B->cols;
	if (T1 == CUBLAS_OP_T) {
		A_rows = A->cols;
		A_cols = A->rows;
	}
	if (T2 == CUBLAS_OP_T) {
		B_rows = B->cols;
		B_cols = B->rows;
	}

	int current_device;
	cudaGetDevice(&current_device);

	//cout << "current device: " << current_device << endl;

	/*
	 cout << "T1: " << T1 << endl;
	 cout << "T2: " << T2 << endl;
	 cout << "A rows: " << A->rows << endl;
	 cout << "A cols: " << A->cols << endl;
	 cout << "B rows: " << B->rows << endl;
	 cout << "B cols: " << B->cols << endl;
	 cout << "out rows: " << out->rows << endl;
	 cout << "out cols: " << out->cols << endl;
	 */

	status = cublasSgemm(m_handles[current_device], T1, T2, A_rows, B_cols,
			A_cols, &alpha, A->data, A->rows, B->data, B->rows, &beta,
			out->data, out->rows);

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cout << "CUBLAS ERROR: Status " << status << std::endl;
		throw "CUBLAS ERROR";
	}
}

Matrix *ClusterNet::dotMPI_batchSlice(Matrix *A, Matrix *B) {
	int split_size = A->rows / MPI_SIZE;
	Matrix *out = empty(split_size, B->cols);
	Matrix *out_rev = empty(split_size, B->cols);

	Matrix *A1 = slice_rows(A, split_size * MYRANK,
			split_size * (MYRANK + 1) - 1);
	dot(A1, B, out);
	for (int i = 0; i < MPI_SIZE; i++) {
		if (MYRANK == i) {
			continue;
		}
		MPI_Request *request = (MPI_Request*) malloc(sizeof(MPI_Request));
		MPI_Isend(out->data, out->size, MPI_FLOAT, i, 100, MPI_COMM_WORLD,
				request);
	}

	for (int i = 0; i < MPI_SIZE; i++) {
		if (MYRANK == i) {
			continue;
		}
		MPI_Request *request = (MPI_Request*) malloc(sizeof(MPI_Request));
		//m_receiveRequests[i].push_back(request);
		MPI_Recv(out_rev->data, out_rev->size, MPI_FLOAT, i, 100,
				MPI_COMM_WORLD, &m_status);
		out = vStack(out, out_rev);
	}

	//waitForAllRequests();

	return out;
}

Matrix *ClusterNet::dotMPI_unitSlice(Matrix *A, Matrix *B) {
	Matrix *out = empty(A->rows, B->cols);
	dotMPI_unitSlice(A, B, out);
	return out;
}
void ClusterNet::dotMPI_unitSlice(Matrix *A, Matrix *B, Matrix *out) {
	int split_size = B->cols / MPI_SIZE;
	std::string strMatrixName = A->rows + "x" + B->cols;

	if (m_matrixCache.count(strMatrixName) == 0) {
		Matrix** arrOut = (Matrix**) malloc(sizeof(Matrix*) * MPI_SIZE);
		for (int i = 0; i < MPI_SIZE; i++) {
			if (i == MPI_SIZE - 1)
				arrOut[i] = empty(A->rows, split_size + (B->cols % split_size));
			else
				arrOut[i] = empty(A->rows, split_size);
		}
		m_matrixCache[strMatrixName] = arrOut;
		m_matrixCacheUsage[strMatrixName] = 1;

		float **h_arrA = (float**) malloc(sizeof(float*) * MPI_SIZE);
		for (int i = 0; i < MPI_SIZE; i++)
			h_arrA[i] = m_matrixCache[strMatrixName][i]->data;

		float **d_arrA;
		cudaMalloc((void**) &d_arrA, sizeof(float*) * MPI_SIZE);
		cudaMemcpy(d_arrA, h_arrA, sizeof(float*) * MPI_SIZE,
				cudaMemcpyDefault);

		m_matrixHStackCache[strMatrixName] = d_arrA;
		free(h_arrA);

	}

	std::map<std::string, int>::iterator iter;
	std::vector<std::string> toDecrement;
	std::vector<std::string> toDelete;
	for (iter = m_matrixCacheUsage.begin(); iter != m_matrixCacheUsage.end();
			++iter) {
		if (iter->first != strMatrixName) {
			toDecrement.push_back(iter->first);
			if (iter->second < -10)
				toDelete.push_back(iter->first);
		}
	}

	for (int i = 0; i < toDecrement.size(); i++)
		m_matrixCacheUsage[toDecrement[i]] -= 1;

	for (int i = 0; i < toDelete.size(); i++) {
		for (int j = 0; j < MPI_SIZE; j++)
			cudaFree(m_matrixCache[toDelete[i]][i]->data);

		m_matrixCache.erase(toDelete[i]);
		m_matrixCacheUsage.erase(toDelete[i]);
		m_matrixHStackCache.erase(toDelete[i]);
	}

	m_matrixCacheUsage[strMatrixName] += 1;

	toDecrement.clear();
	toDelete.clear();

	Matrix *B1;
	if (MYRANK == MPI_SIZE - 1)
		B1 = slice_cols(B, split_size * MYRANK,
				split_size * (MYRANK + 1) - 1 + (B->cols % split_size));
	else
		B1 = slice_cols(B, split_size * MYRANK, split_size * (MYRANK + 1) - 1);
	int matrix_idx = 0;

	dot(A, B1, m_matrixCache[strMatrixName][MYRANK]);

	matrix_idx = MYRANK;
	for (int i = 0; i < MPI_SIZE - 1; i++) {
		MPI_Isend(m_matrixCache[strMatrixName][matrix_idx]->data,
				m_matrixCache[strMatrixName][matrix_idx]->size, MPI_FLOAT,
				m_destination, 100, MPI_COMM_WORLD, &m_sendrequest);
		matrix_idx = (matrix_idx - 1) < 0 ? MPI_SIZE - 1 : (matrix_idx - 1);
		MPI_Irecv(m_matrixCache[strMatrixName][matrix_idx]->data,
				m_matrixCache[strMatrixName][matrix_idx]->size, MPI_FLOAT,
				m_source, 100, MPI_COMM_WORLD, &m_requests[i]);
	}

	cudaFree(B1->data);
	MPI_Waitall(MPI_SIZE - 1, m_requests, MPI_STATUSES_IGNORE );
	hStackN(m_matrixHStackCache[strMatrixName],
			m_matrixCache[strMatrixName][0]->size, out, MPI_SIZE);
}

//Uniform
Matrix *ClusterNet::rand(int rows, int cols) {
	Matrix *out = empty(rows, cols);

	rand(rows, cols, out);

	return out;
}
void ClusterNet::rand(int rows, int cols, Matrix *out) {
	curandGenerateUniform(m_generator, out->data, rows * cols);
	//print_gpu_matrix(*out);
}

//Gaussian
Matrix *ClusterNet::randn(int rows, int cols) {
	return randn(rows, cols, 0, 1);
}
Matrix *ClusterNet::randn(int rows, int cols, float mean, float std) {
	Matrix *out = empty(rows, cols);
	randn(rows, cols, mean, std, out);

	return out;
}
void ClusterNet::randn(int rows, int cols, float mean, float std, Matrix *out) {
	curandGenerateNormal(m_generator, out->data, rows * cols, 0.0f, 1.0f);
}

Matrix *ClusterNet::rand_int(int rows, int cols, int low, int high) {
	Matrix * out = rand(rows, cols);
	::rand_int(out, low, high);

	return out;
}

void ClusterNet::tick() {
	tick("default");
}
void ClusterNet::tick(std::string name) {
	if (m_dictTickTock.count(name) > 0) {
		if (m_dictTickTockCumulative.count(name) > 0) {
			m_dictTickTockCumulative[name] += ::tock(m_dictTickTock[name],
					0.0f);
			m_dictTickTock.erase(name);
		} else {
			m_dictTickTockCumulative[name] = ::tock(m_dictTickTock[name], 0.0f);
			m_dictTickTock.erase(name);
		}
	} else {
		m_dictTickTock[name] = ::tick();
	}
}
void ClusterNet::tock() {
	tock("default");
}
void ClusterNet::tock(std::string name) {
	if (m_dictTickTockCumulative.count(name) > 0) {
		::tock("<<<Cumulative>>>: " + name, m_dictTickTockCumulative[name]);
		m_dictTickTockCumulative.erase(name);
	} else {
		if (m_dictTickTock.count(name) == 0)
			cout << "Error for name: " << name << endl;
		assert(
				("No tick event was registered for the name" + name, m_dictTickTock.count(
						name) > 0));
		::tock(m_dictTickTock[name], name);
		m_dictTickTock.erase(name);
	}
}

Matrix *ClusterNet::dropout(Matrix *A, float dropout_rate) {
	Matrix *out = rand(A->rows, A->cols);
	::dropout(A, out, dropout_rate);
	return out;
}

Matrix *ClusterNet::uniformSqrtWeight(int rows, int cols) {
	Matrix * out = rand(rows, cols);
	::uniformSqrtWeight(out);
	return out;
}

Matrix *ClusterNet::sparseInitWeight(int rows, int cols) {
	return sparseInitWeight(rows, cols, 15);
}
Matrix *ClusterNet::sparseInitWeight(int rows, int cols, int connections) {

	Matrix *rdm = randn(cols, connections);
	Matrix *idx = rand_int(cols, connections, 0, rows - 1);
	Matrix *out = zeros(rows, cols);
	sparseRdmWeight(rdm, idx, out, connections);

	cudaDeviceSynchronize();
	cudaFree(rdm->data);
	cudaFree(idx->data);

	return out;

}

Matrix *ClusterNet::distributed_uniformSqrtWeight(int rows, int cols) {
	assert(m_hasMPI);
	Matrix *W;
	int split_size = cols / MPI_SIZE;
	if (MYRANK < MPI_SIZE - 1)
		W = rand(rows, split_size);
	else
		W = rand(rows, split_size + (cols % split_size));

	W->isDistributed = 1;
	::uniformSqrtWeight(W);

	return W;
}

Matrix *ClusterNet::distributed_sparseInitWeight(int rows, int cols) {
	assert(m_hasMPI);
	int split_size = cols / MPI_SIZE;
	int col_size = MYRANK < MPI_SIZE - 1 ?  split_size + (cols % split_size) : split_size;
	int connections = 15;

	Matrix *W = zeros(rows, col_size);
	Matrix *rdm = randn(col_size, connections);
	Matrix *idx = rand_int(col_size, connections, 0, rows - 1);
	Matrix *out = zeros(rows, col_size);

	sparseRdmWeight(rdm, idx, out, connections);

	cudaDeviceSynchronize();
	cudaFree(rdm->data);
	cudaFree(idx->data);


	W->isDistributed = 1;

	return W;
}
