#ifndef ClusterNet_H
#define ClusterNet_H

#include <cublas_v2.h>
#include <basicOps.cuh>
#include <curand.h>
#include <mpi.h>
#include <list>
#include <vector>
#include <pthread.h>
#include <cstdlib>
#include <iostream>
#include <cusparse_v2.h>
#include <util.cuh>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <pthread.h>
#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <unistd.h>

typedef enum Unittype_t
{
	Logistic = 0,
	Rectified_Linear = 1,
	Softmax = 2,
	Linear = 4,
	Double_Rectified_Linear = 8,
	Input = 16
} Unittype_t;

typedef enum DataPropagationType_t
{
	Training = 0,
	Trainerror = 1,
	CVerror = 2
} DataPropagationType_t;


typedef enum WeightUpdateType_t
{
	NesterovRMSProp = 0,
	NesterovMomentum = 1,
	RMSProp = 2,
	Momentum = 4,
	NoMomentum = 8
} WeightUpdateType_t;


typedef enum Costfunction_t
{
	Cross_Entropy = 0,
	Squared_Error = 1,
	Root_Squared_Error = 2,
	Misclassification = 4
} Costfunction_t;


class ClusterNet
{


public:
	 ClusterNet();
	 ClusterNet(int seed);
	 ClusterNet(int argc, char* argv[]);
	 ClusterNet(int argc, char *argv[], int seed);
	 ClusterNet(int argc, char* argv[], int seed, bool useSameSeed);

	 void dotPCIe(Matrix **A, Matrix **B, Matrix **out);
	 void dotTPCIe(Matrix **A, Matrix **B, Matrix **out);
	 void TdotPCIe(Matrix **A, Matrix **B, Matrix **out);
	 Matrix *dot(Matrix *A, Matrix *B);
	 Matrix *Tdot(Matrix *A, Matrix *B);
	 Matrix *dotT(Matrix *A, Matrix *B);
	 void dot(Matrix *A, Matrix *B, Matrix *out);
	 void Tdot(Matrix *A, Matrix *B, Matrix *out);
	 void dotT(Matrix *A, Matrix *B, Matrix *out);
	 Matrix *dotTMPI(Matrix *A, Matrix *B);
	 Matrix *TdotMPI(Matrix *A, Matrix *B);
	 Matrix *dotMPI(Matrix *A, Matrix *B);
	 void dotMPI(Matrix *A, Matrix *B, Matrix *out);
	 void TdotMPI(Matrix *A, Matrix *B, Matrix *out);
	 void dotTMPI(Matrix *A, Matrix *B, Matrix *out);

	 void add_PCIe(Matrix **A, Matrix **B, Matrix **out);
	 void mul_PCIe(Matrix **A, Matrix **B, Matrix **out);
	 void scalarMul_PCIe(Matrix **A, float a, Matrix **out);
	 void addMatrixVector_PCIe(Matrix **A, Matrix **v, Matrix **out);
	 void logistic_PCIe(Matrix **A, Matrix **out);

	 Matrix *dot_sparse(Matrix *A, Matrix *B);
	 Matrix *Tdot_sparse(Matrix *A, Matrix *B);
	 Matrix *dotT_sparse(Matrix *A, Matrix *B);
	 void dot_sparse(Matrix *A, Matrix *B, Matrix *out);
	 void dotT_sparse(Matrix *A, Matrix *B, Matrix *out);
	 void Tdot_sparse(Matrix *A, Matrix *B, Matrix *out);

	 Matrix *distribute_rows_hdf5_file(std::string path);

	 void RMSprop_with_nesterov_weight_update_PCIe(Matrix **RMS, Matrix **grad, Matrix **w, Matrix **m, float RMS_multiplier, float learning_rate, int batch_size, float momentum);

	 Matrix *rand(int rows, int cols);
	 Matrix *rand_same_seed_MPI(int rows, int cols);
	 void rand(int rows, int cols, bool useSameSeedGenerator, Matrix *out);


	 Matrix *randn(int rows, int cols);
	 Matrix *randn(int rows, int cols, float mean, float std);
	 void randn(int rows, int cols, float mean, float std, Matrix *out);
	 Matrix *dropout(Matrix *A, float dropout_rate);
	 void dropout(Matrix *A, Matrix *out, float dropout_rate);
	 Matrix *rand_int(int rows, int cols, int low, int high);

	 Matrix *compression_8bit(Matrix *A, float precision);
	 void compression_8bit(Matrix *A, float precision, Matrix *out);
	 Matrix *decompression_8bit(Matrix *A, float precision);
	 void decompression_8bit(Matrix *A, float precision, Matrix *out);
	 Matrix *compression_8bit_test(Matrix *A, float precision);
	 void compression_8bit_test(Matrix *A, float precision, Matrix *out);

	 void tick(std::string name);
	 void tick();
	 void tock(std::string name);
	 void tock();

	 void benchmark_dot();
	 void shutdown_MPI();
	 Matrix *distributed_uniformSqrtWeight(int rows, int cols);
	 Matrix *distributed_sparseInitWeight(int rows, int cols);
	 Matrix *distributed_zeros(int rows, int cols);
	 Matrix *distributed_ones(int rows, int cols);
	 Matrix *uniformSqrtWeight(int rows, int cols);
	 Matrix *uniformSqrtWeight(int rows, int cols, int rows_stacked, int cols_stacked);
	 Matrix *uniformSqrtWeight_sameSeed(int rows, int cols);
	 Matrix *sparseInitWeight(int rows, int cols);
	 Matrix *sparseInitWeight(int rows, int cols, int connections);

	 Matrix *dense_to_sparse(Matrix *A);
	 Matrix *sparse_to_dense(Matrix *A);

	 void construct_vocab_matrix(Matrix *vocab_idx, Matrix *vocab_idx_y, Matrix *batch_X, Matrix *batch_y, Matrix *vocab);
	 void add_to_queue(Matrix **gpuArray);
	 bool pop_queue();
	 void add_to_queue_PCIe(Matrix **gpuArray);
	 bool pop_queue_PCIe();
	 int get_queue_length();

	 void addGradients_PCIe(Matrix **grad);

	 Matrix **zeros_PCIe(int rows, int cols);
	 Matrix **zeros_stacked(int rows, int cols);
	 Matrix **zeros_gradient_PCIe(int rows, int cols);
	 Matrix **uniformSqrtWeight_stacked(int rows, int cols);
	 Matrix **ones_PCIe(int rows, int cols);
	 Matrix **uniformSqrtWeight_PCIe(int rows, int cols);

	 bool QUEUE_EMPTY;

	 bool StartBackgroundQueue;
	 int MYRANK;
	 int NODES;
	 int MYGPUID;
	 int MPI_SIZE;
	 int GPU_COUNT;
	 std::vector<int> PCIe_RANKS;
	 std::vector<int> MASTER_GPU_RANKS;

	 void *hello(void)
	 {
		 bool uden = true;
		 std::cout << "test kek" << std::endl;
		 while(uden)
		 {
			 pop_queue_PCIe();
			 usleep(100);
		 }

		 return 0;
	 }

	 static void *hello_helper(void *context)
	 {
		 return ((ClusterNet *)context)->hello();
	 }

private:
	 std::vector<cublasHandle_t> m_handle;
	 cusparseHandle_t m_sparse_handle;
	 curandGenerator_t m_generator;
	 curandGenerator_t m_generator_same_seed;
	 std::map<std::string,cudaEvent_t*> m_dictTickTock;
	 std::map<std::string,float> m_dictTickTockCumulative;
	 MPI_Request* m_requests;
	 MPI_Request m_sendrequest;
	 std::vector<MPI_Request> m_sendrequests;
	 std::map<std::string,Matrix**> m_matrixCache;
	 std::map<std::string,float**> m_matrixHStackCache;
	 std::map<std::string,int> m_matrixCacheUsage;
	 int m_gpucount;
	 pthread_t *m_threads;

	 Matrix *flt_tbl;

	 bool m_hasMPI;
	 bool m_cublasInitialized;
	 bool m_cusparseInitialized;
	 bool waitingForTransfer;
	 MPI_Status m_status;
	 MPI_Comm m_MPIWorld;
	 MPI_Request *m_request_queue;
	 int *m_flag_queue;
	 std::vector<Matrix*> m_send_queue;
	 std::vector<Matrix*> m_receive_queue;
	 std::vector<int> m_sendid_queue;
	 std::vector<int> m_receiveid_queue;
	 std::vector<cudaStream_t> m_streams_PCIe;

	 int m_destination;
	 int m_source;

	 void dot(Matrix *A, Matrix *B, Matrix *out, cublasOperation_t T1, cublasOperation_t T2);
	 void dot_sparse(Matrix *A, Matrix *B, Matrix *out, cublasOperation_t T1, cublasOperation_t T2);
	 void dotMPI(Matrix *A, Matrix *B, Matrix *out, bool applyTranspose_A, bool applyTranspose_B);
	 void init(int seed);
	 void init_MPI(int argc, char *argv[]);

	 void compute_PCIe_ranks();
	 void compute_GPUID_and_Nodes();


};
#endif

