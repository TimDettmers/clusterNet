#ifndef ClusterNet_H
#define ClusterNet_H

#include <cublas_v2.h>
#include <basicOps.cuh>
#include <curand.h>
#include <mpi.h>
#include <list>
#include <vector>


class ClusterNet
{


public:
	 ClusterNet();
	 ClusterNet(int seed);
	 ClusterNet(int argc, char* argv[]);
	 ClusterNet(int argc, char *argv[], int seed);

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

	 Matrix *rand(int rows, int cols);
	 void rand(int rows, int cols, Matrix *out);
	 Matrix *randn(int rows, int cols);
	 Matrix *randn(int rows, int cols, float mean, float std);
	 void randn(int rows, int cols, float mean, float std, Matrix *out);
	 Matrix *dropout(Matrix *A, float dropout_rate);
	 Matrix *rand_int(int rows, int cols, int low, int high);

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
	 Matrix *sparseInitWeight(int rows, int cols);
	 Matrix *sparseInitWeight(int rows, int cols, int connections);

	 int MYRANK;
	 int NODES;
	 int MYGPUID;
	 int MPI_SIZE;
	 std::vector<int> PCIe_RANKS;
	 std::vector<int> MASTER_GPU_RANKS;
private:
	 cublasHandle_t m_handle;
	 curandGenerator_t m_generator;
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

	 bool m_hasMPI;
	 bool m_cublasInitialized;
	 MPI_Status m_status;
	 MPI_Comm m_MPIWorld;

	 int m_destination;
	 int m_source;

	 void dot(Matrix *A, Matrix *B, Matrix *out, cublasOperation_t T1, cublasOperation_t T2);
	 void dotMPI(Matrix *A, Matrix *B, Matrix *out, bool applyTranspose_A, bool applyTranspose_B);
	 void init(int seed);
	 void init_MPI(int argc, char *argv[]);

	 void compute_PCIe_ranks();
	 void compute_GPUID_and_Nodes();
};
#endif

