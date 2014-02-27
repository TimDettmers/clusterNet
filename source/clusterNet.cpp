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

Matrix *ClusterNet::dot(Matrix *A, Matrix *B)
{
	//if(m_hasMPI){ return dotMPI(A,B);}
	Matrix *out = zeros(A->shape[0],B->shape[1]);
	if(checkMatrixOperation(A, B, out, 1) == 1){ throw "Matrix *size error:\n"; }
	dot(A, B, out);

	return out;
}

Matrix *ClusterNet::dotMPI_batchSlice(Matrix *A, Matrix *B)
{
	int split_size = A->shape[0]/m_nodes;
	Matrix *out = empty(split_size,B->shape[1]);
	Matrix *out_rev = empty(split_size,B->shape[1]);

	tick("slice batch");
	Matrix *A1 = slice_rows(A, split_size*m_rank,split_size*(m_rank+1)-1);
	tick("slice batch");
	tick("dot batch");
	dot(A1,B,out);
	tick("dot batch");
	for(int i = 0; i < m_nodes; i++)
	{
		if(m_rank == i) { continue; }
		MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request));
		tick("send batch");
		MPI_Isend(out->data, out->size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, request);
		tick("send batch");
	}

	for(int i = 0; i < m_nodes; i++)
	{
		if(m_rank == i) { continue; }
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
	int split_size = B->shape[1]/m_nodes;
	std::string matrix_size = A->shape[0] + "x" + split_size;
	Matrix *out;
	Matrix *out_rev;
	if(m_matrixCache.count("out " + matrix_size) > 0)
	{
		out = m_matrixCache["out " + matrix_size];
		m_matrixCacheUsage["out " + matrix_size] -= 1;
	}
	else
	{
		out = empty(A->shape[0],split_size);
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
		out_rev = empty(A->shape[0],split_size);
		m_matrixCache["out_rev " + matrix_size] = out_rev;
		m_matrixCacheUsage["out_rev " + matrix_size] = 0;
	}

	tick("slice unit");
	Matrix *B1 = slice_cols(B, split_size*m_rank,split_size*(m_rank+1)-1);
	tick("slice unit");
	tick("dot unit");
	dot(A,B1,out);
	tick("dot unit");
	for(int i = 0; i < m_nodes; i++)
	{
		if(m_rank == i) { continue; }
		MPI_Request *request = (MPI_Request*)malloc(sizeof(MPI_Request));
		tick("send unit");
		MPI_Isend(out->data, out->size, MPI_FLOAT, i, 100, MPI_COMM_WORLD, request);
		tick("send unit");
	}

	for(int i = 0; i < m_nodes; i++)
	{
		if(m_rank == i) { continue; }
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

void ClusterNet::dot(Matrix *A, Matrix *B, Matrix *out)
{
	if(checkMatrixOperation(A, B, out, 1) == 1){ throw "Matrix *size error:\n"; }
	cublasStatus_t status;

	const float alpha = 1.0f;
	const float beta = 0.0f;

	status = cublasSgemm(m_handle, CUBLAS_OP_N, CUBLAS_OP_N,
				A->shape[0], B->shape[1], A->shape[1],
				&alpha, A->data, A->shape[0],
				B->data, B->shape[0],
				&beta, out->data, out->shape[0]);

	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "CUBLAS ERROR!\n";
		throw "CUBLAS ERROR";
	}
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

void ClusterNet::init_batch_allocator(Matrix *X, Matrix *y, float cross_validation_size, int batch_size, int batch_size_cv)
{

	float * pinned_memory_X;
	cudaHostAlloc(&pinned_memory_X, X->bytes, cudaHostAllocPortable);
	memcpy(pinned_memory_X,X->data,X->bytes);
	free(X->data);

	m_full_X = (Matrix*)malloc(sizeof(Matrix));
	m_full_X->shape[0] = X->shape[0];
	m_full_X->shape[1] = X->shape[1];
	m_full_X->bytes = X->bytes;
	m_full_X->size = X->size;
	m_full_X->data = pinned_memory_X;

	float * pinned_memory_y;
	cudaHostAlloc(&pinned_memory_y, y->bytes, cudaHostAllocPortable);
	memcpy(pinned_memory_y,y->data,y->bytes);
	free(y->data);

	m_full_y = (Matrix*)malloc(sizeof(Matrix));
	m_full_y->shape[0] = y->shape[0];
	m_full_y->shape[1] = y->shape[1];
	m_full_y->bytes = y->bytes;
	m_full_y->size = y->size;
	m_full_y->data = pinned_memory_y;

	m_batch_size = batch_size;
	m_batch_size_cv = batch_size_cv;
	m_cv_beginning = ceil(X->shape[0] - (X->shape[0]*cross_validation_size));
	m_total_batches = ceil(m_cv_beginning /(m_batch_size*1.0f));
	m_total_batches_cv = ceil((m_full_X->shape[0] - m_cv_beginning)/(m_batch_size_cv*1.0f));

	cudaStreamCreate(&m_streamNext_batch_X);
	cudaStreamCreate(&m_streamNext_batch_y);
	cudaStreamCreate(&m_streamNext_batch_cv_X);
	cudaStreamCreate(&m_streamNext_batch_cv_y);

	m_current_batch_X = empty(m_batch_size,m_full_X->shape[1]);
	m_next_batch_X = empty(m_batch_size,m_full_X->shape[1]);
	m_current_batch_y = empty(m_batch_size,m_full_y->shape[1]);
	m_next_batch_y = empty(m_batch_size,m_full_y->shape[1]);

	m_current_batch_cv_X = empty(m_batch_size_cv,m_full_X->shape[1]);
	m_next_batch_cv_X = empty(m_batch_size_cv,m_full_X->shape[1]);
	m_current_batch_cv_y = empty(m_batch_size_cv,m_full_y->shape[1]);
	m_next_batch_cv_y = empty(m_batch_size_cv,m_full_y->shape[1]);


	cudaMemcpy(&m_current_batch_X->data[0],&m_full_X->data[0],m_current_batch_X->bytes,cudaMemcpyDefault);
	cudaMemcpy(&m_current_batch_y->data[0],&m_full_y->data[0],m_current_batch_y->bytes,cudaMemcpyDefault);
	cudaMemcpy(&m_current_batch_cv_X->data[0],&m_full_X->data[m_cv_beginning*m_full_X->shape[1]],m_current_batch_cv_X->bytes,cudaMemcpyDefault);
	cudaMemcpy(&m_current_batch_cv_y->data[0],&m_full_y->data[m_cv_beginning*m_full_y->shape[1]],m_current_batch_cv_y->bytes,cudaMemcpyDefault);

	Matrix * X_T = to_col_major(m_current_batch_X);
	m_current_batch_X = X_T;
	Matrix * y_T = to_col_major(m_current_batch_y);
	m_current_batch_y = y_T;

	Matrix * X_T_cv = to_col_major(m_current_batch_cv_X);
	m_current_batch_cv_X = X_T_cv;
	Matrix * y_T_cv = to_col_major(m_current_batch_cv_y);
	m_current_batch_cv_y = y_T_cv;

	m_next_batch_number = 1;
	m_next_batch_number_cv = 1;
}


void ClusterNet::allocate_next_batch_async()
{
	int copy_range_bytes_X = m_next_batch_X->bytes;
	int copy_range_bytes_y = m_next_batch_y->bytes;

	if((m_batch_size * (m_next_batch_number + 1)) > m_cv_beginning)
	{
		//the next batch is smaller than the given standard batch size

		int partial_batch_size = m_cv_beginning % m_batch_size;
		copy_range_bytes_X = partial_batch_size*m_full_X->shape[1]*sizeof(float);
		copy_range_bytes_y = partial_batch_size*m_full_y->shape[1]*sizeof(float);
		cudaFree(m_next_batch_X->data);
		cudaFree(m_next_batch_y->data);
		m_next_batch_X = empty(partial_batch_size, m_full_X->shape[1]);
		m_next_batch_y = empty(partial_batch_size, m_full_y->shape[1]);
	}

	cudaMemcpyAsync(&m_next_batch_X->data[0],&m_full_X->data[(m_full_X->shape[1] * m_next_batch_number * m_batch_size)],
					copy_range_bytes_X, cudaMemcpyHostToDevice,m_streamNext_batch_X);
	cudaMemcpyAsync(&m_next_batch_y->data[0],&m_full_y->data[(m_full_y->shape[1] * m_next_batch_number * m_batch_size)],
					copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_y);
}

void ClusterNet::allocate_next_cv_batch_async()
{
	int copy_range_bytes_X = m_next_batch_cv_X->bytes;
	int copy_range_bytes_y = m_next_batch_cv_y->bytes;

	if((m_batch_size_cv * (m_next_batch_number_cv + 1)) > (m_full_X->shape[0] - m_cv_beginning))
	{
		//the next batch is smaller than the given standard batch size
		int partial_batch_size = (m_full_X->shape[0] - m_cv_beginning) % m_batch_size_cv;
		copy_range_bytes_X = partial_batch_size*m_full_X->shape[1]*sizeof(float);
		copy_range_bytes_y = partial_batch_size*m_full_y->shape[1]*sizeof(float);
		cudaFree(m_next_batch_cv_X->data);
		cudaFree(m_next_batch_cv_y->data);
		m_next_batch_cv_X = empty(partial_batch_size, m_full_X->shape[1]);
		m_next_batch_cv_y = empty(partial_batch_size, m_full_y->shape[1]);
	}

	cudaMemcpyAsync(&m_next_batch_cv_X->data[0],&m_full_X->data[(m_cv_beginning * m_full_X->shape[1])  + (m_next_batch_number_cv * m_batch_size_cv * m_full_X->shape[1])],
					copy_range_bytes_X, cudaMemcpyHostToDevice,m_streamNext_batch_cv_X);
	cudaMemcpyAsync(&m_next_batch_cv_y->data[0],&m_full_y->data[(m_cv_beginning * m_full_y->shape[1])  + (m_next_batch_number_cv * m_batch_size_cv * m_full_y->shape[1])],
					copy_range_bytes_y, cudaMemcpyHostToDevice,m_streamNext_batch_cv_y);
}

void ClusterNet::replace_current_batch_with_next()
{

	if(m_next_batch_X->shape[0] != m_batch_size)
	{
		cudaFree(m_current_batch_X->data);
		cudaFree(m_current_batch_y->data);
		m_current_batch_X = empty(m_next_batch_X->shape[0],m_next_batch_X->shape[1]);
		m_current_batch_y = empty(m_next_batch_y->shape[0],m_next_batch_y->shape[1]);
	}

	cudaStreamSynchronize(m_streamNext_batch_X);
	to_col_major(m_next_batch_X, m_current_batch_X);
	cudaStreamSynchronize(m_streamNext_batch_y);
	to_col_major(m_next_batch_y, m_current_batch_y);
	m_next_batch_number += 1;

	if(m_next_batch_number > m_total_batches)
	{
		//reset to the intial state
		m_next_batch_number = 0;
		if(m_current_batch_X->shape[0] != m_batch_size)
		{
			cudaFree(m_current_batch_X->data);
			cudaFree(m_next_batch_X->data);
			cudaFree(m_current_batch_y->data);
			cudaFree(m_next_batch_y->data);
			m_current_batch_X = empty(m_batch_size,m_full_X->shape[1]);
			m_next_batch_X = empty(m_batch_size,m_full_X->shape[1]);

			m_current_batch_y = empty(m_batch_size,m_full_y->shape[1]);
			m_next_batch_y = empty(m_batch_size,m_full_y->shape[1]);
		}
	}
}

void ClusterNet::replace_current_cv_batch_with_next()
{

	if(m_next_batch_cv_X->shape[0] != m_batch_size_cv)
	{
		cudaFree(m_current_batch_cv_X->data);
		cudaFree(m_current_batch_cv_y->data);
		m_current_batch_cv_X = empty(m_next_batch_cv_X->shape[0],m_next_batch_cv_X->shape[1]);
		m_current_batch_cv_y = empty(m_next_batch_cv_y->shape[0],m_next_batch_cv_y->shape[1]);
	}

	cudaStreamSynchronize(m_streamNext_batch_cv_X);
	to_col_major(m_next_batch_cv_X,m_current_batch_cv_X);
	cudaStreamSynchronize(m_streamNext_batch_cv_y);
	to_col_major(m_next_batch_cv_y,m_current_batch_cv_y);
	m_next_batch_number_cv += 1;

	if(m_next_batch_number_cv > m_total_batches_cv)
	{
		//std::cout << "reset size" << std::endl;
		//reset to the intial state
		m_next_batch_number_cv = 0;
		if(m_current_batch_cv_X->shape[0] != m_batch_size_cv)
		{
			cudaFree(m_current_batch_cv_X->data);
			cudaFree(m_next_batch_cv_X->data);
			cudaFree(m_current_batch_cv_y->data);
			cudaFree(m_next_batch_cv_y->data);
			m_current_batch_cv_X = empty(m_batch_size_cv,m_full_X->shape[1]);
			m_next_batch_cv_X = empty(m_batch_size_cv,m_full_X->shape[1]);

			m_current_batch_cv_y = empty(m_batch_size_cv,m_full_y->shape[1]);
			m_next_batch_cv_y = empty(m_batch_size_cv,m_full_y->shape[1]);
		}
	}
}

void ClusterNet::finish_batch_allocator()
{
	cudaStreamDestroy(m_streamNext_batch_X);
	cudaStreamDestroy(m_streamNext_batch_y);
	cudaStreamDestroy(m_streamNext_batch_cv_X);
	cudaStreamDestroy(m_streamNext_batch_cv_y);
}




