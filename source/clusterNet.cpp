#include <clusterNet.h>


#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

using std::cout;
using std::endl;


ClusterNet::ClusterNet()
{
	init((int) (time(0) + (10000*MYRANK+12345) % 10000));
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
ClusterNet::ClusterNet(int argc, char* argv[], int seed, bool useSameSeed)
{
	init_MPI(argc, argv);
	if(useSameSeed)
		init(seed);
	else
		init(seed + (10000*MYRANK+12345));
}
ClusterNet::ClusterNet(int argc, char* argv[])
{
	init_MPI(argc, argv);
	init((int) (time(0) + (10000*MYRANK+12345) % 10000));
}

int ClusterNet::get_queue_length(){ return m_send_queue.size(); }

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
	//cout << "seed: " << seed << endl;

	curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_generator, seed);
	curandSetGeneratorOffset(m_generator, 100);


	curandCreateGenerator(&m_generator_same_seed, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_generator_same_seed, 12345);
	curandSetGeneratorOffset(m_generator_same_seed, 100);




	char buff[4096];
	ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
	std::string path = std::string(buff);
	replace(path,"/build/clusterNet.out","/source/");
	flt_tbl = to_gpu(read_hdf5("/home/tim/git/clusterNet/source/8bit_floats.hdf5"));

	int current_device = 0;
	cudaGetDevice(&current_device);
	cout << "Active device: GPU" << current_device << endl;


	seeds = rand_int(512,128,0,2147483647);
	/*
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		curandGenerator_t gen;
		curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen, seed);
		curandSetGeneratorOffset(gen, 100);
		m_generator.push_back(gen);
	}
	*/



	m_cublasInitialized = false;
	m_cusparseInitialized = false;


	cudaGetDeviceCount(&GPU_COUNT);

	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		cudaStream_t t;
		cudaStreamCreate(&t);
		m_streams_PCIe.push_back(t);

		//for(int j = 0; j < GPU_COUNT; j++)
			//if(i != j)
				//cudaDeviceEnablePeerAccess(j,0);
	}

	cudaSetDevice(current_device);



	if (!m_hasMPI)
	{
		MYGPUID = 0;
		NODES = 1;
		PCIe_RANKS.push_back(0);
		MYRANK = 0;
	}




	m_request_queue = (MPI_Request*)malloc(sizeof(MPI_Request)*2);
	m_flag_queue = (int*)malloc(sizeof(int));
	m_flag_queue[0] = 0;
	m_request_queue[0] = MPI_REQUEST_NULL;
	m_request_queue[1] = MPI_REQUEST_NULL;
	QUEUE_EMPTY = true;
	waitingForTransfer = false;


	StartBackgroundQueue = false;




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
	if(B->isDistributed == 1 || out->isDistributed == 1)
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
		 if(out->rows == 9000 && out->cols == 100)
		 {


			 Matrix *s1 = sparse_to_dense(A);
			 s1 = to_host(dense_to_sparse(s1));
			 Matrix *s2 = to_host(s1);

			 for(int i = 0; i < s1->size; i++)
			 {
				 ASSERT(s1->data[i] ==s2->data[i],"test");
				 ASSERT(s1->idx_cols[i] ==s2->idx_cols[i],"test");
			 }

			 for(int i = 0; i < s1->rows+1; i++)
				 ASSERT(s1->ptr_rows[i] == s2->ptr_rows[i],"test");
				//B = zeros(B->rows,B->cols);
				//out = zeros(out->rows,out->cols);


			 cout << "bytes: " << A->bytes << " vs " << s1->bytes << endl;
			 cout << "bytes 2: " << A->ptr_bytes << " vs " << s1->ptr_bytes << endl;
			 cout << "bytes 3: " << A->idx_bytes << " vs " << s1->idx_bytes << endl;
			 cout << "size: " << A->size << " vs " << s1->size << endl;



		 }
		 */




	status = cusparseScsrmm2(m_sparse_handle,
		T1 == CUBLAS_OP_N ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE,
		T2 == CUBLAS_OP_N ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE,
				A->rows, B_cols, A->cols,
		A->size, &alpha, descriptor_A,
		A->data, A->ptr_rows, A->idx_cols,
		B->data, B->rows,  &beta,
		out->data, out->rows);



	/*
		 cout << "T1: " << T1 << endl;
		 cout << "T2: " << T2 << endl;
		 cout << "A rows: " << A->rows << endl;
		 cout << "A cols: " << A->cols << endl;
		 cout << "B rows: " << B->rows << endl;
		 cout << "B cols: " << B_cols << endl;
		 cout << "out rows: " << out->rows << endl;
		 cout << "out cols: " << out->cols << endl;
		 cout << "A distributed: " << A->isDistributed << endl;
		 cout << "B distributed: " << B->isDistributed << endl;
		 cout << "out distributed: " << out->isDistributed << endl;

		 cout << "sum A: " << sum(A) << endl;
		 cout << "sum B: "  << sum(B) << endl;
		 cout << "sum out: " << sum(out) << endl;
		 */



	if (status != CUSPARSE_STATUS_SUCCESS)
	{
		cout << "CUSPARSE ERROR: " << status <<  "!" << endl;
		throw "CUSPARSE ERROR!";
	}



}


void ClusterNet::dot(Matrix *A, Matrix *B, Matrix *out, cublasOperation_t T1, cublasOperation_t T2)
{
	int current_device = 0;
	cudaGetDevice(&current_device);

	if(A->isSparse == 0)
	{
		if(checkMatrixOperation(A, B, out, T1, T2, 1) == 1){ throw "Matrix *size error:\n"; }
		cublasStatus_t status;
		if(!m_cublasInitialized)
		{
			m_cublasInitialized = true;
			int gpus = 0;
			cudaGetDeviceCount(&gpus);
			for(int i = 0; i < gpus; i++)
			{
				cudaSetDevice(i);
				cublasHandle_t handle;
				cublasCreate_v2(&handle);
				m_handle.push_back(handle);
			}
			cudaSetDevice(current_device);
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
		 int B_rows = T2 == CUBLAS_OP_T ? B->cols : B->rows;
		 cout << "T1: " << T1 << endl;
		 cout << "T2: " << T2 << endl;
		 cout << "A rows: " << A_rows << endl;
		 cout << "A cols: " << A_cols << endl;
		 cout << "B rows: " << B_rows << endl;
		 cout << "B cols: " << B_cols << endl;
		 cout << "out rows: " << out->rows << endl;
		 cout << "out cols: " << out->cols << endl;
		 cout << "sum A: " << sum(A) << endl;
		 cout << "sum B: "  << sum(B) << endl;
		 cout << "sum out: " << sum(out) << endl;
		 */

		 //MPI_Barrier(MPI_COMM_WORLD);


		status = cublasSgemm(m_handle[current_device], T1, T2, A_rows, B_cols,
				A_cols, &alpha, A->data, A->rows, B->data, B->rows, &beta,
				out->data, out->rows);

		if (status != CUBLAS_STATUS_SUCCESS)
		{
			printmat(A,0,1,0,10);
			printmat(A,A->rows-1,A->rows, A->cols-10,A->cols);
			std::cout << "CUBLAS ERROR: Status " << status << std::endl;
			throw "CUBLAS ERROR";

		}


			/*
		if( T1 == CUBLAS_OP_N && T2 == CUBLAS_OP_N)
			matmul(A,B,out,0,0);
		else if( T1 == CUBLAS_OP_N && T2 == CUBLAS_OP_T)
			matmul(A,B,out,0,1);
		else
			matmul(A,B,out,1,0);
			*/

	}
	else
	{
		dot_sparse(A,B,out,T1,T2);
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
								"T" + SSTR((applyTranspose_B ? 1 : 0)) + " Sparse: "  + SSTR((A->isSparse == 1 ? 1 : 0));

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
				Matrix** arrOut2 = (Matrix**) malloc(sizeof(Matrix*) * MPI_SIZE);
				for (int i = 0; i < MPI_SIZE; i++)
				{
					if (i == MPI_SIZE - 1)
						arrOut2[i] = empty_char(A->rows, col_split_size + remainder);
					else
						arrOut2[i] = empty_char(A->rows, col_split_size);
				}
				m_matrixCacheChar[strMatrixName] = arrOut2;
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
			if(A->isSparse == 1)
				cout << "Sparse distributed dotT is not supported!" << endl;
			assert(A->isSparse == 0);
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
			/*
			compression_8bit(m_matrixCache[strMatrixName][matrix_idx], 1.0f, m_matrixCacheChar[strMatrixName][matrix_idx]);
			MPI_Isend(m_matrixCacheChar[strMatrixName][matrix_idx]->char_data, m_matrixCacheChar[strMatrixName][matrix_idx]->size, MPI_CHAR, m_destination, 100, MPI_COMM_WORLD, &m_sendrequests[i]);
			matrix_idx = (matrix_idx - 1) < 0 ? MPI_SIZE - 1 : (matrix_idx - 1);
			MPI_Recv(m_matrixCacheChar[strMatrixName][matrix_idx]->char_data, m_matrixCacheChar[strMatrixName][matrix_idx]->size, MPI_CHAR, m_source, 100, MPI_COMM_WORLD, &m_status);

			decompression_8bit(m_matrixCacheChar[strMatrixName][matrix_idx], 1.0f, m_matrixCache[strMatrixName][matrix_idx]);
			*/
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


Matrix *ClusterNet::rand_numbers(int rows, int cols)
{
	return ::rand_numbers(rows, cols, seeds);
}
void ClusterNet::rand_numbers(int rows, int cols, Matrix *out)
{
	::rand_numbers(out, seeds);
}

//Uniform
Matrix *ClusterNet::rand(int rows, int cols)
{
	Matrix *out = empty(rows, cols);

	rand(rows, cols, false, out);

	return out;
}

Matrix *ClusterNet::rand_same_seed_MPI(int rows, int cols)
{
	Matrix *out = empty(rows, cols);

	rand(rows, cols, true, out);

	return out;
}
void ClusterNet::rand(int rows, int cols, bool useSameSeedGenerator, Matrix *out)
{
	if(useSameSeedGenerator)
		curandGenerateUniform(m_generator_same_seed, out->data, rows * cols);
	else
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
	int current_device = 0;
	cudaGetDevice(&current_device);
	curandGenerateNormal(m_generator, out->data, rows * cols, mean, std);
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
float ClusterNet::tock(){ return tock("default"); }
float ClusterNet::tock(std::string name)
{
	if (m_dictTickTockCumulative.count(name) > 0)
	{
		::tock("<<<Cumulative>>>: " + name, m_dictTickTockCumulative[name]);
		float value = m_dictTickTockCumulative[name];
		m_dictTickTockCumulative.erase(name);
		return value;
	}
	else
	{
		if (m_dictTickTock.count(name) == 0)
			cout << "Error for name: " << name << endl;
		assert(("No tick event was registered for the name" + name, m_dictTickTock.count(name) > 0));
		float value = ::tock(m_dictTickTock[name], name);
		m_dictTickTock.erase(name);
		return value;
	}
}

Matrix *ClusterNet::dropout(Matrix *A, float dropout_rate)
{
	Matrix *out;
	if(A->isSparse == 0)
	{
		out = rand(A->rows, A->cols);
	}
	else
	{
		int current_device = 0;
		cudaGetDevice(&current_device);
		out = empty_sparse(A->rows,A->cols,A->size);
		curandGenerateUniform(m_generator, out->data, A->size);
	}
	::dropout(A, out, dropout_rate);
	return out;
}

void ClusterNet::dropout(Matrix *A, Matrix *out, float dropout_rate)
{
	int current_device = 0;
	cudaGetDevice(&current_device);
	if(A->isSparse == 0)
		curandGenerateUniform(m_generator, out->data, out->rows*out->cols);
	else
	{
		out = empty_sparse(A->rows,A->cols,A->size);
		curandGenerateUniform(m_generator, out->data, A->size);
	}
	::dropout(A, out, dropout_rate);
}

Matrix *ClusterNet::uniformSqrtWeight(int rows, int cols)
{
	Matrix * out = rand(rows, cols);
	::uniformSqrtWeight(out);
	return out;
}

Matrix *ClusterNet::uniformSqrtWeight(int rows, int cols, int rows_stacked, int cols_stacked)
{
	Matrix * out = rand(rows, cols);
	::uniformSqrtWeight(out, rows_stacked, cols_stacked);
	return out;
}

Matrix *ClusterNet::uniformSqrtWeight_sameSeed(int rows, int cols)
{
	Matrix *out = empty(rows, cols);
	rand(rows, cols,true,out);
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
	if(cols == 1)
	{
		cout << "Warning: Columns size 1, cannot split by column! Create normal uniformSqrtWeight( instead!" << endl;
		return uniformSqrtWeight_sameSeed(rows, cols);
	}

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
	if(cols == 1)
	{
		cout << "Warning: Columns size 1, cannot split by column! Create normal zeros instead!" << endl;
		return ::zeros(rows, cols);
	}

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

void ClusterNet::construct_vocab_matrix(Matrix *vocab_idx, Matrix *vocab_idx_y, Matrix *batch_X, Matrix *batch_y, Matrix *vocab)
{
	Matrix *rdm_idx = empty(batch_X->rows, 1);

	rand(batch_X->rows, 1, true, rdm_idx);
	::rand_int(rdm_idx, 0, vocab->cols-1);
	//cout << "MYRANK: " << MYRANK << " rdm 2: " << sum(rdm_idx) << endl;
	::construct_vocab_matrix(vocab_idx,vocab_idx_y,batch_X,batch_y,vocab,rdm_idx);
	cudaFree(rdm_idx->data);
	free(rdm_idx);

}

void ClusterNet::add_to_queue(Matrix **gpuArray)
{
	if(MPI_SIZE > 1)
	{
		int send_matrix_idx = (MYRANK + 1) == MPI_SIZE ? 0 : (MYRANK + 1);
		int receive_matrix_idx = (MYRANK - 1) < 0 ? MPI_SIZE - 1 : (MYRANK - 1);
		for(int i = 0; i < MPI_SIZE - 1; i++)
		{
			m_receive_queue.push_back(gpuArray[receive_matrix_idx]);
			m_receiveid_queue.push_back(receive_matrix_idx);
			m_send_queue.push_back(gpuArray[MYRANK]);
			m_sendid_queue.push_back(send_matrix_idx);

			send_matrix_idx = (send_matrix_idx + 1) == MPI_SIZE ? 0 : (send_matrix_idx + 1);
			receive_matrix_idx = (receive_matrix_idx - 1) < 0 ? MPI_SIZE - 1 : (receive_matrix_idx - 1);
		}

		QUEUE_EMPTY = false;
	}

	//pop_queue();
}



bool ClusterNet::pop_queue()
{
	if(QUEUE_EMPTY){ return true;}


	if(!waitingForTransfer)
	{
		MPI_Irecv(m_receive_queue[0]->data, m_receive_queue[0]->size, MPI_FLOAT, m_receiveid_queue[0], m_receiveid_queue[0], MPI_COMM_WORLD, &m_request_queue[0]);
		MPI_Isend(m_send_queue[0]->data, m_send_queue[0]->size, MPI_FLOAT, m_sendid_queue[0], MYRANK, MPI_COMM_WORLD, &m_request_queue[1]);
		waitingForTransfer = true;
	}
	else
	{
		//MPI_Waitall(2,m_request_queue,MPI_STATUSES_IGNORE);
		MPI_Testall(2,m_request_queue,m_flag_queue,MPI_STATUSES_IGNORE);


		usleep(10);

		if(m_flag_queue[0] == 1)
		{
			waitingForTransfer = false;
			m_flag_queue[0] = 0;
			m_receive_queue.erase(m_receive_queue.begin() + 0);
			m_send_queue.erase(m_send_queue.begin() + 0);
			m_receiveid_queue.erase(m_receiveid_queue.begin() + 0);
			m_sendid_queue.erase(m_sendid_queue.begin() + 0);
			if(m_receive_queue.size() > 0)
				pop_queue();
			else
				QUEUE_EMPTY = true;
		}

	}

	return QUEUE_EMPTY;
}

Matrix **ClusterNet::zeros_PCIe(int rows, int cols)
{
	Matrix **out = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);

	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		out[i] = zeros(rows,cols);
	}

	return out;

}


Matrix **ClusterNet::zeros_gradient_PCIe(int rows, int cols)
{
	Matrix **out = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT*GPU_COUNT);

	for(int j = 0; j < GPU_COUNT; j++)
	{
		for(int i = 0; i < GPU_COUNT; i++)
		{
			cudaSetDevice(i);
			out[i+(j*GPU_COUNT)] = zeros(rows,cols);
		}
	}

	return out;
}


Matrix **ClusterNet::zeros_stacked(int rows, int cols)
{
	Matrix **out = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);

	for(int i = 0; i < GPU_COUNT; i++)
	{
		out[i] = zeros(rows,cols);
	}

	return out;
}

Matrix **ClusterNet::uniformSqrtWeight_stacked(int rows, int cols)
{
	Matrix **out = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);

	for(int i = 0; i < GPU_COUNT; i++)
	{
		out[i] = uniformSqrtWeight(rows,cols);
	}

	return out;
}

Matrix **ClusterNet::ones_PCIe(int rows, int cols)
{
	Matrix **out = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);

	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		out[i] = zeros(rows,cols);
	}

	return out;

}

Matrix **ClusterNet::uniformSqrtWeight_PCIe(int rows, int cols)
{
	Matrix **out = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);

	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		out[i] = uniformSqrtWeight(rows,cols);
	}

	return out;

}

void ClusterNet::dotPCIe(Matrix **A, Matrix **B, Matrix **out)
{
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		dot(A[i],B[i],out[i]);
	}

}

void ClusterNet::dotTPCIe(Matrix **A, Matrix **B, Matrix **out)
{
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		dotT(A[i],B[i],out[i]);
	}

}

void ClusterNet::TdotPCIe(Matrix **A, Matrix **B, Matrix **out)
{
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		Tdot(A[i],B[i],out[i]);
	}

}



void ClusterNet::add_PCIe(Matrix **A, Matrix **B, Matrix **out)
{
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		add(A[i],B[i],out[i]);
	}
}

void ClusterNet::mul_PCIe(Matrix **A, Matrix **B, Matrix **out)
{
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		mul(A[i],B[i],out[i]);
	}
}

void ClusterNet::scalarMul_PCIe(Matrix **A, float a, Matrix **out)
{
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		scalarMul(A[i],a,out[i]);
	}
}


void ClusterNet::addMatrixVector_PCIe(Matrix **A, Matrix **v, Matrix **out)
{
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		addMatrixVector(A[i],v[i],out[i]);
	}
}

void ClusterNet::logistic_PCIe(Matrix **A, Matrix **out)
{
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		logistic(A[i],out[i]);
	}
}


void ClusterNet::RMSprop_with_nesterov_weight_update_PCIe(Matrix **RMS, Matrix **grad, Matrix **w, Matrix **m, float RMS_multiplier, float learning_rate, int batch_size, float momentum)
{
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		RMSprop_with_nesterov_weight_update(RMS[i],grad[i],w[i],m[i],RMS_multiplier,learning_rate,batch_size,momentum);
	}
}


void ClusterNet::add_to_queue_PCIe(Matrix **gpuArray)
{
	for(int j = 1; j < GPU_COUNT; j++)
	{
		int receive_matrix_idx = j;
		for(int i = 0; i < GPU_COUNT; i++)
		{
			m_send_queue.push_back(gpuArray[i]);
			m_sendid_queue.push_back(i);
			m_receive_queue.push_back(gpuArray[receive_matrix_idx+(GPU_COUNT*j)]);
			m_receiveid_queue.push_back(receive_matrix_idx);

			cout << "added " << i << " to " << receive_matrix_idx << " for entry with idx " << receive_matrix_idx+(GPU_COUNT*j) << endl;

			receive_matrix_idx = receive_matrix_idx + 1 == GPU_COUNT ? 0 : receive_matrix_idx + 1;
		}
	}


	QUEUE_EMPTY = false;

	if(StartBackgroundQueue == false)
	{
		StartBackgroundQueue = true;
		pthread_t t;
		//pthread_create(&t,NULL,&ClusterNet::hello_helper,this);
	}
}



bool ClusterNet::pop_queue_PCIe()
{
	if(QUEUE_EMPTY){ return true;}



		cout << "copy stream" << endl;
		for(int i = 0; i < GPU_COUNT; i++)
		{
			cout << m_sendid_queue[i] << " to " << m_receiveid_queue[i] << endl;
			cudaSetDevice(m_sendid_queue[i]);
			cout << "receive: " << m_receive_queue[i]->bytes << " send: " << m_send_queue[i]->bytes << endl;
			cudaMemcpyPeer(m_receive_queue[i]->data, m_receiveid_queue[i],m_send_queue[i]->data,m_sendid_queue[i],m_send_queue[i]->bytes);
		}

		for(int i = 0; i < GPU_COUNT; i++)
		{
			m_receive_queue.erase(m_receive_queue.begin() + 0);
			m_send_queue.erase(m_send_queue.begin() + 0);
			m_receiveid_queue.erase(m_receiveid_queue.begin() + 0);
			m_sendid_queue.erase(m_sendid_queue.begin() + 0);
		}
		if(m_receive_queue.size() > 0)
			pop_queue_PCIe();
		else
			QUEUE_EMPTY = true;


	return QUEUE_EMPTY;
}

void ClusterNet::compression_8bit_test(Matrix *A, float precision,  Matrix *out)
{
	::compression_8bit_test(flt_tbl, A, precision, out);
}

Matrix *ClusterNet::compression_8bit_test(Matrix *A, float precision)
{
	Matrix *out = empty(A->rows,A->cols);
	::compression_8bit_test(flt_tbl, A, precision, out);
	return out;
}

void ClusterNet::compression_8bit(Matrix *A, float precision,  Matrix *out)
{
	::compression_8bit(flt_tbl, A, precision, out);
}

Matrix *ClusterNet::compression_8bit(Matrix *A, float precision)
{
	Matrix *out = empty_char(A->rows,A->cols);
	::compression_8bit(flt_tbl, A, precision, out);
	return out;
}

void ClusterNet::decompression_8bit(Matrix *A, float precision,  Matrix *out)
{
	::decompression_8bit(flt_tbl, A, precision, out);
}

Matrix *ClusterNet::decompression_8bit(Matrix *A, float precision)
{
	Matrix *out = empty(A->rows,A->cols);
	::decompression_8bit(flt_tbl, A, precision, out);
	return out;
}


void ClusterNet::dot8bit(Matrix *A, Matrix *B, float precisionA, float precisionB,  Matrix *out)
{
	::dot8bit(A,B,out,flt_tbl, precisionA, precisionB);
}

Matrix *ClusterNet::dot8bit(Matrix *A, Matrix *B, float precisionA, float precisionB)
{
	Matrix *out = empty(A->rows,A->cols);
	::dot8bit(A,B,out,flt_tbl, precisionA, precisionB);
	return out;
}

void ClusterNet::dot8bit_shared(Matrix *A, Matrix *B, float precisionA, float precisionB,  Matrix *out)
{
	::dot8bit_shared(A,B,out,flt_tbl, precisionA, precisionB);
}

Matrix *ClusterNet::dot8bit_shared(Matrix *A, Matrix *B, float precisionA, float precisionB)
{
	Matrix *out = empty(A->rows,A->cols);
	::dot8bit_shared(A,B,out,flt_tbl, precisionA, precisionB);
	return out;
}



void ClusterNet::addGradients_PCIe(Matrix **grad)
{
	for(int j = 1; j < GPU_COUNT; j++)
	{
		int receive_matrix_idx = j;
		for(int i = 0; i < GPU_COUNT; i++)
		{
			cudaSetDevice(i);
			add(grad[i],grad[receive_matrix_idx],grad[i]);

			receive_matrix_idx = receive_matrix_idx + 1 == GPU_COUNT ? 0 : receive_matrix_idx + 1;
		}
	}

	QUEUE_EMPTY = false;

}

Matrix *ClusterNet::distribute_rows_hdf5_file(std::string path)
{

	int rows = 0;
	int cols = 0;
	Matrix *out;
	if(MYRANK == 0)
	{
		Matrix *cpu = read_hdf5(path.c_str());
		Matrix *gpu = to_gpu(cpu);

		std::vector<Matrix*> splits;

		int split_size = cpu->rows/MPI_SIZE;
		int offsize = cpu->rows - (split_size*MPI_SIZE);

		for(int i = 0; i < MPI_SIZE; i++)
		{
			int start = (split_size*i) + offsize;
			int end = split_size*(i+1);


			if(i==0){ start -= offsize; end +=offsize; }

			Matrix *gpuslice = slice_rows(gpu,start,end-1);
			Matrix *cpuslice = to_host(gpuslice);

			splits.push_back(cpuslice);

			cudaFree(gpuslice->data);
			free(gpuslice);
		}


		for(int i=1; i < MPI_SIZE; i++)
		{
			MPI_Send(&splits[i]->rows,1,MPI_INT,i,999,MPI_COMM_WORLD);
			MPI_Send(&splits[i]->cols,1,MPI_INT,i,999,MPI_COMM_WORLD);
			MPI_Send(splits[i]->data,splits[i]->size,MPI_FLOAT,i,999,MPI_COMM_WORLD);
			free(splits[i]->data);
			free(splits[i]);
		}

		out = splits[0];
	}
	else
	{
		MPI_Recv(&rows,1,MPI_INT,0,999,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Recv(&cols,1,MPI_INT,0,999,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		out = empty_cpu(rows,cols);
		MPI_Recv(out->data,out->size,MPI_FLOAT,0,999,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	}

	return out;
}

Matrix *ClusterNet::distribute_file(std::string path)
{
	int rows = 0;
	int cols = 0;
	Matrix *out;
	if(MYRANK == 0)
	{
		Matrix *cpu = read_hdf5(path.c_str());


		for(int i=1; i < MPI_SIZE; i++)
		{
			MPI_Send(&cpu->rows,1,MPI_INT,i,999,MPI_COMM_WORLD);
			MPI_Send(&cpu->cols,1,MPI_INT,i,999,MPI_COMM_WORLD);
			MPI_Send(cpu->data,cpu->size,MPI_FLOAT,i,999,MPI_COMM_WORLD);
		}

		out = cpu;
	}
	else
	{
		MPI_Recv(&rows,1,MPI_INT,0,999,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPI_Recv(&cols,1,MPI_INT,0,999,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		out = empty_cpu(rows,cols);
		MPI_Recv(out->data,out->size,MPI_FLOAT,0,999,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	}

	return out;
}


