#include <WikiMaxoutNet.h>
using std::cout;
using std::endl;


WikiMaxoutNet::WikiMaxoutNet(ClusterNet gpu)
{
	int nWordVectorDim = 50;
	int nWindowSize = 11;
	_nTrainErrorPeriodicity = 500;
	_nTrainErrorLength = 100;
	_gpu = gpu;
	_nCurrentDataSet = _gpu.MYRANK;
	_X = gpu.rand(1,1);
	_nNextBatchNumber = 0;
	_nBatchSize = 128;
	cudaStreamCreate(&_streamNextBatch);

	_Vocab = gpu.uniformSqrtWeight(nWordVectorDim,300000);
	_W1 = gpu.uniformSqrtWeight(nWordVectorDim*nWindowSize,1024);
	_B1 = zeros(1,1024);
	_W2 = gpu.uniformSqrtWeight(128,1);
	_B2 = zeros(1,1);

	_grad1_X = zeros(nWordVectorDim*nWindowSize,1024);
	_grad1_Y = zeros(nWordVectorDim*nWindowSize,1024);
	_gradB1 = zeros(1,1024);
	_grad2_X = zeros(128,1);
	_grad2_Y = zeros(128,1);
	_gradB2 = zeros(1,1);
	_grad3_X = zeros(128,nWordVectorDim*nWindowSize);
	_grad3_Y = zeros(128,nWordVectorDim*nWindowSize);


	d0 = empty(_nBatchSize,nWordVectorDim*nWindowSize);
	z1 = empty(_nBatchSize, 1024);
	a1_Y = empty(_nBatchSize, 128);
	a1_idx_Y = empty(_nBatchSize, 128);
	a1_X = empty(_nBatchSize, 128);
	a1_idx_X = empty(_nBatchSize, 128);
	d1 = empty(_nBatchSize, 128);
	z2_X = empty(_nBatchSize, 1);
	z2_Y = empty(_nBatchSize, 1);


	_batchX = empty(_nBatchSize, nWordVectorDim*nWindowSize);
	_batchY = empty(_nBatchSize, nWordVectorDim*nWindowSize);
	_currentBatchIdx_X = empty(_nBatchSize,nWindowSize);
	_currentBatchIdx_Y = empty(_nBatchSize,nWindowSize);
	_nextBatchIdx = empty(_nBatchSize,nWindowSize);

	_fSumError = 0.0f;

	loadNextDataSet();
	allocateNextBatch();
}
void WikiMaxoutNet::run()
{
	allocateNextBatch();
	int batches = _X->rows/ _nBatchSize;

	for(int i = 0; i < batches; i++)
	{
		if(i > 0 && i % _nTrainErrorPeriodicity == 0)
		{
			calculateError();
			i+=10;
		}
		else
		{
			feedforward();
		}

		//cout << i << endl;

		allocateNextBatch();
	}
}

void WikiMaxoutNet::loadNextDataSet()
{
	std::string path = "/home/tim/data/wiki/extracted2/AA/data/wiki_";
	std::string number = "";
	std::string ending = ".p.hdf5";

	if(_nCurrentDataSet < 10)
		number += "0";

	number+= NumberToString(_nCurrentDataSet);

	cudaFree(_X);
	_X = read_hdf5((path + number + ending).c_str());
	_nCurrentDataSet += _gpu.MPI_SIZE;
}

void WikiMaxoutNet::allocateNextBatch()
{
	if(_nNextBatchNumber > 0)
	{
		cudaStreamSynchronize(_streamNextBatch);
		to_col_major(_nextBatchIdx, _currentBatchIdx_X);
		_gpu.construct_vocab_matrix(_currentBatchIdx_X, _currentBatchIdx_Y, _batchX, _batchY, _Vocab);
	}

	cudaMemcpyAsync(_nextBatchIdx->data,&_X->data[_nBatchSize*11*sizeof(float)*_nNextBatchNumber],
					_nBatchSize*11*sizeof(float),
					cudaMemcpyHostToDevice,_streamNextBatch);

	_nNextBatchNumber+=1;
}

void WikiMaxoutNet::feedforward()
{
	_gpu.dropout(_batchX, d0, 0.2f);
	_gpu.dot(d0,_W1,z1);
	addMatrixVector(z1,_B1,z1);

	maxout(z1, a1_X, a1_idx_X, 8);
	_gpu.dropout(a1_Y, d1, 0.5f);
	_gpu.dot(d1,_W2,z2_X);
	addMatrixVector(z2_X,_B2,z2_X);

	_gpu.dropout(_batchY, d0, 0.2f);
	_gpu.dot(d0,_W1,z1);
	addMatrixVector(z1,_B1,z1);
	maxout(z1, a1_Y, a1_idx_Y, 8);
	_gpu.dropout(a1_Y, d1, 0.5f);
	_gpu.dot(d1,_W2,z2_X);
	addMatrixVector(z2_X,_B2,z2_X);



	Matrix *out = pairwise_ranking(z2_X,z2_Y);
	Matrix *pairwise_grad = pairwise_ranking_derivative(z2_X,z2_Y);
	scalarMul(pairwise_grad,-1.0f,pairwise_grad);

	Matrix *e1_Y = empty(_nBatchSize,1);
	mul(out, pairwise_grad, e1_Y);

	_gpu.Tdot(a1_Y,e1_Y,_grad2_Y);
	Matrix *e2_partial_Y = _gpu.dotT(e1_Y, _W2);
	Matrix *e2_Y = empty(_nBatchSize,e2_partial_Y->cols*8);
    expand_to_maxout_grad(e2_partial_Y, a1_idx_X,e2_Y);
    _gpu.Tdot(_batchX,e2_Y,_grad1_Y);

    //Matrix *e3 = _gpu.dotT(e2_Y,_W1);

    //update_vocab_gradient(e3,_Vocab, _currentBatchIdx_X);


	Matrix *e1_X = empty(_nBatchSize,1);
	mul(out, pairwise_grad, e1_X);

	_gpu.Tdot(a1_X,e1_X,_grad2_X);
	Matrix *e2_partial_X = _gpu.dotT(e1_X, _W2);
	Matrix *e2_X = empty(_nBatchSize,e2_partial_X->cols*8);
    expand_to_maxout_grad(e2_partial_X, a1_idx_X,e2_X);
    _gpu.Tdot(_batchX,e2_X,_grad1_X);


	scalarMul(_grad2_Y,0.001/(float)_batchY->rows,_grad2_Y);
	sub(_W2,_grad2_Y,_W2);
	scalarMul(_grad1_Y,0.001/(float)_batchY->rows,_grad1_Y);
	sub(_W1,_grad1_Y,_W1);

	scalarMul(_grad2_X,0.001/(float)_batchX->rows,_grad2_X);
	sub(_W2,_grad2_X,_W2);
	scalarMul(_grad1_X,0.001/(float)_batchX->rows,_grad1_X);
	sub(_W1,_grad1_X,_W1);


	cudaFree(out->data);
	cudaFree(pairwise_grad->data);
	cudaFree(e1_Y->data);
	cudaFree(e2_partial_Y->data);
	cudaFree(e2_Y->data);
	cudaFree(e1_X->data);
	cudaFree(e2_partial_X->data);
	cudaFree(e2_X->data);



}

void WikiMaxoutNet::calculateError()
{
	for(int i = 0; i < _nTrainErrorLength; i++)
	{
		scalarMul(_batchX, 0.8f,d0);
		_gpu.dot(d0,_W1,z1);
		addMatrixVector(z1,_B1,z1);
		maxout(z1, a1_X, a1_idx_X, 8);
		scalarMul(a1_X, 0.5f,d1);

		_gpu.dot(d1,_W2,z2_X);
		addMatrixVector(z2_X,_B2,z2_X);

		scalarMul(_batchY, 0.8f,d0);
		_gpu.dot(d0,_W1,z1);
		addMatrixVector(z1,_B1,z1);
		maxout(z1, a1_Y, a1_idx_Y, 8);
		scalarMul(a1_Y, 0.5f,d1);
		_gpu.dot(d1,_W2,z2_X);
		addMatrixVector(z2_X,_B2,z2_X);

		Matrix *out = pairwise_ranking(z2_X,z2_Y);
		_fSumError += sum(out);

		cudaFree(out->data);


		allocateNextBatch();
	}

	cout << _fSumError/(float)_nBatchSize << endl;

	_fSumError = 0.0f;
}





