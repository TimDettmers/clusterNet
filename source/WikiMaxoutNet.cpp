#include <WikiMaxoutNet.h>
#include <assert.h>
using std::cout;
using std::endl;


WikiMaxoutNet::WikiMaxoutNet(ClusterNet gpu)
{
	int nWordVectorDim = 50;
	int nWindowSize = 11;
	_nMaxoutSize = 8;
	_learningRate = 0.01;
	_nTrainErrorPeriodicity = 1000;
	_nTrainErrorLength = 100;
	_gpu = gpu;
	_nCurrentDataSet = _gpu.MYRANK;
	_X = gpu.rand(1,1);
	_nNextBatchNumber = 0;
	_nBatchSize = 128;
	cudaStreamCreate(&_streamNextBatch);

	_Vocab = gpu.uniformSqrtWeight(nWordVectorDim,300002);
	//scalarMul(_Vocab,0.0002,_Vocab);
	_W1 = gpu.uniformSqrtWeight(nWordVectorDim*nWindowSize,1024);
	_B1 = zeros(1,1024);
	_W2 = gpu.uniformSqrtWeight(128,1);
	_B2 = zeros(1,1);

	_grad1_X = zeros(nWordVectorDim*nWindowSize,1024);
	_grad1_Y = zeros(nWordVectorDim*nWindowSize,1024);
	_gradB1_X = zeros(1,1024);
	_gradB1_Y = zeros(1,1024);
	_grad2_X = zeros(128,1);
	_grad2_Y = zeros(128,1);
	_gradB2_X = zeros(1,1);
	_gradB2_Y = zeros(1,1);
	_Vocab_grad = zeros(nWordVectorDim,300002);
	_Vocab_grad_idx = zeros(1,300002);
	_grad0_X = zeros(128,nWordVectorDim*nWindowSize);
	_grad0_Y = zeros(128,nWordVectorDim*nWindowSize);


	d0 = zeros(_nBatchSize,nWordVectorDim*nWindowSize);
	z1 = zeros(_nBatchSize, 1024);
	a1_Y = zeros(_nBatchSize, 128);
	a1_idx_Y = zeros(_nBatchSize, 128);
	a1_X = zeros(_nBatchSize, 128);
	a1_idx_X = zeros(_nBatchSize, 128);
	d1 = zeros(_nBatchSize, 128);
	z2_X = zeros(_nBatchSize, 1);
	z2_Y = zeros(_nBatchSize, 1);


	_batchX = zeros(_nBatchSize, nWordVectorDim*nWindowSize);
	_batchY = zeros(_nBatchSize, nWordVectorDim*nWindowSize);
	_currentBatchIdx_X = zeros(_nBatchSize,nWindowSize);
	_currentBatchIdx_Y = zeros(_nBatchSize,nWindowSize);
	_nextBatchIdx = zeros(_nBatchSize,nWindowSize);

	_dSumError = 0.0;

	loadNextDataSet();
	loadNextDataSet();
	allocateNextBatch();
}
void WikiMaxoutNet::run()
{
	allocateNextBatch();

	int i = 0;
	while(true)
	{
		if(i > 0 && i % _nTrainErrorPeriodicity == 0)
		{
			cout << "BatchNo: " << i << endl;
			calculateError();
			i+=_nTrainErrorLength;
		}
		else
		{
			feedforward();
		}

		allocateNextBatch();
		i++;
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
	_batches = _X->rows/ _nBatchSize;
	_nNextBatchNumber = 0;
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

	if (_nNextBatchNumber > _batches)
		loadNextDataSet();
}

void WikiMaxoutNet::feedforward()
{
	_gpu.dot(_batchX,_W1,z1);
	addMatrixVector(z1,_B1,z1);

	maxout(z1, a1_X, a1_idx_X, _nMaxoutSize);
	_gpu.dot(a1_X,_W2,z2_X);
	addMatrixVector(z2_X,_B2,z2_X);

	_gpu.dot(_batchY,_W1,z1);
	addMatrixVector(z1,_B1,z1);
	maxout(z1, a1_Y, a1_idx_Y, _nMaxoutSize);
	_gpu.dot(a1_Y,_W2,z2_Y);
	addMatrixVector(z2_Y,_B2,z2_Y);


	Matrix *out = pairwise_ranking(z2_X,z2_Y);
	Matrix *pairwise_grad = pairwise_ranking_derivative(z2_X,z2_Y);

	Matrix *e1_Y = empty(_nBatchSize,1);
	mul(out, pairwise_grad, e1_Y);
	Matrix *aB = ones(1,_nBatchSize);
	_gpu.dot(aB,e1_Y,_gradB2_Y);

	_gpu.Tdot(a1_Y,e1_Y,_grad2_Y);
	Matrix *e2_partial_Y = _gpu.dotT(e1_Y, _W2);
	Matrix *e2_Y = empty(_nBatchSize,e2_partial_Y->cols*_nMaxoutSize);
    expand_to_maxout_grad(e2_partial_Y, a1_idx_Y,e2_Y);
    _gpu.Tdot(_batchY,e2_Y,_grad1_Y);
    _gpu.dot(aB,e2_Y,_gradB1_Y);
    _gpu.dotT(e2_Y,_W1,_grad0_Y);


	Matrix *e1_X = empty(_nBatchSize,1);
	scalarMul(pairwise_grad,-1.0f,pairwise_grad);
	mul(out, pairwise_grad, e1_X);

	_gpu.dot(aB,e1_X,_gradB2_X);

	_gpu.Tdot(a1_X,e1_X,_grad2_X);
	Matrix *e2_partial_X = _gpu.dotT(e1_X, _W2);
	Matrix *e2_X = empty(_nBatchSize,e2_partial_X->cols*_nMaxoutSize);
    expand_to_maxout_grad(e2_partial_X, a1_idx_X,e2_X);
    _gpu.Tdot(_batchX,e2_X,_grad1_X);
	_gpu.dot(aB,e2_X,_gradB1_X);
    _gpu.dotT(e2_X,_W1,_grad0_X);

	scalarMul(_grad2_Y,_learningRate/(float)_batchY->rows,_grad2_Y);
	sub(_W2,_grad2_Y,_W2);
	scalarMul(_gradB2_Y,_learningRate/(float)_batchX->rows,_gradB2_Y);
	sub(_B2,_gradB2_Y,_B2);
	scalarMul(_grad1_Y,0.1*_learningRate/(float)_batchY->rows,_grad1_Y);
	sub(_W1,_grad1_Y,_W1);
	scalarMul(_gradB1_Y,_learningRate/(float)_batchX->rows,_gradB1_Y);
	sub(_B1,_gradB1_Y,_B1);
	update_vocab_with_gradient(_grad0_Y,_currentBatchIdx_Y, _Vocab,_learningRate*0.01f);


	scalarMul(_grad2_X,_learningRate/(float)_batchX->rows,_grad2_X);
	sub(_W2,_grad2_X,_W2);
	scalarMul(_gradB2_X,_learningRate/(float)_batchX->rows,_gradB2_X);
	sub(_B2,_gradB2_X,_B2);
	scalarMul(_grad1_X,0.1*_learningRate/(float)_batchX->rows,_grad1_X);
	sub(_W1,_grad1_X,_W1);
	scalarMul(_gradB1_X,_learningRate/(float)_batchX->rows,_gradB1_X);
	sub(_B1,_gradB1_X,_B1);
	update_vocab_with_gradient(_grad0_X,_currentBatchIdx_X, _Vocab,_learningRate*0.01f);




	cudaFree(out->data);
	cudaFree(pairwise_grad->data);
	cudaFree(e1_Y->data);
	cudaFree(e2_partial_Y->data);
	cudaFree(e2_Y->data);
	cudaFree(e1_X->data);
	cudaFree(e2_partial_X->data);
	cudaFree(e2_X->data);
	cudaFree(aB->data);



}

void WikiMaxoutNet::calculateError()
{
	for(int i = 0; i < _nTrainErrorLength; i++)
	{
		_gpu.dot(_batchX,_W1,z1);
		addMatrixVector(z1,_B1,z1);
		maxout(z1, a1_X, a1_idx_X, _nMaxoutSize);
		_gpu.dot(a1_X,_W2,z2_X);
		addMatrixVector(z2_X,_B2,z2_X);

		_gpu.dot(_batchY,_W1,z1);
		addMatrixVector(z1,_B1,z1);
		maxout(z1, a1_Y, a1_idx_Y, _nMaxoutSize);
		_gpu.dot(a1_Y,_W2,z2_Y);
		addMatrixVector(z2_Y,_B2,z2_Y);

		Matrix *out = pairwise_ranking(z2_X,z2_Y);
		_dSumError += (double)sum(out);

		cudaFree(out->data);

		allocateNextBatch();
	}
	//size_t free, total;
	//cudaMemGetInfo(&free, &total);
	//cout << free << endl;

	cout << "Cross validation error: " << _dSumError/(double)(_nBatchSize*_nTrainErrorLength) << endl;

	_dSumError = 0.0;
}





