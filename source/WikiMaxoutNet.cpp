#include <WikiMaxoutNet.h>
#include <assert.h>
#include <exception>
using std::cout;
using std::endl;


WikiMaxoutNet::WikiMaxoutNet(ClusterNet gpu)
{


	int nWordVectorDim = 50;
	int nWindowSize = 11;
	_layers.push_back(1024);
	_nMaxoutSize = 8;
	_learningRate = 0.003;
	_nTrainErrorPeriodicity = 1000;
	_nTrainErrorLength = 100;
	MOMENTUM = 0.5;
	_gpu = gpu;
	_nCurrentDataSet = _gpu.MYRANK;
	_X = gpu.rand(1,1);
	_nNextBatchNumber = 0;
	_nBatchSize = 128;
	cudaStreamCreate(&_streamNextBatch);

	W.push_back(gpu.uniformSqrtWeight(nWordVectorDim*nWindowSize,_layers[0]));
	W.push_back(gpu.uniformSqrtWeight(_layers[0]/_nMaxoutSize, 1));
	B.push_back(zeros(1,_layers[0]));
	B.push_back(zeros(1,1));
	M.push_back(zeros(nWordVectorDim*nWindowSize,_layers[0]));
	M.push_back(zeros(_layers[0]/_nMaxoutSize, 1));
	BM.push_back(zeros(1,_layers[0]));
	BM.push_back(zeros(1,1));

	//scalarMul(_Vocab,20,_Vocab);


	for(int i = W.size()-1; i >= 0; i--)
	{
		GRAD.push_back(zeros(W[i]->rows, W[i]->cols));
		GRAD.push_back(zeros(W[i]->rows, W[i]->cols));
		MSGRAD.push_back(zeros(W[i]->rows, W[i]->cols));
		MSGRAD.push_back(zeros(W[i]->rows, W[i]->cols));
		BGRAD.push_back(zeros(B[i]->rows, B[i]->cols));
		BGRAD.push_back(zeros(B[i]->rows, B[i]->cols));
		MSBGRAD.push_back(zeros(B[i]->rows, B[i]->cols));
		MSBGRAD.push_back(zeros(B[i]->rows, B[i]->cols));
	}
	GRAD.push_back(zeros(_nBatchSize,nWordVectorDim*nWindowSize));
	GRAD.push_back(zeros(_nBatchSize,nWordVectorDim*nWindowSize));
	MSGRAD.push_back(zeros(_nBatchSize,nWordVectorDim*nWindowSize));
	MSGRAD.push_back(zeros(_nBatchSize,nWordVectorDim*nWindowSize));

	_Vocab = gpu.uniformSqrtWeight(nWordVectorDim,300002);
	_Vocab_grad = zeros(nWordVectorDim,300002);
	_MSVocab_grad = zeros(nWordVectorDim,300002);
	_MVocab = zeros(nWordVectorDim,300002);
	_Vocab_grad_idx = zeros(1,300002);

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
	allocateNextBatch();
}
void WikiMaxoutNet::run()
{
	allocateNextBatch();

	size_t freemem, total;
	cudaMemGetInfo(&freemem,&total);
	cout << freemem << endl;

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

		cudaMemcpyAsync(_nextBatchIdx->data,&_X->data[_nBatchSize*11*_nNextBatchNumber],
					_nBatchSize*11*sizeof(float),
					cudaMemcpyHostToDevice,_streamNextBatch);


	_nNextBatchNumber+=1;

	if (_nNextBatchNumber > _batches)
		loadNextDataSet();
}

void WikiMaxoutNet::feedforward()
{

	//nesterov
	for(int i = 0;i < M.size(); i++)
	{
		scalarMul(M[i],MOMENTUM,M[i]);
		add(W[i],M[i],W[i]);
	}

	for(int i = 0;i < B.size(); i++)
	{
		scalarMul(BM[i],MOMENTUM,BM[i]);
		add(B[i],BM[i],B[i]);
	}

	scalarMul(_MVocab, MOMENTUM, _MVocab);
	add(_Vocab,_MVocab,_Vocab);


	//_gpu.dropout(_batchX,d0,0.2);
	_gpu.dot(_batchX,W[0],z1);
	addMatrixVector(z1,B[0],z1);

	maxout(z1, a1_X, a1_idx_X, _nMaxoutSize);
	//_gpu.dropout(a1_X,d1,0.5);
	_gpu.dot(a1_X,W[1],z2_X);
	addMatrixVector(z2_X,B[1],z2_X);

	//_gpu.dropout(_batchY,d0,0.2);
	_gpu.dot(_batchY,W[0],z1);
	addMatrixVector(z1,B[0],z1);
	maxout(z1, a1_Y, a1_idx_Y, _nMaxoutSize);
	//_gpu.dropout(a1_Y,d1,0.5);
	_gpu.dot(a1_Y,W[1],z2_Y);
	addMatrixVector(z2_Y,B[1],z2_Y);


	Matrix *out = pairwise_ranking(z2_X,z2_Y);
	Matrix *pairwise_grad = pairwise_ranking_derivative(z2_X,z2_Y);

	Matrix *e1_Y = empty(_nBatchSize,1);
	mul(out, pairwise_grad, e1_Y);
	Matrix *aB = ones(1,_nBatchSize);

	_gpu.dot(aB,e1_Y,BGRAD[0]);
	_gpu.Tdot(a1_Y,e1_Y,GRAD[0]);

	Matrix *e2_partial_Y = _gpu.dotT(e1_Y, W[1]);
	Matrix *e2_Y = empty(_nBatchSize,e2_partial_Y->cols*_nMaxoutSize);
    expand_to_maxout_grad(e2_partial_Y, a1_idx_Y,e2_Y);

    _gpu.Tdot(_batchY,e2_Y,GRAD[2]);
    _gpu.dot(aB,e2_Y,BGRAD[2]);
    _gpu.dotT(e2_Y,W[0],GRAD[4]);

	Matrix *e1_X = empty(_nBatchSize,1);
	scalarMul(pairwise_grad,-1.0f,pairwise_grad);
	mul(out, pairwise_grad, e1_X);

	_gpu.dot(aB,e1_X,BGRAD[1]);
	_gpu.Tdot(a1_X,e1_X,GRAD[1]);

	Matrix *e2_partial_X = _gpu.dotT(e1_X, W[1]);
	Matrix *e2_X = empty(_nBatchSize,e2_partial_X->cols*_nMaxoutSize);
    expand_to_maxout_grad(e2_partial_X, a1_idx_X,e2_X);

    _gpu.Tdot(_batchX,e2_X,GRAD[3]);
	_gpu.dot(aB,e2_X,BGRAD[3]);
    _gpu.dotT(e2_X,W[0],GRAD[5]);

    float multiplier = _learningRate/(float)_nBatchSize;



    RMSprop_with_nesterov_weight_update(MSGRAD[0],GRAD[0],W[1],M[1],0.9f,_learningRate/(float)GRAD[0]->rows,_nBatchSize);
    RMSprop_with_nesterov_weight_update(MSGRAD[1],GRAD[1],W[1],M[1],0.9f,_learningRate/(float)GRAD[1]->rows,_nBatchSize);
    RMSprop_with_nesterov_weight_update(MSGRAD[2],GRAD[2],W[0],M[0],0.9f,_learningRate/(float)GRAD[2]->rows,_nBatchSize);
    RMSprop_with_nesterov_weight_update(MSGRAD[3],GRAD[3],W[0],M[0],0.9f,_learningRate/(float)GRAD[3]->rows,_nBatchSize);

    RMSprop_with_nesterov_weight_update(MSBGRAD[0],BGRAD[0],B[1],BM[1],0.9f,_learningRate,_nBatchSize);
    RMSprop_with_nesterov_weight_update(MSBGRAD[1],BGRAD[1],B[1],BM[1],0.9f,_learningRate,_nBatchSize);
    RMSprop_with_nesterov_weight_update(MSBGRAD[2],BGRAD[2],B[0],BM[0],0.9f,_learningRate,_nBatchSize);
    RMSprop_with_nesterov_weight_update(MSBGRAD[3],BGRAD[3],B[0],BM[0],0.9f,_learningRate,_nBatchSize);

	//update_vocab_with_gradient(GRAD[4],_currentBatchIdx_Y, _Vocab,_learningRate/(float)_nBatchSize);
	//update_vocab_with_gradient(GRAD[5],_currentBatchIdx_X, _Vocab,_learningRate/(float)_nBatchSize);

	fill_matrix(_Vocab_grad,0.0f);
	update_vocab_with_gradient(GRAD[5],GRAD[4],_currentBatchIdx_X,_currentBatchIdx_Y,_Vocab,_Vocab_grad,_Vocab_grad_idx,_learningRate/(float)_nBatchSize);
	RMSprop_with_nesterov_weight_update(_MSVocab_grad,_Vocab_grad,_Vocab,_MVocab,0.9f,_learningRate/(float)_nBatchSize,1);



	cudaFree(out->data);
	cudaFree(pairwise_grad->data);
	cudaFree(e1_Y->data);
	cudaFree(e2_partial_Y->data);
	cudaFree(e2_Y->data);
	cudaFree(e1_X->data);
	cudaFree(e2_partial_X->data);
	cudaFree(e2_X->data);
	cudaFree(aB->data);

	/*
	free(out);
	free(pairwise_grad);
	free(e1_Y);
	free(e2_partial_Y);
	free(e2_Y);
	free(e1_X);
	free(e2_partial_X);
	free(e2_X);
	free(aB);
	*/



}

void WikiMaxoutNet::calculateError()
{

	for(int i = 0; i < _nTrainErrorLength; i++)
	{

		//scalarMul(_batchX,0.8,d0);
		_gpu.dot(_batchX,W[0],z1);
		addMatrixVector(z1,B[0],z1);
		maxout(z1, a1_X, a1_idx_X, _nMaxoutSize);
		//scalarMul(a1_X,0.5,d1);
		_gpu.dot(a1_X,W[1],z2_X);
		addMatrixVector(z2_X,B[1],z2_X);

		//scalarMul(_batchY,0.8,d0);
		_gpu.dot(_batchY,W[0],z1);
		addMatrixVector(z1,B[0],z1);
		maxout(z1, a1_Y, a1_idx_Y, _nMaxoutSize);
		//scalarMul(a1_Y,0.5,d1);
		_gpu.dot(a1_Y,W[1],z2_Y);
		addMatrixVector(z2_Y,B[1],z2_Y);

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





