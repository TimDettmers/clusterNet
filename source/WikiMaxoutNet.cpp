#include <WikiMaxoutNet.h>
using std::cout;
using std::endl;

WikiMaxoutNet::WikiMaxoutNet(ClusterNet gpus)
{


	int vocabSize = 100002;
	int nWordVectorDim = 64;
	int nWindowSize = 11;
	_layers.push_back(512);
	_nMaxoutSize = 32;
	_learningRate = 0.003;
	_nCVErrorPeriodicity = 5000;
	_nCVErrorLength = 2000;
	MOMENTUM = 0.5;
	gpu = gpus;
	_nCurrentDataSet = gpu.MYRANK;
	_X = gpu.rand(1,1);
	int cv_set_number = 64-gpu.MYRANK;
	_CV_X = read_hdf5(("/home/tim/data/wiki/extracted2/AA/data100000/wiki_" + NumberToString(cv_set_number) + ".p").c_str());
	_nNextBatchNumber = 0;
	_nNextBatchNumber_CV = 0;
	_nBatchSize = 128;
	_RMS_multiplier = 0.9f;
	cudaStreamCreate(&_streamNextBatch);

	useRMSProp = true;
	useMaxout = true;

	cout << "_nMaxoutSize: " << _nMaxoutSize << endl;
	cout << "_layers: " << _layers[0] << endl;
	cout << "nWordVectorDim: " << nWordVectorDim << endl;
	cout << "_nBatchSize: " << _nBatchSize << endl;
	cout << "_learningRate: " << _learningRate << endl;
    cout << "Use RMSProp: "  << useRMSProp << endl;

	W.push_back(gpu.uniformSqrtWeight(nWordVectorDim*nWindowSize,_layers[0]));
	W.push_back(gpu.uniformSqrtWeight(_layers[0]/_nMaxoutSize, 1));
	B.push_back(zeros(1,_layers[0]));
	B.push_back(zeros(1,1));
	M.push_back(zeros(nWordVectorDim*nWindowSize,_layers[0]));
	M.push_back(zeros(_layers[0]/_nMaxoutSize, 1));
	M_B.push_back(zeros(1,_layers[0]));
	M_B.push_back(zeros(1,1));


	cout << gpu.MPI_SIZE << " MPI SIZE" << endl;
	cout << gpu.MYRANK << " MYRANK " << endl;
	for(int i = W.size()-1; i >= 0; i--)
	{
		Matrix **gradX = (Matrix**)malloc(sizeof(Matrix*)*gpu.MPI_SIZE);
		arrGRAD.push_back(gradX);
		Matrix **gradY = (Matrix**)malloc(sizeof(Matrix*)*gpu.MPI_SIZE);
		arrGRAD.push_back(gradY);
		Matrix **gradX_B = (Matrix**)malloc(sizeof(Matrix*)*gpu.MPI_SIZE);
		arrGRAD_B.push_back(gradX_B);
		Matrix **gradY_B = (Matrix**)malloc(sizeof(Matrix*)*gpu.MPI_SIZE);
		arrGRAD_B.push_back(gradY_B);
	}
	Matrix **gradX = (Matrix**)malloc(sizeof(Matrix*)*gpu.MPI_SIZE);
	arrGRAD.push_back(gradX);
	Matrix **gradY = (Matrix**)malloc(sizeof(Matrix*)*gpu.MPI_SIZE);
	arrGRAD.push_back(gradY);

	cout << arrGRAD.size() << " size" << endl;


	for(int i = W.size()-1; i >= 0; i--)
	{
		MSGRAD.push_back(zeros(W[i]->rows, W[i]->cols));
		MSGRAD.push_back(zeros(W[i]->rows, W[i]->cols));
		MSBGRAD.push_back(zeros(B[i]->rows, B[i]->cols));
		MSBGRAD.push_back(zeros(B[i]->rows, B[i]->cols));
	}

	for(int j =0; j < gpu.MPI_SIZE; j++)
	{
		int idx = 0;
		for(int i = W.size()-1; i >= 0; i--)
		{
			arrGRAD[idx][j] = zeros(W[i]->rows, W[i]->cols);
			arrGRAD_B[idx][j] = zeros(B[i]->rows, B[i]->cols);
			idx++;
			arrGRAD[idx][j] = (zeros(W[i]->rows, W[i]->cols));
			arrGRAD_B[idx][j] = zeros(B[i]->rows, B[i]->cols);
			idx++;
		}
	}
	arrGRAD[4][gpu.MYRANK] = zeros(_nBatchSize,nWordVectorDim*nWindowSize);
	arrGRAD[5][gpu.MYRANK] = zeros(_nBatchSize,nWordVectorDim*nWindowSize);

	MSGRAD.push_back(zeros(_nBatchSize*gpu.MPI_SIZE,nWordVectorDim*nWindowSize));
	MSGRAD.push_back(zeros(_nBatchSize*gpu.MPI_SIZE,nWordVectorDim*nWindowSize));

	_Vocab = gpu.uniformSqrtWeight(nWordVectorDim,vocabSize);
	//_Vocab = gpu.sparseInitWeight(nWordVectorDim,vocabSize);
	//_Vocab = gpu.rand(nWordVectorDim,vocabSize);
	//scalarMul(_Vocab,0.01f,_Vocab);
	//scalarAdd(_Vocab,-0.5f,_Vocab);
	cout << sum(_Vocab) << endl;
	_Vocab_grad = zeros(nWordVectorDim,vocabSize);
	_MSVocab_grad = zeros(nWordVectorDim,vocabSize);
	_MSVocab_grad_Y = zeros(nWordVectorDim,vocabSize);
	M_VocabX = zeros(nWordVectorDim,vocabSize);
	M_VocabY = zeros(nWordVectorDim,vocabSize);
	_Vocab_grad_idx = zeros(1,vocabSize);

	d0 = zeros(_nBatchSize,nWordVectorDim*nWindowSize);
	z1 = zeros(_nBatchSize, _layers[0]);
	a1_Y = zeros(_nBatchSize, _layers[0]/_nMaxoutSize);
	a1_idx_Y = zeros(_nBatchSize, _layers[0]/_nMaxoutSize);
	a1_X = zeros(_nBatchSize, _layers[0]/_nMaxoutSize);
	a1_idx_X = zeros(_nBatchSize, _layers[0]/_nMaxoutSize);
	d1 = zeros(_nBatchSize, _layers[0]/_nMaxoutSize);
	z2_X = zeros(_nBatchSize, 1);
	z2_Y = zeros(_nBatchSize, 1);

	out = zeros(_nBatchSize,1);
	pairwise_grad = zeros(_nBatchSize,1);
	e1 = empty(_nBatchSize,1);
	aB = ones(1,_nBatchSize);
	e2_partial = zeros(_nBatchSize,W[1]->rows);
	e2 = empty(_nBatchSize,e2_partial->cols*_nMaxoutSize);


	_batchX = zeros(_nBatchSize, nWordVectorDim*nWindowSize);
	_batchY = zeros(_nBatchSize, nWordVectorDim*nWindowSize);
	_currentBatchIdx_X = zeros(_nBatchSize,nWindowSize);
	_currentBatchIdx_Y = zeros(_nBatchSize,nWindowSize);
	_nextBatchIdx = zeros(_nBatchSize,nWindowSize);

	_dSumError = 0.0;

	loadNextDataSet();
}
void WikiMaxoutNet::run()
{
	allocateNextBatch(false);

	size_t freemem, total;
	cudaMemGetInfo(&freemem,&total);
	cout << freemem << endl;


	srand( time(NULL) );
	start = clock();

	int i = 0;
	while(true)
	{
		if(i > 0 && i % _nCVErrorPeriodicity == 0)
		{
			if( i > 0 && i % 100000 == 0)
			{
				cout << "Saving vocabulary matrix to disk..." << endl;
				Matrix *host = to_host(_Vocab);
				write_hdf5("/home/tim/data/wiki/vocab.hdf5",host);
				free(host->data);
				free(host);
			}

			cout << "BatchNo: " << i << endl;
			cout << "Cross validation error: " <<  calculateError() << endl;
			i++;

			//MOMENTUM+= 0.01;
			//if( MOMENTUM > 0.95)
				//MOMENTUM = 0.95;

			_RMS_multiplier-= 0.01;
			if( _RMS_multiplier < 0.25)
				_RMS_multiplier = 0.25;

			stop = clock();

			double time_interval_seconds = (double((stop - start)) / CLOCKS_PER_SEC) ;
			cout << "Approximate time left in hours: " << (1.0f/(((i*_nBatchSize)/(float)_X->rows)/63.0))*time_interval_seconds/(float)3600.0 << endl;

		}
		else
		{
			nesterov();
			feedforward();
			backprop();
			weightUpdates();
		}

		allocateNextBatch(false);
		i++;
	}
}

void WikiMaxoutNet::loadNextDataSet()
{

	cout << "Loading next data set..." << endl;
	std::string path = "/home/tim/data/wiki/extracted2/AA/data100000/wiki_";
	//std::string path = "/home/tim/data/wiki/extracted2/AA/data/wiki_";
	std::string number = "";
	std::string ending = ".p";
	//std::string ending = ".p.hdf5";

	if(_nCurrentDataSet < 10)
		number += "0";

	number+= NumberToString(_nCurrentDataSet);

	cudaFree(_X);
	_X = read_hdf5((path + number + ending).c_str());
	_nCurrentDataSet += gpu.MPI_SIZE;
	_batches = _X->rows/ _nBatchSize;
	_nNextBatchNumber = 0;
}

void WikiMaxoutNet::allocateNextBatch(bool isCV)
{
	if(!isCV)
	{
		if(_nNextBatchNumber < 0)
			_nNextBatchNumber = 0;

		if (_nBatchSize*11*(_nNextBatchNumber+1) > _X->size)
			loadNextDataSet();

		if(_nNextBatchNumber > 0 || _nCurrentDataSet > gpu.MYRANK)
		{
			cudaStreamSynchronize(_streamNextBatch);
			to_col_major(_nextBatchIdx, _currentBatchIdx_X);
			gpu.construct_vocab_matrix(_currentBatchIdx_X, _currentBatchIdx_Y, _batchX, _batchY, _Vocab);
		}



			cudaMemcpyAsync(_nextBatchIdx->data,&_X->data[_nBatchSize*11*_nNextBatchNumber],
						_nBatchSize*11*sizeof(float),
						cudaMemcpyHostToDevice,_streamNextBatch);


		_nNextBatchNumber+=1;
	}
	else
	{
		if(_nNextBatchNumber_CV > 0)
		{
			cudaStreamSynchronize(_streamNextBatch);
			to_col_major(_nextBatchIdx, _currentBatchIdx_X);
			gpu.construct_vocab_matrix(_currentBatchIdx_X, _currentBatchIdx_Y, _batchX, _batchY, _Vocab);
		}

		cudaMemcpyAsync(_nextBatchIdx->data,&_CV_X->data[_nBatchSize*11*_nNextBatchNumber_CV],
					_nBatchSize*11*sizeof(float),
					cudaMemcpyHostToDevice,_streamNextBatch);


		_nNextBatchNumber_CV+=1;
	}


}

void WikiMaxoutNet::nesterov()
{
	//nesterov
	for(int i = 0;i < M.size(); i++)
	{
		scalarMul(M[i],MOMENTUM,M[i]);
		add(W[i],M[i],W[i]);
	}

	for(int i = 0;i < B.size(); i++)
	{
		scalarMul(M_B[i],MOMENTUM,M_B[i]);
		add(B[i],M_B[i],B[i]);
	}

	scalarMul(M_VocabX, MOMENTUM, M_VocabX);
	add(_Vocab,M_VocabX,_Vocab);
}

void WikiMaxoutNet::feedforward()
{

	if(useMaxout)
	{
		gpu.dot(_batchX,W[0],z1);
		addMatrixVector(z1,B[0],z1);
		maxout(z1, a1_X, a1_idx_X, _nMaxoutSize);
		gpu.dot(a1_X,W[1],z2_X);
		addMatrixVector(z2_X,B[1],z2_X);

		gpu.dot(_batchY,W[0],z1);
		addMatrixVector(z1,B[0],z1);
		maxout(z1, a1_Y, a1_idx_Y, _nMaxoutSize);
		gpu.dot(a1_Y,W[1],z2_Y);
		addMatrixVector(z2_Y,B[1],z2_Y);
	}
	else
	{
		gpu.dot(_batchX,W[0],z1);
		addMatrixVector(z1,B[0],z1);
		rectified_linear(z1,a1_X);
		gpu.dot(a1_X,W[1],z2_X);
		addMatrixVector(z2_X,B[1],z2_X);

		gpu.dot(_batchY,W[0],z1);
		addMatrixVector(z1,B[0],z1);
		rectified_linear(z1,a1_Y);
		gpu.dot(a1_Y,W[1],z2_Y);
		addMatrixVector(z2_Y,B[1],z2_Y);

	}
}

void WikiMaxoutNet::weightUpdates()
{
	float multiplier = _learningRate/(float)_nBatchSize;

	if(!useRMSProp)
	{
		scalarMul(arrGRAD[0][gpu.MYRANK],multiplier/(float)arrGRAD[0][gpu.MYRANK]->rows,arrGRAD[0][gpu.MYRANK]);
		scalarMul(arrGRAD[1][gpu.MYRANK],multiplier/(float)arrGRAD[1][gpu.MYRANK]->rows,arrGRAD[1][gpu.MYRANK]);
		scalarMul(arrGRAD[2][gpu.MYRANK],multiplier/(float)arrGRAD[1][gpu.MYRANK]->rows,arrGRAD[2][gpu.MYRANK]);
		scalarMul(arrGRAD[3][gpu.MYRANK],multiplier/(float)arrGRAD[1][gpu.MYRANK]->rows,arrGRAD[3][gpu.MYRANK]);
		scalarMul(arrGRAD_B[0][gpu.MYRANK],multiplier,arrGRAD_B[0][gpu.MYRANK]);
		scalarMul(arrGRAD_B[1][gpu.MYRANK],multiplier,arrGRAD_B[1][gpu.MYRANK]);
		scalarMul(arrGRAD_B[2][gpu.MYRANK],multiplier,arrGRAD_B[2][gpu.MYRANK]);
		scalarMul(arrGRAD_B[3][gpu.MYRANK],multiplier,arrGRAD_B[3][gpu.MYRANK]);

		sub(W[1],arrGRAD[0][gpu.MYRANK],W[1]);
		sub(W[1],arrGRAD[1][gpu.MYRANK],W[1]);
		sub(W[0],arrGRAD[2][gpu.MYRANK],W[0]);
		sub(W[0],arrGRAD[3][gpu.MYRANK],W[0]);
		sub(B[1],arrGRAD_B[0][gpu.MYRANK],B[1]);
		sub(B[1],arrGRAD_B[1][gpu.MYRANK],B[1]);
		sub(B[0],arrGRAD_B[2][gpu.MYRANK],B[0]);
		sub(B[0],arrGRAD_B[3][gpu.MYRANK],B[0]);

		update_vocab_with_gradient(arrGRAD[4][gpu.MYRANK],_currentBatchIdx_Y,_Vocab,multiplier);
		update_vocab_with_gradient(arrGRAD[5][gpu.MYRANK],_currentBatchIdx_Y,_Vocab,multiplier);
	}
	else
	{
		RMSprop_with_nesterov_weight_update(MSGRAD[0],arrGRAD[0][gpu.MYRANK],W[1],M[1],0.9f,_learningRate/(float)arrGRAD[0][gpu.MYRANK]->rows,_nBatchSize);
		RMSprop_with_nesterov_weight_update(MSGRAD[1],arrGRAD[1][gpu.MYRANK],W[1],M[1],0.9f,_learningRate/(float)arrGRAD[1][gpu.MYRANK]->rows,_nBatchSize);
		RMSprop_with_nesterov_weight_update(MSGRAD[2],arrGRAD[2][gpu.MYRANK],W[0],M[0],0.9f,_learningRate/(float)arrGRAD[2][gpu.MYRANK]->rows,_nBatchSize);
		RMSprop_with_nesterov_weight_update(MSGRAD[3],arrGRAD[3][gpu.MYRANK],W[0],M[0],0.9f,_learningRate/(float)arrGRAD[3][gpu.MYRANK]->rows,_nBatchSize);


		RMSprop_with_nesterov_weight_update(MSBGRAD[0],arrGRAD_B[0][gpu.MYRANK],B[1],M_B[1],0.9f,_learningRate,_nBatchSize);
		RMSprop_with_nesterov_weight_update(MSBGRAD[1],arrGRAD_B[1][gpu.MYRANK],B[1],M_B[1],0.9f,_learningRate,_nBatchSize);
		RMSprop_with_nesterov_weight_update(MSBGRAD[2],arrGRAD_B[2][gpu.MYRANK],B[0],M_B[0],0.9f,_learningRate,_nBatchSize);
		RMSprop_with_nesterov_weight_update(MSBGRAD[3],arrGRAD_B[3][gpu.MYRANK],B[0],M_B[0],0.9f,_learningRate,_nBatchSize);


		//update_vocab_with_gradient(GRAD[4],_currentBatchIdx_Y,_Vocab,0.01/(float)_nBatchSize);
		//update_vocab_with_gradient(GRAD[5],_currentBatchIdx_Y,_Vocab,0.01/(float)_nBatchSize);


		//fill_matrix(_Vocab_grad,0.0f);
		//expand_double_vocab_gradient(GRAD[5],GRAD[4],_currentBatchIdx_X,_currentBatchIdx_Y,_Vocab,_Vocab_grad,_Vocab_grad_idx,_learningRate/(float)_nBatchSize);
		//RMSprop_with_nesterov_weight_update(_MSVocab_grad,_Vocab_grad,_Vocab,_MVocab,_RMS_multiplier,_learningRate/(float)_nBatchSize,1);



		fill_matrix(_Vocab_grad,0.0f);
		expand_vocab_gradient(arrGRAD[5][gpu.MYRANK],_currentBatchIdx_X,_Vocab_grad);
		RMSprop_with_nesterov_weight_update(_MSVocab_grad,_Vocab_grad,_Vocab,M_VocabX,_RMS_multiplier,_learningRate/(float)_nBatchSize,1);

		fill_matrix(_Vocab_grad,0.0f);
		expand_vocab_gradient(arrGRAD[4][gpu.MYRANK],_currentBatchIdx_Y,_Vocab_grad);
		RMSprop_with_nesterov_weight_update(_MSVocab_grad_Y,_Vocab_grad,_Vocab,M_VocabY,_RMS_multiplier,_learningRate/(float)_nBatchSize,1);


	}

}

void WikiMaxoutNet::backprop()
{
	pairwise_ranking(z2_X,z2_Y, out);
	pairwise_ranking_derivative(z2_X,z2_Y, pairwise_grad);

	mul(out, pairwise_grad, e1);
	gpu.dotT(e1, W[1],e2_partial);

	gpu.dot(aB,e1,arrGRAD_B[0][gpu.MYRANK]);
	gpu.Tdot(a1_Y,e1,arrGRAD[0][gpu.MYRANK]);

    if(!useMaxout)
    {
    	rectified_linear_derivative(a1_Y,a1_Y);
		mul(e2_partial,a1_Y,e2);
    }
    else
    {
        expand_to_maxout_grad(e2_partial, a1_idx_Y,e2);
    }
    gpu.Tdot(_batchY,e2,arrGRAD[2][gpu.MYRANK]);
    gpu.dot(aB,e2,arrGRAD_B[2][gpu.MYRANK]);
    gpu.dotT(e2,W[0],arrGRAD[4][gpu.MYRANK]);


	scalarMul(pairwise_grad,-1.0f,pairwise_grad);
	mul(out, pairwise_grad, e1);

	gpu.dot(aB,e1,arrGRAD_B[1][gpu.MYRANK]);
	gpu.Tdot(a1_X,e1,arrGRAD[1][gpu.MYRANK]);
	gpu.dotT(e1, W[1],e2_partial);


    if(!useMaxout)
    {
    	rectified_linear_derivative(a1_X,a1_X);
 		mul(e2_partial,a1_X,e2);
    }
    else
    {
        expand_to_maxout_grad(e2_partial, a1_idx_X,e2);
    }
    gpu.Tdot(_batchX,e2,arrGRAD[3][gpu.MYRANK]);
	gpu.dot(aB,e2,arrGRAD_B[3][gpu.MYRANK]);
    gpu.dotT(e2,W[0],arrGRAD[5][gpu.MYRANK]);

}

double WikiMaxoutNet::calculateError()
{

	allocateNextBatch(true);
	for(int i = 0; i < _nCVErrorLength; i++)
	{

		feedforward();

		pairwise_ranking(z2_X,z2_Y, out);
		_dSumError += (double)sum(out);

		allocateNextBatch(true);
	}
	//size_t free, total;
	//cudaMemGetInfo(&free, &total);
	//cout << free << endl;

	double error = _dSumError/(double)(_nBatchSize*_nCVErrorLength);
	_dSumError = 0.0;

	_nNextBatchNumber_CV = 0;

	return error;
}





