#include <WikiMaxoutNet.h>
using std::cout;
using std::endl;

WikiMaxoutNet::WikiMaxoutNet(ClusterNet gpus)
{


	int vocabSize = 100002;
	int nWordVectorDim = 256;
	int nWindowSize = 11;
	_layers.push_back(512);
	_nMaxoutSize = 1;
	_learningRate = 0.01;
	_nCVErrorPeriodicity = 5000;
	_nCVErrorLength = 10000;
	MOMENTUM = 0.5;
	gpu = gpus;
	_nCurrentDataSet = gpu.MYRANK;
	_X = 0;
	int cv_set_number = 63-gpu.MYRANK;
	_CV_X = read_hdf5(("/home/tim/data/wiki/extracted2/AA/data100000/wiki_" + NumberToString(cv_set_number) + ".p").c_str());
	_nNextBatchNumber = 0;
	_nNextBatchNumber_CV = 0;
	_nBatchSize = 256;
	_RMS_multiplier = 0.9f;
	cudaGetDeviceCount(&GPU_COUNT);
	cudaSetDevice(0);
	CURRENT_GPU = 0;

	STREAMS = (cudaStream_t*)malloc(sizeof(cudaStream_t)*GPU_COUNT);

	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		cudaStream_t s = STREAMS[i];
		cudaStreamCreate(&s);

	}

	CV_container = empty_cpu(10000,1);
 	for(int i = 0; i < CV_container->size; i++)


	useRMSProp = true;
	useMaxout = false;

	cout << "_nMaxoutSize: " << _nMaxoutSize << endl;
	cout << "_layers: " << _layers[0] << endl;
	cout << "nWordVectorDim: " << nWordVectorDim << endl;
	cout << "_nBatchSize: " << _nBatchSize << endl;
	cout << "_learningRate: " << _learningRate << endl;
    cout << "Use RMSProp: "  << useRMSProp << endl;


    Matrix **w1 = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
    Matrix **w2 = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
    W.push_back(w1);
    W.push_back(w2);
    Matrix **b1 = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
    Matrix **b2 = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
    B.push_back(b1);
    B.push_back(b2);
    Matrix **m1 = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
    Matrix **m2 = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
    M.push_back(m1);
    M.push_back(m2);
    Matrix **mb1 = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
    Matrix **mb2 = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
    M_B.push_back(mb1);
    M_B.push_back(mb2);

    for(int i = 0; i < GPU_COUNT; i++)
    {
    	cudaSetDevice(i);
		W[0][i] = gpu.uniformSqrtWeight(nWordVectorDim*nWindowSize,_layers[0]);
		W[1][i] = gpu.uniformSqrtWeight(_layers[0]/_nMaxoutSize, 1);
		B[0][i] = zeros(1,_layers[0]);
		B[1][i] = zeros(1,1);
		M[0][i] = zeros(nWordVectorDim*nWindowSize,_layers[0]);
		M[1][i] = zeros(_layers[0]/_nMaxoutSize, 1);
		M_B[0][i] = zeros(1,_layers[0]);
		M_B[1][i]  = zeros(1,1);
		cudaStream_t s;
		cudaStreamCreate(&s);
		_streamNextBatch.push_back(s);
    }

	for(int i = W.size()-1; i >= 0; i--)
	{
		Matrix **gradX = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
		arrGRAD.push_back(gradX);
		Matrix **gradY = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
		arrGRAD.push_back(gradY);
		Matrix **gradX_B = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
		arrGRAD_B.push_back(gradX_B);
		Matrix **gradY_B = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
		arrGRAD_B.push_back(gradY_B);
		Matrix **msgrad = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
		MSGRAD.push_back(msgrad);
		Matrix **msgrad_bias = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
		MSBGRAD.push_back(msgrad_bias);
	}
	Matrix **gradX = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
	arrGRAD.push_back(gradX);
	Matrix **gradY = (Matrix**)malloc(sizeof(Matrix*)*GPU_COUNT);
	arrGRAD.push_back(gradY);


	for(int j = 0; j < GPU_COUNT; j++)
	{
		cudaSetDevice(j);
		for(int i = W.size()-1; i >= 0; i--)
		{
			MSGRAD[0][j] = zeros(W[i][j]->rows, W[i][j]->cols);
			MSGRAD[1][j] = zeros(W[i][j]->rows, W[i][j]->cols);
			MSBGRAD[0][j] = zeros(B[i][j]->rows, B[i][j]->cols);
			MSBGRAD[1][j] = zeros(B[i][j]->rows, B[i][j]->cols);
		}
	}




	for(int j =0; j < GPU_COUNT; j++)
	{
		cudaSetDevice(j);
		int idx = 0;
		for(int i = W.size()-1; i >= 0; i--)
		{
			arrGRAD[idx][j] = zeros(W[i][j]->rows, W[i][j]->cols);
			arrGRAD_B[idx][j] = zeros(B[i][j]->rows, B[i][j]->cols);
			idx++;
			arrGRAD[idx][j] = (zeros(W[i][j]->rows, W[i][j]->cols));
			arrGRAD_B[idx][j] = zeros(B[i][j]->rows, B[i][j]->cols);
			idx++;
		}
		arrGRAD[4][j] = zeros(_nBatchSize,nWordVectorDim*nWindowSize);
		arrGRAD[5][j] = zeros(_nBatchSize,nWordVectorDim*nWindowSize);


		//???
		//MSGRAD[j].push_back(zeros(_nBatchSize*gpu.MPI_SIZE,nWordVectorDim*nWindowSize));
		//MSGRAD[j].push_back(zeros(_nBatchSize*gpu.MPI_SIZE,nWordVectorDim*nWindowSize));
	}







	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		_Vocab.push_back(gpu.uniformSqrtWeight(nWordVectorDim,vocabSize));
		//_Vocab = gpu.sparseInitWeight(nWordVectorDim,vocabSize);
		//_Vocab = gpu.rand(nWordVectorDim,vocabSize);
		//scalarMul(_Vocab,0.01f,_Vocab);
		//scalarAdd(_Vocab,-0.5f,_Vocab);
		_Vocab_grad.push_back(zeros(nWordVectorDim,vocabSize));
		_MSVocab_grad.push_back(zeros(nWordVectorDim,vocabSize));
		_MSVocab_grad_Y.push_back(zeros(nWordVectorDim,vocabSize));
		M_VocabX.push_back(zeros(nWordVectorDim,vocabSize));
		M_VocabY.push_back(zeros(nWordVectorDim,vocabSize));
		_Vocab_grad_idx.push_back(zeros(1,vocabSize));

		d0.push_back(zeros(_nBatchSize,nWordVectorDim*nWindowSize));
		z1.push_back(zeros(_nBatchSize, _layers[0]));
		a1_Y.push_back(zeros(_nBatchSize, _layers[0]/_nMaxoutSize));
		a1_idx_Y.push_back(zeros(_nBatchSize, _layers[0]/_nMaxoutSize));
		a1_X.push_back(zeros(_nBatchSize, _layers[0]/_nMaxoutSize));
		a1_idx_X.push_back(zeros(_nBatchSize, _layers[0]/_nMaxoutSize));
		d1.push_back(zeros(_nBatchSize, _layers[0]/_nMaxoutSize));
		z2_X.push_back(zeros(_nBatchSize, 1));
		z2_Y.push_back(zeros(_nBatchSize, 1));

		out.push_back(zeros(_nBatchSize,1));
		pairwise_grad.push_back(zeros(_nBatchSize,1));
		e1.push_back(empty(_nBatchSize,1));
		aB.push_back(ones(1,_nBatchSize));
		e2_partial.push_back(zeros(_nBatchSize,W[1][i]->rows));
		e2.push_back(empty(_nBatchSize,e2_partial[i]->cols*_nMaxoutSize));


		_batchX.push_back(zeros(_nBatchSize, nWordVectorDim*nWindowSize));
		_batchY.push_back(zeros(_nBatchSize, nWordVectorDim*nWindowSize));
		_currentBatchIdx_X.push_back(zeros(_nBatchSize,nWindowSize));
		_currentBatchIdx_Y.push_back(zeros(_nBatchSize,nWindowSize));
		_nextBatchIdx.push_back(zeros(_nBatchSize,nWindowSize));
	}





	cudaSetDevice(0);
	for(int i = 1; i < GPU_COUNT; i++)
	{
		cudaMemcpyPeer(W[0][i]->data,i,W[0][0]->data,0,W[0][0]->bytes);
		cudaMemcpyPeer(W[1][i]->data,i,W[1][0]->data,0,W[1][0]->bytes);
		cudaMemcpyPeer(_Vocab[i]->data,i,_Vocab[0]->data,0,_Vocab[0]->bytes);
	}






	_dSumError = 0.0;
	loadNextDataSet();
}


void WikiMaxoutNet::run()
{
	cudaSetDevice(0);
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
			if( i > 0 && i % 25000 == 0)
			{
				cout << "Saving vocabulary matrix to disk..." << endl;
				Matrix *host = to_host(_Vocab[0]);
				write_hdf5("/home/tim/data/wiki/vocab.hdf5",host);
				free(host->data);
				free(host);
				write_hdf5("/home/tim/data/wiki/CV_values.hdf5",CV_container);
			}

			double error = 0.0;//calculateError();
			CV_container->data[i/_nCVErrorPeriodicity] = (float)error;
			cout << "BatchNo: " << i << endl;
			cout << "Cross validation error: " <<  error << endl;
			i++;

			MOMENTUM+= 0.01;
			if( MOMENTUM > 0.95)
				MOMENTUM = 0.95;

			//_RMS_multiplier-= 0.01;
			//if( _RMS_multiplier < 0.25)
				//_RMS_multiplier = 0.25;

			stop = clock();

			double time_interval_seconds = (double((stop - start)) / CLOCKS_PER_SEC) ;
			cout << "Approximate time left in hours: " << (1.0f/(((i*_nBatchSize)/(float)_X->rows)/63.0))*time_interval_seconds/(float)3600.0 << endl;

		}
		else
		{
			//nesterov();
			//feedforward();
			//backprop();
			//weightUpdates();
		}

		allocateNextBatch(false);
		i++;
		cout << i << endl;
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

	if(_X != 0)
		cudaFreeHost(_X->data);
	_X = read_hdf5((path + number + ending).c_str());
	_nCurrentDataSet += 1;
	_batches = _X->rows/ _nBatchSize;
	_nNextBatchNumber = 0;
}


void WikiMaxoutNet::allocateNextBatch(bool isCV)
{
	for(int i = 0; i < GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		if(!isCV)
		{
			if(_nNextBatchNumber < 0)
				_nNextBatchNumber = 0;

			if (_nBatchSize*11*(_nNextBatchNumber+1) > _X->size)
				loadNextDataSet();

			cout << i << " gpuid" << endl;
			cout << "batchno: " << _nNextBatchNumber << endl;

			if(_nNextBatchNumber > GPU_COUNT-1 || _nCurrentDataSet > 1)
			{
				cout << "sync" << endl;
				cudaStreamSynchronize(_streamNextBatch[i]);
				cout << "post sync" << endl;
				to_col_major(_nextBatchIdx[i], _currentBatchIdx_X[i]);
				cout << "post colmajor" << endl;
				gpu.construct_vocab_matrix(_currentBatchIdx_X[i], _currentBatchIdx_Y[i], _batchX[i], _batchY[i], _Vocab[i]);
				cout << "post vocab construct" << endl;

			}

			cout << "pre copy" << endl;

			cudaMemcpyAsync(_nextBatchIdx[i]->data,&_X->data[_nBatchSize*11*_nNextBatchNumber],
						_nBatchSize*11*sizeof(float),
						cudaMemcpyHostToDevice,_streamNextBatch[i]);

			cout << "post copy" << endl;


			_nNextBatchNumber+=1;

		}
		else
		{
			if(_nNextBatchNumber_CV > GPU_COUNT-1)
			{
				cudaStreamSynchronize(_streamNextBatch[i]);
				to_col_major(_nextBatchIdx[i], _currentBatchIdx_X[i]);
				gpu.construct_vocab_matrix(_currentBatchIdx_X[i], _currentBatchIdx_Y[i], _batchX[i], _batchY[i], _Vocab[i]);
			}

			cudaMemcpyAsync(_nextBatchIdx[i]->data,&_CV_X->data[_nBatchSize*11*_nNextBatchNumber_CV],
						_nBatchSize*11*sizeof(float),
						cudaMemcpyHostToDevice,_streamNextBatch[i]);


			_nNextBatchNumber_CV+=1;
		}
	}

}


void WikiMaxoutNet::nesterov()
{


	/*
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

	//added
	scalarMul(M_VocabY, MOMENTUM, M_VocabY);
	add(_Vocab,M_VocabY,_Vocab);
	*/
}

/*
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
		logistic(z1,a1_X);
		gpu.dot(a1_X,W[1],z2_X);
		addMatrixVector(z2_X,B[1],z2_X);

		gpu.dot(_batchY,W[0],z1);
		addMatrixVector(z1,B[0],z1);
		logistic(z1,a1_Y);
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
		RMSprop_with_nesterov_weight_update(MSGRAD[0],arrGRAD[0][gpu.MYRANK],W[1],M[1],0.9f,_learningRate/(float)arrGRAD[0][gpu.MYRANK]->rows,_nBatchSize, MOMENTUM);
		RMSprop_with_nesterov_weight_update(MSGRAD[1],arrGRAD[1][gpu.MYRANK],W[1],M[1],0.9f,_learningRate/(float)arrGRAD[1][gpu.MYRANK]->rows,_nBatchSize, MOMENTUM);
		RMSprop_with_nesterov_weight_update(MSGRAD[2],arrGRAD[2][gpu.MYRANK],W[0],M[0],0.9f,_learningRate/(float)arrGRAD[2][gpu.MYRANK]->rows,_nBatchSize, MOMENTUM);
		RMSprop_with_nesterov_weight_update(MSGRAD[3],arrGRAD[3][gpu.MYRANK],W[0],M[0],0.9f,_learningRate/(float)arrGRAD[3][gpu.MYRANK]->rows,_nBatchSize, MOMENTUM);


		RMSprop_with_nesterov_weight_update(MSBGRAD[0],arrGRAD_B[0][gpu.MYRANK],B[1],M_B[1],0.9f,_learningRate,_nBatchSize, MOMENTUM);
		RMSprop_with_nesterov_weight_update(MSBGRAD[1],arrGRAD_B[1][gpu.MYRANK],B[1],M_B[1],0.9f,_learningRate,_nBatchSize, MOMENTUM);
		RMSprop_with_nesterov_weight_update(MSBGRAD[2],arrGRAD_B[2][gpu.MYRANK],B[0],M_B[0],0.9f,_learningRate,_nBatchSize, MOMENTUM);
		RMSprop_with_nesterov_weight_update(MSBGRAD[3],arrGRAD_B[3][gpu.MYRANK],B[0],M_B[0],0.9f,_learningRate,_nBatchSize, MOMENTUM);


		//update_vocab_with_gradient(GRAD[4],_currentBatchIdx_Y,_Vocab,0.01/(float)_nBatchSize);
		//update_vocab_with_gradient(GRAD[5],_currentBatchIdx_Y,_Vocab,0.01/(float)_nBatchSize);


		//fill_matrix(_Vocab_grad,0.0f);
		//expand_double_vocab_gradient(GRAD[5],GRAD[4],_currentBatchIdx_X,_currentBatchIdx_Y,_Vocab,_Vocab_grad,_Vocab_grad_idx,_learningRate/(float)_nBatchSize);
		//RMSprop_with_nesterov_weight_update(_MSVocab_grad,_Vocab_grad,_Vocab,_MVocab,_RMS_multiplier,_learningRate/(float)_nBatchSize,1);



		fill_matrix(_Vocab_grad,0.0f);
		expand_vocab_gradient(arrGRAD[5][gpu.MYRANK],_currentBatchIdx_X,_Vocab_grad);
		RMSprop_with_nesterov_weight_update(_MSVocab_grad,_Vocab_grad,_Vocab,M_VocabX,_RMS_multiplier,_learningRate/(float)_nBatchSize,1, MOMENTUM);

		fill_matrix(_Vocab_grad,0.0f);
		expand_vocab_gradient(arrGRAD[4][gpu.MYRANK],_currentBatchIdx_Y,_Vocab_grad);
		RMSprop_with_nesterov_weight_update(_MSVocab_grad_Y,_Vocab_grad,_Vocab,M_VocabY,_RMS_multiplier,_learningRate/(float)_nBatchSize,1, MOMENTUM);


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
    	logisticGrad(a1_Y,a1_Y);
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
    	logisticGrad(a1_X,a1_X);
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

*/



