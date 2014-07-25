#include <WikiMaxoutNet_PCIe.h>
#include <sched.h>
using std::cout;
using std::endl;


WikiMaxoutNet_PCIe::WikiMaxoutNet_PCIe(ClusterNet *gpus)
{

	int vocabSize = 100002;
	int nWordVectorDim = 64;
	int nWindowSize = 11;
	_layers.push_back(512);
	_learningRate = 0.01;
	_nCVErrorPeriodicity = 6000;
	_nCVErrorLength = 6000;
	MOMENTUM = 0.5;
	gpu = gpus[0];
	_nCurrentDataSet = gpu.MYRANK;
	_X = 0;
	int cv_set_number = 63-gpu.MYRANK;
	cudaSetDevice(0);
	_CV_X = read_hdf5(("/home/tim/data/wiki/extracted2/AA/data100000/wiki_" + NumberToString(cv_set_number) + ".p").c_str());
	_nNextBatchNumber = 0;
	_nNextBatchNumber_CV = 0;
	_nBatchSize = 128;
	_RMS_multiplier = 0.9f;


	//TODO set weights equal

	gpu.GPU_COUNT = 3;


	for(int i = 0; i < gpu.GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		cudaStream_t t;
		cudaStreamCreate(&t);
		_streamNextBatch.push_back(t);

	}
	cudaSetDevice(0);




	useRMSProp = true;

	cout << "_layers: " << _layers[0] << endl;
	cout << "nWordVectorDim: " << nWordVectorDim << endl;
	cout << "_nBatchSize: " << _nBatchSize << endl;
	cout << "_learningRate: " << _learningRate << endl;
    cout << "Use RMSProp: "  << useRMSProp << endl;

	W.push_back(gpu.uniformSqrtWeight_PCIe(nWordVectorDim*nWindowSize,_layers[0]));
	W.push_back(gpu.uniformSqrtWeight_PCIe(_layers[0], 1));
	B.push_back(gpu.zeros_PCIe(1,_layers[0]));
	B.push_back(gpu.zeros_PCIe(1,1));
	M.push_back(gpu.zeros_PCIe(nWordVectorDim*nWindowSize,_layers[0]));
	M.push_back(gpu.zeros_PCIe(_layers[0], 1));
	M_B.push_back(gpu.zeros_PCIe(1,_layers[0]));
	M_B.push_back(gpu.zeros_PCIe(1,1));



	CV_container = empty_cpu(10000,1);
	for(int i = 0; i < CV_container->size; i++)
		CV_container->data[i] = 0.0f;



	cout << gpu.GPU_COUNT << " MPI SIZE" << endl;
	cout << gpu.MYRANK << " MYRANK " << endl;
	for(int i = W.size()-1; i >= 0; i--)
	{
		Matrix **gradX = (Matrix**)malloc(sizeof(Matrix*)*gpu.GPU_COUNT);
		arrGRAD.push_back(gradX);
		Matrix **gradY = (Matrix**)malloc(sizeof(Matrix*)*gpu.GPU_COUNT);
		arrGRAD.push_back(gradY);
		Matrix **gradX_B = (Matrix**)malloc(sizeof(Matrix*)*gpu.GPU_COUNT);
		arrGRAD_B.push_back(gradX_B);
		Matrix **gradY_B = (Matrix**)malloc(sizeof(Matrix*)*gpu.GPU_COUNT);
		arrGRAD_B.push_back(gradY_B);
	}
	Matrix **gradX = (Matrix**)malloc(sizeof(Matrix*)*gpu.GPU_COUNT);
	arrGRAD.push_back(gradX);
	Matrix **gradY = (Matrix**)malloc(sizeof(Matrix*)*gpu.GPU_COUNT);
	arrGRAD.push_back(gradY);

	cout << arrGRAD.size() << " size" << endl;


	for(int i = W.size()-1; i >= 0; i--)
	{
		MSGRAD.push_back(gpu.zeros_PCIe(W[i][0]->rows, W[i][0]->cols));
		MSGRAD.push_back(gpu.zeros_PCIe(W[i][0]->rows, W[i][0]->cols));
		MSBGRAD.push_back(gpu.zeros_PCIe(B[i][0]->rows, B[i][0]->cols));
		MSBGRAD.push_back(gpu.zeros_PCIe(B[i][0]->rows, B[i][0]->cols));
	}

	for(int j =0; j < gpu.GPU_COUNT; j++)
	{
		int idx = 0;
		for(int i = W.size()-1; i >= 0; i--)
		{
			arrGRAD[idx] = gpu.zeros_gradient_PCIe(W[i][0]->rows, W[i][0]->cols);
			arrGRAD_B[idx] = gpu.zeros_gradient_PCIe(B[i][0]->rows, B[i][0]->cols);
			idx++;
			arrGRAD[idx] = (gpu.zeros_gradient_PCIe(W[i][0]->rows, W[i][0]->cols));
			arrGRAD_B[idx] = gpu.zeros_gradient_PCIe(B[i][0]->rows, B[i][0]->cols);
			idx++;
		}

		arrGRAD[4] = gpu.zeros_gradient_PCIe(_nBatchSize,nWordVectorDim*nWindowSize);
		arrGRAD[5] = gpu.zeros_gradient_PCIe(_nBatchSize,nWordVectorDim*nWindowSize);
	}

	_Vocab = gpu.uniformSqrtWeight_PCIe(nWordVectorDim,vocabSize);
	//_Vocab = gpu.sparseInitWeight(nWordVectorDim,vocabSize);
	//_Vocab = gpu.rand(nWordVectorDim,vocabSize);
	//scalarMul(_Vocab,0.01f,_Vocab);
	//scalarAdd(_Vocab,-0.5f,_Vocab);
	_Vocab_grad = gpu.zeros_PCIe(nWordVectorDim,vocabSize);
	_MSVocab_grad = gpu.zeros_PCIe(nWordVectorDim,vocabSize);
	_MSVocab_grad_Y = gpu.zeros_PCIe(nWordVectorDim,vocabSize);
	M_VocabX = gpu.zeros_PCIe(nWordVectorDim,vocabSize);
	M_VocabY = gpu.zeros_PCIe(nWordVectorDim,vocabSize);
	_Vocab_grad_idx = gpu.zeros_PCIe(1,vocabSize);

	d0 = gpu.zeros_PCIe(_nBatchSize,nWordVectorDim*nWindowSize);
	z1 = gpu.zeros_PCIe(_nBatchSize, _layers[0]);
	a1_Y = gpu.zeros_PCIe(_nBatchSize, _layers[0]);
	a1_idx_Y = gpu.zeros_PCIe(_nBatchSize, _layers[0]);
	a1_X = gpu.zeros_PCIe(_nBatchSize, _layers[0]);
	a1_idx_X = gpu.zeros_PCIe(_nBatchSize, _layers[0]);
	d1 = gpu.zeros_PCIe(_nBatchSize, _layers[0]);
	z2_X = gpu.zeros_PCIe(_nBatchSize, 1);
	z2_Y = gpu.zeros_PCIe(_nBatchSize, 1);

	out = gpu.zeros_PCIe(_nBatchSize,1);
	pairwise_grad = gpu.zeros_PCIe(_nBatchSize,1);
	e1 = gpu.zeros_PCIe(_nBatchSize,1);
	aB = gpu.ones_PCIe(1,_nBatchSize);
	e2_partial = gpu.zeros_PCIe(_nBatchSize,W[1][0]->rows);
	e2 = gpu.zeros_PCIe(_nBatchSize,e2_partial[0]->cols);


	_batchX = gpu.zeros_PCIe(_nBatchSize, nWordVectorDim*nWindowSize);
	_batchY = gpu.zeros_PCIe(_nBatchSize, nWordVectorDim*nWindowSize);
	_currentBatchIdx_X = gpu.zeros_PCIe(_nBatchSize,nWindowSize);
	_currentBatchIdx_Y = gpu.zeros_PCIe(_nBatchSize,nWindowSize);
	_nextBatchIdx = gpu.zeros_PCIe(_nBatchSize,nWindowSize);

	_dSumError = 0.0;



	loadNextDataSet();
}

void WikiMaxoutNet_PCIe::loadNextDataSet()
{


	std::string path = "/home/tim/data/wiki/extracted2/AA/data100000/wiki_";
	//std::string path = "/home/tim/data/wiki/extracted2/AA/data/wiki_";
	std::string number = "";
	std::string ending = ".p";
	//std::string ending = ".p.hdf5";

	if(_nCurrentDataSet < 10)
		number += "0";

	number+= NumberToString(_nCurrentDataSet);

	cout << "Loading next data set: " << (path + number + ending) << endl;
	if(_X != 0)
		cudaFreeHost(_X->data);
	_X = read_hdf5((path + number + ending).c_str());
	_nCurrentDataSet += 1;
	_batches = _X->rows/ _nBatchSize;
	_nNextBatchNumber = 0;
}



void WikiMaxoutNet_PCIe::allocateNextBatch(bool isCV)
{
	for(int i = 0; i < gpu.GPU_COUNT; i++)
	{
		cudaSetDevice(i);

		if(!isCV)
		{
			if(_nNextBatchNumber < 0)
				_nNextBatchNumber = 0;

			if (_nBatchSize*11*(_nNextBatchNumber+1) > _X->size)
				loadNextDataSet();

			if(_nNextBatchNumber >= gpu.GPU_COUNT)
			{
				cout << "pre sync" << endl;
				cudaStreamSynchronize(_streamNextBatch[i]);
				cout << "post sync" << endl;
				to_col_major(_nextBatchIdx[i], _currentBatchIdx_X[i]);
				cout << "post col major" << endl;
				gpu.construct_vocab_matrix(_currentBatchIdx_X[i], _currentBatchIdx_Y[i], _batchX[i], _batchY[i], _Vocab[i]);
				cout << "post constructed" << endl;
			}



				cudaMemcpyAsync(_nextBatchIdx[i]->data,&_X->data[_nBatchSize*11*_nNextBatchNumber],
							_nBatchSize*11*sizeof(float),
							cudaMemcpyHostToDevice,_streamNextBatch[i]);


			_nNextBatchNumber+=1;
		}
		else
		{
			if(_nNextBatchNumber_CV >= gpu.GPU_COUNT)
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




void WikiMaxoutNet_PCIe::run()
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
				cudaSetDevice(0);
				Matrix *host = to_host(_Vocab[0]);
				write_hdf5("/home/tim/data/wiki/vocab.hdf5",host);
				free(host->data);
				free(host);
				write_hdf5("/home/tim/data/wiki/CV_values.hdf5",CV_container);
			}

			double error = calculateError();
			CV_container->data[i/_nCVErrorPeriodicity] = (float)error;
			cout << "BatchNo: " << i << endl;
			cout << "Cross validation error: " <<  error << endl;
			i+=gpu.GPU_COUNT;

			//MOMENTUM+= 0.01;
			//if( MOMENTUM > 0.95)
				//MOMENTUM = 0.95;

			_RMS_multiplier-= 0.01;
			if( _RMS_multiplier < 0.25)
				_RMS_multiplier = 0.25;

			stop = clock();

			double time_interval_seconds = (double((stop - start)) / CLOCKS_PER_SEC) ;
			cout << "Approximate time left in hours: " << ((1.0f/(((i*_nBatchSize)/(float)_X->rows)/63.0))*time_interval_seconds/(float)3600.0)  -
					(time_interval_seconds/(float)3600.0)<< endl;

		}
		else
		{
			cout <<  "nesterov" << endl;
			nesterov();
			feedforward();
			backprop();
			weightUpdates();
		}

		cout << i << endl;
		allocateNextBatch(false);
		i+=gpu.GPU_COUNT;

		cout << i << endl;
	}
}





void WikiMaxoutNet_PCIe::nesterov()
{
	//nesterov
	for(int i = 0;i < M.size(); i++)
	{
		gpu.scalarMul_PCIe(M[i],MOMENTUM,M[i]);
		gpu.add_PCIe(W[i],M[i],W[i]);
	}

	for(int i = 0;i < B.size(); i++)
	{
		gpu.scalarMul_PCIe(M_B[i],MOMENTUM,M_B[i]);
		gpu.add_PCIe(B[i],M_B[i],B[i]);
	}

	gpu.scalarMul_PCIe(M_VocabX, MOMENTUM, M_VocabX);
	gpu.add_PCIe(_Vocab,M_VocabX,_Vocab);

	gpu.scalarMul_PCIe(M_VocabY, MOMENTUM, M_VocabY);
	gpu.add_PCIe(_Vocab,M_VocabY,_Vocab);

}


void WikiMaxoutNet_PCIe::feedforward()
{
	gpu.dotPCIe(_batchX,W[0],z1);
	gpu.addMatrixVector_PCIe(z1,B[0],z1);
	gpu.logistic_PCIe(z1,a1_X);
	gpu.dotPCIe(a1_X,W[1],z2_X);
	gpu.addMatrixVector_PCIe(z2_X,B[1],z2_X);

	gpu.dotPCIe(_batchY,W[0],z1);
	gpu.addMatrixVector_PCIe(z1,B[0],z1);
	gpu.logistic_PCIe(z1,a1_Y);
	gpu.dotPCIe(a1_Y,W[1],z2_Y);
	gpu.addMatrixVector_PCIe(z2_Y,B[1],z2_Y);


}


void WikiMaxoutNet_PCIe::weightUpdates()
{
	float multiplier = _learningRate/(float)_nBatchSize;

	if(!useRMSProp)
	{
		/*
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
		update_vocab_with_gradient(arrGRAD[5][gpu.MYRANK],_currentBatchIdx_X,_Vocab,multiplier);
		*/
	}
	else
	{

		//10*GPU_COUNT gradients added

		cout << gpu.get_queue_length() << endl;
		while(gpu.get_queue_length() > 0)
		{
			gpu.pop_queue_PCIe();
			usleep(100);
		}

		cout << "past lol" << endl;

		for(int i = 0; i < gpu.GPU_COUNT; i++)
		{
			cudaSetDevice(i);
			fill_matrix(_Vocab_grad[i],0.0f);
			expand_vocab_gradient(arrGRAD[4][i],_currentBatchIdx_Y[i],_Vocab_grad[i]);
		}
		//cout << gpu.get_queue_length() << endl;
		gpu.RMSprop_with_nesterov_weight_update_PCIe(_MSVocab_grad_Y,_Vocab_grad,_Vocab,M_VocabY,_RMS_multiplier,_learningRate/(float)_nBatchSize,1, MOMENTUM);
		gpu.RMSprop_with_nesterov_weight_update_PCIe(MSGRAD[2],arrGRAD[2],W[0],M[0],0.9f,_learningRate/(float)arrGRAD[2][gpu.MYRANK]->rows,_nBatchSize, MOMENTUM);
		for(int i = 0; i < gpu.GPU_COUNT; i++)
		{
			cudaSetDevice(i);
			fill_matrix(_Vocab_grad[i],0.0f);
			expand_vocab_gradient(arrGRAD[5][i],_currentBatchIdx_X[i],_Vocab_grad[i]);
		}
		gpu.RMSprop_with_nesterov_weight_update_PCIe(_MSVocab_grad,_Vocab_grad,_Vocab,M_VocabX,_RMS_multiplier,_learningRate/(float)_nBatchSize,1, MOMENTUM);
		gpu.RMSprop_with_nesterov_weight_update_PCIe(MSGRAD[3],arrGRAD[3],W[0],M[0],0.9f,_learningRate/(float)arrGRAD[3][gpu.MYRANK]->rows,_nBatchSize, MOMENTUM);
		gpu.RMSprop_with_nesterov_weight_update_PCIe(MSGRAD[0],arrGRAD[0],W[1],M[1],0.9f,_learningRate/(float)arrGRAD[0][gpu.MYRANK]->rows,_nBatchSize, MOMENTUM);
		gpu.RMSprop_with_nesterov_weight_update_PCIe(MSGRAD[1],arrGRAD[1],W[1],M[1],0.9f,_learningRate/(float)arrGRAD[1][gpu.MYRANK]->rows,_nBatchSize, MOMENTUM);
		gpu.RMSprop_with_nesterov_weight_update_PCIe(MSBGRAD[2],arrGRAD_B[2],B[0],M_B[0],0.9f,_learningRate,_nBatchSize, MOMENTUM);
		gpu.RMSprop_with_nesterov_weight_update_PCIe(MSBGRAD[0],arrGRAD_B[0],B[1],M_B[1],0.9f,_learningRate,_nBatchSize, MOMENTUM);
		gpu.RMSprop_with_nesterov_weight_update_PCIe(MSBGRAD[1],arrGRAD_B[1],B[1],M_B[1],0.9f,_learningRate,_nBatchSize, MOMENTUM);
		gpu.RMSprop_with_nesterov_weight_update_PCIe(MSBGRAD[3],arrGRAD_B[3],B[0],M_B[0],0.9f,_learningRate,_nBatchSize, MOMENTUM);

	}

}


void WikiMaxoutNet_PCIe::backprop()
{
	for(int i = 0; i < gpu.GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		pairwise_ranking(z2_X[i],z2_Y[i], out[i]);
		pairwise_ranking_derivative(z2_X[i],z2_Y[i], pairwise_grad[i]);
	}



	gpu.mul_PCIe(out, pairwise_grad, e1);
	gpu.dotTPCIe(e1, W[1],e2_partial);

	gpu.dotPCIe(aB,e1,arrGRAD_B[0]);

	//gpu.add_to_queue_PCIe(arrGRAD_B[0]);

	gpu.TdotPCIe(a1_Y,e1,arrGRAD[0]);
	gpu.add_to_queue_PCIe(arrGRAD[0]);

	for(int i = 0; i < gpu.GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		logisticGrad(a1_Y[i],a1_Y[i]);
	}
	gpu.mul_PCIe(e2_partial,a1_Y,e2);


    gpu.TdotPCIe(_batchY,e2,arrGRAD[2]);
	//gpu.add_to_queue_PCIe(arrGRAD[2]);
    gpu.dotPCIe(aB,e2,arrGRAD_B[2]);
	//gpu.add_to_queue_PCIe(arrGRAD_B[2]);
    gpu.dotTPCIe(e2,W[0],arrGRAD[4]);
	//gpu.add_to_queue_PCIe(arrGRAD[4]);

	gpu.scalarMul_PCIe(pairwise_grad,-1.0f,pairwise_grad);
	gpu.mul_PCIe(out, pairwise_grad, e1);

	gpu.dotPCIe(aB,e1,arrGRAD_B[1]);
	//gpu.add_to_queue_PCIe(arrGRAD_B[1]);
	gpu.TdotPCIe(a1_X,e1,arrGRAD[1]);
	//gpu.add_to_queue_PCIe(arrGRAD[1]);
	gpu.dotTPCIe(e1, W[1],e2_partial);


	for(int i = 0; i < gpu.GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		logisticGrad(a1_X[i],a1_X[i]);
	}
	gpu.mul_PCIe(e2_partial,a1_X,e2);


    gpu.TdotPCIe(_batchX,e2,arrGRAD[3]);
	//gpu.add_to_queue_PCIe(arrGRAD[3]);
	gpu.dotPCIe(aB,e2,arrGRAD_B[3]);
	//gpu.add_to_queue_PCIe(arrGRAD_B[3]);
    gpu.dotTPCIe(e2,W[0],arrGRAD[5]);
    //gpu.add_to_queue_PCIe(arrGRAD[5]);
}


double WikiMaxoutNet_PCIe::calculateError()
{
	//scalarMul(W[0],0.9,W[0]);
	allocateNextBatch(true);
	for(int i = 0; i < _nCVErrorLength; i+=gpu.GPU_COUNT)
	{

		feedforward();

		for(int j = 0; j < gpu.GPU_COUNT; j++)
		{
			cudaSetDevice(j);
			pairwise_ranking(z2_X[j],z2_Y[j], out[j]);
		}
		cudaSetDevice(0);
		_dSumError += (double)sum(out[0]);

		allocateNextBatch(true);
	}
	//scalarMul(W[0],1.1,W[0]);
	//size_t free, total;
	//cudaMemGetInfo(&free, &total);
	//cout << free << endl;
	//cout << "Free system memory: " << sysconf(_SC_PAGE_SIZE)*sysconf(_SC_PHYS_PAGES) << endl;


	double error = _dSumError/(double)(_nBatchSize*_nCVErrorLength);
	_dSumError = 0.0;



	_nNextBatchNumber_CV = 0;

	return error;
}





