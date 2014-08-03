#include <WikiMaxoutNet.h>
#include <sched.h>
using std::cout;
using std::endl;

WikiMaxoutNet::WikiMaxoutNet(ClusterNet gpus)
{

	int vocabSize = 100002;
	int nWordVectorDim = 120;
	int nWindowSize = 11;
	_layers.push_back(128);
	_learningRate = 0.1;
	_nCVErrorPeriodicity = 6000;
	_nCVErrorLength = 6000;
	MOMENTUM = 0.9;
	gpu = gpus;
	_nCurrentDataSet = gpu.MYRANK;
	_X = 0;
	int cv_set_number = 63;
	_CV_X = read_hdf5(("/home/tim/data/wiki/extracted2/AA/data100000/wiki_" + NumberToString(cv_set_number) + ".p").c_str());
	_nNextBatchNumber = 0;
	_nNextBatchNumber_CV = gpu.MYRANK*_nCVErrorLength;
	_nBatchSize = 512;
	_RMS_multiplier = 0.9f;


	cudaStreamCreate(&_streamNextBatch);

	Matrix *learning_rate_matrix_cpu = empty_cpu(nWordVectorDim,vocabSize);

	float learning_rate = 0.0000001;
	int next_level = 2000;
	for(int col = 0; col < vocabSize; col++)
		for(int row = 0; row < nWordVectorDim; row++)
		{
			if(col > next_level)
			{
				learning_rate = learning_rate * 10.00f;
				next_level = next_level == 50000 ? vocabSize : next_level;
				next_level = next_level == 25000 ? 50000 : next_level;
				next_level = next_level == 10000 ? 25000 : next_level;
				next_level = next_level == 2000 ? 10000 : next_level;
			}

			if((col == vocabSize-2) || (col == vocabSize-1))
			{
				learning_rate_matrix_cpu->data[col + (row*vocabSize)] = 0.0000001;
			}
			else
			{
				learning_rate_matrix_cpu->data[col + (row*vocabSize)] = learning_rate;
			}
		}


	learning_rate_matrix = to_gpu(learning_rate_matrix_cpu);
	free(learning_rate_matrix_cpu->data);


	useRMSProp = true;

	cout << "_layers: " << _layers[0] << endl;
	cout << "nWordVectorDim: " << nWordVectorDim << endl;
	cout << "_nBatchSize: " << _nBatchSize << endl;
	cout << "_learningRate: " << _learningRate << endl;
    cout << "Use RMSProp: "  << useRMSProp << endl;

	W.push_back(gpu.uniformSqrtWeight(nWordVectorDim*nWindowSize,_layers[0]));
	W.push_back(gpu.uniformSqrtWeight(_layers[0], 1));
	B.push_back(zeros(1,_layers[0]));
	B.push_back(zeros(1,1));
	M.push_back(zeros(nWordVectorDim*nWindowSize,_layers[0]));
	M.push_back(zeros(_layers[0], 1));
	M_B.push_back(zeros(1,_layers[0]));
	M_B.push_back(zeros(1,1));



	for(int i = 0; i < W.size(); i++)
	{
		cout << sum(W[i]) << endl;
		MPI_Barrier(MPI_COMM_WORLD);
	}

	CV_container = empty_cpu(10000,1);
	for(int i = 0; i < CV_container->size; i++)
		CV_container->data[i] = 0.0f;


	if(gpu.MPI_SIZE == 0)
		gpu.MPI_SIZE = 1;


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

		arrGRAD[4][j] = zeros(_nBatchSize,nWordVectorDim*nWindowSize);
		arrGRAD[5][j] = zeros(_nBatchSize,nWordVectorDim*nWindowSize);
	}



	stackedVocabGrad_X = zeros(_nBatchSize*gpu.MPI_SIZE,nWordVectorDim*nWindowSize);
	stackedVocabGrad_Y = zeros(_nBatchSize*gpu.MPI_SIZE,nWordVectorDim*nWindowSize);
	stackedBatchIdx_X = zeros(_nBatchSize*gpu.MPI_SIZE,nWindowSize);
	stackedBatchIdx_Y = zeros(_nBatchSize*gpu.MPI_SIZE,nWindowSize);
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
	a1_Y = zeros(_nBatchSize, _layers[0]);
	a1_idx_Y = zeros(_nBatchSize, _layers[0]);
	a1_X = zeros(_nBatchSize, _layers[0]);
	a1_idx_X = zeros(_nBatchSize, _layers[0]);
	d1 = zeros(_nBatchSize, _layers[0]);
	z2_X = zeros(_nBatchSize, 1);
	z2_Y = zeros(_nBatchSize, 1);

	out = zeros(_nBatchSize,1);
	pairwise_grad = zeros(_nBatchSize,1);
	e1 = empty(_nBatchSize,1);
	aB = ones(1,_nBatchSize);
	e2_partial = zeros(_nBatchSize,W[1]->rows);
	e2 = empty(_nBatchSize,e2_partial->cols);


	_batchX = zeros(_nBatchSize, nWordVectorDim*nWindowSize);
	_batchY = zeros(_nBatchSize, nWordVectorDim*nWindowSize);

	_currentBatchIdx_X = (Matrix**)malloc(sizeof(Matrix*)*gpu.MPI_SIZE);
	_currentBatchIdx_Y = (Matrix**)malloc(sizeof(Matrix*)*gpu.MPI_SIZE);
	for(int i = 0; i < gpu.MPI_SIZE; i++)
	{
		_currentBatchIdx_X[i] = zeros(_nBatchSize, nWindowSize);
		_currentBatchIdx_Y[i] = zeros(_nBatchSize, nWindowSize);
	}
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
			if( i > 0 && i % 12000 == 0)
			{
				cout << "Saving vocabulary matrix to disk..." << endl;
				Matrix *host = to_host(_Vocab);
				write_hdf5("/home/tim/data/wiki/vocab.hdf5",host);
				free(host->data);
				free(host);
				write_hdf5("/home/tim/data/wiki/CV_values.hdf5",CV_container);
			}

			double error = calculateError();
			CV_container->data[i/_nCVErrorPeriodicity] = (float)error;
			cout << "BatchNo: " << i << endl;
			cout << "Cross validation error: " <<  error << endl;
			i+=gpu.MPI_SIZE;

			//MOMENTUM+= 0.01;
			//if( MOMENTUM > 0.95)
				//MOMENTUM = 0.95;

			_RMS_multiplier-= 0.01;
			if( _RMS_multiplier < 0.25)
				_RMS_multiplier = 0.25;


			MOMENTUM-= 0.01;
			if( MOMENTUM < 0.25)
				MOMENTUM = 0.25;

			stop = clock();

			double time_interval_seconds = (double((stop - start)) / CLOCKS_PER_SEC) ;
			cout << "Approximate time left in hours: " << ((1.0f/(((i*_nBatchSize)/(float)_X->rows)/63.0))*time_interval_seconds/(float)3600.0)  -
					(time_interval_seconds/(float)3600.0)<< endl;

		}
		else
		{
			nesterov();
			feedforward();
			backprop();
			weightUpdates();
		}
		//cout << i << endl;
		allocateNextBatch(false);
		i+=gpu.MPI_SIZE;
	}
}

void WikiMaxoutNet::loadNextDataSet()
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
			to_col_major(_nextBatchIdx, _currentBatchIdx_X[gpu.MYRANK]);
			gpu.construct_vocab_matrix(_currentBatchIdx_X[gpu.MYRANK], _currentBatchIdx_Y[gpu.MYRANK], _batchX, _batchY, _Vocab);

			gpu.add_to_queue(_currentBatchIdx_X);
			gpu.add_to_queue(_currentBatchIdx_Y);
		}



			cudaMemcpyAsync(_nextBatchIdx->data,&_X->data[_nBatchSize*11*_nNextBatchNumber],
						_nBatchSize*11*sizeof(float),
						cudaMemcpyHostToDevice,_streamNextBatch);


		_nNextBatchNumber+=1;
	}
	else
	{
		if(_nNextBatchNumber_CV > gpu.MYRANK*_nCVErrorLength)
		{
			cudaStreamSynchronize(_streamNextBatch);
			to_col_major(_nextBatchIdx, _currentBatchIdx_X[gpu.MYRANK]);
			gpu.construct_vocab_matrix(_currentBatchIdx_X[gpu.MYRANK], _currentBatchIdx_Y[gpu.MYRANK], _batchX, _batchY, _Vocab);
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

	for(int i = 0;i < M_B.size(); i++)
	{
		scalarMul(M_B[i],MOMENTUM,M_B[i]);
		add(B[i],M_B[i],B[i]);
	}

	scalarMul(M_VocabX, MOMENTUM, M_VocabX);
	add(_Vocab,M_VocabX,_Vocab);

	scalarMul(M_VocabY, MOMENTUM, M_VocabY);
	add(_Vocab,M_VocabY,_Vocab);

}


void WikiMaxoutNet::feedforward()
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

void WikiMaxoutNet::weightUpdates()
{
	float multiplier = _learningRate/(float)_nBatchSize;

	if(!useRMSProp)
	{
		scalarMul(M_VocabX,MOMENTUM,M_VocabX);
		scalarMul(M_VocabY,MOMENTUM,M_VocabY);
		for(int i = 0; i < M.size(); i++)
			scalarMul(M[i],MOMENTUM,M[i]);
		for(int i = 0; i < M_B.size(); i++)
			scalarMul(M_B[i],MOMENTUM,M_B[i]);

		while(gpu.get_queue_length() > (9*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		vStackN(arrGRAD[4],stackedVocabGrad_Y,gpu.MPI_SIZE);
		update_vocab_with_gradient(stackedVocabGrad_Y,stackedBatchIdx_Y,_Vocab,multiplier/(float)gpu.MPI_SIZE);
		update_vocab_with_gradient(stackedVocabGrad_Y,stackedBatchIdx_Y,M_VocabY,multiplier/(float)gpu.MPI_SIZE);
		while(gpu.get_queue_length() > (8*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD[2],gpu.MYRANK,gpu.MPI_SIZE,multiplier/(float)(gpu.MPI_SIZE*arrGRAD[1][gpu.MYRANK]->rows));
		sub(W[0],arrGRAD[2][gpu.MYRANK],W[0]);
		sub(M[0],arrGRAD[2][gpu.MYRANK],M[0]);
		while(gpu.get_queue_length() > (7*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		vStackN(arrGRAD[5],stackedVocabGrad_X,gpu.MPI_SIZE);
		update_vocab_with_gradient(stackedVocabGrad_X,stackedBatchIdx_X,_Vocab,multiplier/(float)gpu.MPI_SIZE);
		update_vocab_with_gradient(stackedVocabGrad_X,stackedBatchIdx_X,M_VocabX,multiplier/(float)gpu.MPI_SIZE);
		while(gpu.get_queue_length() > (6*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD[3],gpu.MYRANK,gpu.MPI_SIZE,multiplier/(float)(gpu.MPI_SIZE*arrGRAD[3][gpu.MYRANK]->rows));
		sub(W[0],arrGRAD[3][gpu.MYRANK],W[0]);
		sub(M[0],arrGRAD[3][gpu.MYRANK],M[0]);
		while(gpu.get_queue_length() > (5*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD[0],gpu.MYRANK,gpu.MPI_SIZE,multiplier/(float)(gpu.MPI_SIZE*arrGRAD[0][gpu.MYRANK]->rows));
		sub(W[1],arrGRAD[0][gpu.MYRANK],W[1]);
		sub(M[1],arrGRAD[0][gpu.MYRANK],M[1]);
		while(gpu.get_queue_length() > (4*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD[1],gpu.MYRANK,gpu.MPI_SIZE,multiplier/(float)(gpu.MPI_SIZE*arrGRAD[1][gpu.MYRANK]->rows));
		sub(W[1],arrGRAD[1][gpu.MYRANK],W[1]);
		sub(M[1],arrGRAD[1][gpu.MYRANK],M[1]);
		while(gpu.get_queue_length() > (3*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD_B[2],gpu.MYRANK,gpu.MPI_SIZE,multiplier/(float)gpu.MPI_SIZE);
		sub(B[0],arrGRAD_B[2][gpu.MYRANK],B[0]);
		sub(M_B[0],arrGRAD_B[2][gpu.MYRANK],M_B[0]);
		while(gpu.get_queue_length() > (2*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD_B[0],gpu.MYRANK,gpu.MPI_SIZE,multiplier/(float)gpu.MPI_SIZE);
		sub(B[1],arrGRAD_B[0][gpu.MYRANK],B[1]);
		sub(M_B[1],arrGRAD_B[0][gpu.MYRANK],M_B[1]);
		while(gpu.get_queue_length() > (1*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD_B[1],gpu.MYRANK,gpu.MPI_SIZE,multiplier/(float)gpu.MPI_SIZE);
		sub(B[1],arrGRAD_B[1][gpu.MYRANK],B[1]);
		sub(M_B[1],arrGRAD_B[1][gpu.MYRANK],M_B[1]);
		while(gpu.get_queue_length() > 0){ gpu.pop_queue(); }
		addGradientsN(arrGRAD_B[3],gpu.MYRANK,gpu.MPI_SIZE,multiplier/(float)gpu.MPI_SIZE);
		sub(B[0],arrGRAD_B[3][gpu.MYRANK],B[0]);
		sub(M_B[0],arrGRAD_B[3][gpu.MYRANK],M_B[0]);
	}
	else
	{

		//10*MPI_SIZE gradients added

		fill_matrix(_Vocab_grad,0.0f);
		while(gpu.get_queue_length() > (9*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		vStackN(arrGRAD[4],stackedVocabGrad_Y,gpu.MPI_SIZE);
		expand_vocab_gradient(stackedVocabGrad_Y,stackedBatchIdx_Y,_Vocab_grad);




		RMSprop_with_nesterov_weight_update(_MSVocab_grad_Y,_Vocab_grad,_Vocab,M_VocabY,_RMS_multiplier,_learningRate/(float)_nBatchSize,_nBatchSize*gpu.MPI_SIZE, MOMENTUM);
		while(gpu.get_queue_length() > (8*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD[2],gpu.MYRANK,gpu.MPI_SIZE,1.0f/(float)gpu.MPI_SIZE);
		RMSprop_with_nesterov_weight_update(MSGRAD[2],arrGRAD[2][gpu.MYRANK],W[0],M[0],0.9f,_learningRate/(float)arrGRAD[2][gpu.MYRANK]->rows,_nBatchSize*gpu.MPI_SIZE, MOMENTUM);

		fill_matrix(_Vocab_grad,0.0f);
		while(gpu.get_queue_length() > (7*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		vStackN(arrGRAD[5],stackedVocabGrad_X,gpu.MPI_SIZE);
		expand_vocab_gradient(stackedVocabGrad_X,stackedBatchIdx_X,_Vocab_grad);
		RMSprop_with_nesterov_weight_update(_MSVocab_grad,_Vocab_grad,_Vocab,M_VocabX,_RMS_multiplier,_learningRate/(float)_nBatchSize,_nBatchSize*gpu.MPI_SIZE, MOMENTUM);


		while(gpu.get_queue_length() > (6*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD[3],gpu.MYRANK,gpu.MPI_SIZE,1.0f/(float)gpu.MPI_SIZE);
		RMSprop_with_nesterov_weight_update(MSGRAD[3],arrGRAD[3][gpu.MYRANK],W[0],M[0],0.9f,_learningRate/(float)arrGRAD[3][gpu.MYRANK]->rows,_nBatchSize*gpu.MPI_SIZE, MOMENTUM);

		while(gpu.get_queue_length() > (5*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD[0],gpu.MYRANK,gpu.MPI_SIZE,1.0f/(float)gpu.MPI_SIZE);
		RMSprop_with_nesterov_weight_update(MSGRAD[0],arrGRAD[0][gpu.MYRANK],W[1],M[1],0.9f,_learningRate/(float)arrGRAD[0][gpu.MYRANK]->rows,_nBatchSize*gpu.MPI_SIZE, MOMENTUM);
		while(gpu.get_queue_length() > (4*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD[1],gpu.MYRANK,gpu.MPI_SIZE,1.0f/(float)gpu.MPI_SIZE);
		RMSprop_with_nesterov_weight_update(MSGRAD[1],arrGRAD[1][gpu.MYRANK],W[1],M[1],0.9f,_learningRate/(float)arrGRAD[1][gpu.MYRANK]->rows,_nBatchSize*gpu.MPI_SIZE, MOMENTUM);
		while(gpu.get_queue_length() > (3*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD_B[2],gpu.MYRANK,gpu.MPI_SIZE,1.0f/(float)gpu.MPI_SIZE);
		RMSprop_with_nesterov_weight_update(MSBGRAD[2],arrGRAD_B[2][gpu.MYRANK],B[0],M_B[0],0.9f,_learningRate,_nBatchSize*gpu.MPI_SIZE, MOMENTUM);
		while(gpu.get_queue_length() > (2*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD_B[0],gpu.MYRANK,gpu.MPI_SIZE,1.0f/(float)gpu.MPI_SIZE);
		RMSprop_with_nesterov_weight_update(MSBGRAD[0],arrGRAD_B[0][gpu.MYRANK],B[1],M_B[1],0.9f,_learningRate,_nBatchSize*gpu.MPI_SIZE, MOMENTUM);
		while(gpu.get_queue_length() > (1*(gpu.MPI_SIZE-1))){ gpu.pop_queue(); }
		addGradientsN(arrGRAD_B[1],gpu.MYRANK,gpu.MPI_SIZE,1.0f/(float)gpu.MPI_SIZE);
		RMSprop_with_nesterov_weight_update(MSBGRAD[1],arrGRAD_B[1][gpu.MYRANK],B[1],M_B[1],0.9f,_learningRate,_nBatchSize*gpu.MPI_SIZE, MOMENTUM);
		while(gpu.get_queue_length() > 0){ gpu.pop_queue(); }
		addGradientsN(arrGRAD_B[3],gpu.MYRANK,gpu.MPI_SIZE,1.0f/(float)gpu.MPI_SIZE);
		RMSprop_with_nesterov_weight_update(MSBGRAD[3],arrGRAD_B[3][gpu.MYRANK],B[0],M_B[0],0.9f,_learningRate,_nBatchSize*gpu.MPI_SIZE, MOMENTUM);

		/*

		for(int i = 0; i < arrGRAD.size(); i++)
		{
			cout << "G: " << i << " " << sum(arrGRAD[i][gpu.MYRANK]) << endl;
			MPI_Barrier(MPI_COMM_WORLD);
		}

		for(int i = 0; i < arrGRAD_B.size(); i++)
		{
			cout << "B: " << i << " " << sum(arrGRAD_B[i][gpu.MYRANK]) << endl;
			MPI_Barrier(MPI_COMM_WORLD);
		}
		*/

	}


	MPI_Barrier(MPI_COMM_WORLD);

}

void WikiMaxoutNet::backprop()
{
	pairwise_ranking(z2_X,z2_Y, out);
	pairwise_ranking_derivative(z2_X,z2_Y, pairwise_grad);

	mul(out, pairwise_grad, e1);
	gpu.dotT(e1, W[1],e2_partial);

	gpu.dot(aB,e1,arrGRAD_B[0][gpu.MYRANK]);
	gpu.tick();
	gpu.Tdot(a1_Y,e1,arrGRAD[0][gpu.MYRANK]);

    logisticGrad(a1_Y,a1_Y);
	mul(e2_partial,a1_Y,e2);

    gpu.Tdot(_batchY,e2,arrGRAD[2][gpu.MYRANK]);
    gpu.dot(aB,e2,arrGRAD_B[2][gpu.MYRANK]);
    gpu.dotT(e2,W[0],arrGRAD[4][gpu.MYRANK]);



	while(gpu.get_queue_length() > 0){ gpu.pop_queue(); }

	vStackN(_currentBatchIdx_X,stackedBatchIdx_X,gpu.MPI_SIZE);
	vStackN(_currentBatchIdx_Y,stackedBatchIdx_Y,gpu.MPI_SIZE);

    gpu.add_to_queue(arrGRAD[4]);
    gpu.add_to_queue(arrGRAD[2]);


	scalarMul(pairwise_grad,-1.0f,pairwise_grad);
	gpu.pop_queue();
	mul(out, pairwise_grad, e1);
	gpu.pop_queue();

	gpu.dot(aB,e1,arrGRAD_B[1][gpu.MYRANK]);
	gpu.pop_queue();
	gpu.Tdot(a1_X,e1,arrGRAD[1][gpu.MYRANK]);
	gpu.pop_queue();
	gpu.dotT(e1, W[1],e2_partial);
	gpu.pop_queue();

	logisticGrad(a1_X,a1_X);
	gpu.pop_queue();
	mul(e2_partial,a1_X,e2);
	gpu.pop_queue();

    gpu.Tdot(_batchX,e2,arrGRAD[3][gpu.MYRANK]);
	gpu.pop_queue();
	gpu.dot(aB,e2,arrGRAD_B[3][gpu.MYRANK]);
	gpu.pop_queue();
    gpu.dotT(e2,W[0],arrGRAD[5][gpu.MYRANK]);

    gpu.add_to_queue(arrGRAD[5]);
    gpu.add_to_queue(arrGRAD[3]);


	gpu.add_to_queue(arrGRAD[0]);
	gpu.add_to_queue(arrGRAD[1]);

    gpu.add_to_queue(arrGRAD_B[2]);
	gpu.add_to_queue(arrGRAD_B[0]);
	gpu.add_to_queue(arrGRAD_B[1]);
	gpu.add_to_queue(arrGRAD_B[3]);

}

double WikiMaxoutNet::calculateError()
{
	//scalarMul(W[0],0.9,W[0]);
	allocateNextBatch(true);
	for(int i = 0; i < _nCVErrorLength; i+=gpu.MPI_SIZE)
	{

		feedforward();

		pairwise_ranking(z2_X,z2_Y, out);
		_dSumError += (double)sum(out);

		allocateNextBatch(true);
	}
	//scalarMul(W[0],1.1,W[0]);
	//size_t free, total;
	//cudaMemGetInfo(&free, &total);
	//cout << free << endl;
	//cout << "Free system memory: " << sysconf(_SC_PAGE_SIZE)*sysconf(_SC_PHYS_PAGES) << endl;


	double error = _dSumError/(double)(_nBatchSize*_nCVErrorLength/gpu.MPI_SIZE);
	_dSumError = 0.0;



	_nNextBatchNumber_CV = gpu.MYRANK*_nCVErrorLength;

	return error;
}




