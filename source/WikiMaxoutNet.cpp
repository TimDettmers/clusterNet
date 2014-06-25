#include <WikiMaxoutNet.h>
using std::cout;
using std::endl;


WikiMaxoutNet::WikiMaxoutNet(ClusterNet gpu)
{
	int nWordVectorDim = 50;
	_b = BatchAllocator();
	_gpu = gpu;
	_nCurrentDataSet = _gpu.MYRANK;
	_X = gpu.rand(1,1);
	_nNextBatchNumber = 0;
	_nBatchSize = 2;
	cudaStreamCreate(&_streamNextBatch);

	_Vocab = gpu.uniformSqrtWeight(nWordVectorDim,300000);
	_batchX = empty(_nBatchSize, nWordVectorDim*11);
	_batchY = empty(_nBatchSize, nWordVectorDim*11);
	_currentBatchIdx = empty(_nBatchSize,11);
	_nextBatchIdx = empty(_nBatchSize,11);

	loadNextDataSet();
	allocateNextBatch();
}
void WikiMaxoutNet::run()
{
	allocateNextBatch();
	printmat(_currentBatchIdx);
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
		to_col_major(_nextBatchIdx, _currentBatchIdx);
		_gpu.construct_vocab_matrix(_currentBatchIdx, _batchX, _batchY, _Vocab);
	}

	cudaMemcpyAsync(_nextBatchIdx->data,&_X->data[_nBatchSize*11*sizeof(float)*_nNextBatchNumber],
					_nBatchSize*11*sizeof(float),
					cudaMemcpyHostToDevice,_streamNextBatch);

	_nNextBatchNumber+=1;
}

