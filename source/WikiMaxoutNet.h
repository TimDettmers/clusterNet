/*
 * WikiMaxoutNet.h
 *
 *  Created on: Jun 25, 2014
 *      Author: tim
 */

#ifndef WIKIMAXOUTNET_H_
#define WIKIMAXOUTNET_H_

#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <basicOps.cuh>
#include <batchAllocator.h>
#include <clusterNet.h>
#include <util.cuh>
#include <assert.h>
#include <exception>
#include <ctime>

class WikiMaxoutNet
{
public:
	WikiMaxoutNet(ClusterNet gpus);
	void run();

private:
	std::vector<Matrix*> _currentBatchIdx_X;
	std::vector<Matrix*> _currentBatchIdx_Y;
	std::vector<Matrix*> _nextBatchIdx;
	ClusterNet gpu;
	Matrix *_X;
	Matrix *_CV_X;
	std::vector<Matrix*>_Vocab;
	std::vector<Matrix*>_Vocab_grad;
	std::vector<Matrix*>_MSVocab_grad;
	std::vector<Matrix*>_MSVocab_grad_Y;
	std::vector<Matrix*> M_VocabX;
	std::vector<Matrix*> M_VocabY;
	std::vector<Matrix*> _Vocab_grad_idx;
	std::vector<Matrix*> _batchX;
	std::vector<Matrix*> _batchY;
	Matrix *stackedVocabGrad_X;
	Matrix *stackedVocabGrad_Y;
	Matrix *stackedBatchIdx_X;
	Matrix *stackedBatchIdx_Y;
	Matrix *CV_container;

	std::vector<Matrix*> out;
	std::vector<Matrix*> pairwise_grad;
	std::vector<Matrix*> e1;
	std::vector<Matrix*> aB;
	std::vector<Matrix*> e2_partial;
	std::vector<Matrix*> e2;



	int _nCurrentDataSet;
	int _nNextBatchNumber;
	int _nNextBatchNumber_CV;
	float _RMS_multiplier;
	int _nBatchSize;
	int _batches;
	std::vector<int> _layers;
	std::vector<Matrix**> W;
	std::vector<Matrix**> B;
	std::vector<Matrix**> M;
	std::vector<Matrix**> M_B;
	std::vector<Matrix**> arrGRAD;
	std::vector<Matrix**> MSGRAD;
	std::vector<Matrix**> arrGRAD_B;
	std::vector<Matrix**> MSBGRAD;
	clock_t start,stop;
	int GPU_COUNT;
	int CURRENT_GPU;

	std::vector<Matrix*> d0;
	std::vector<Matrix*> z1;
	std::vector<Matrix*> a1_Y;
	std::vector<Matrix*> a1_idx_Y;
	std::vector<Matrix*> a1_X;
	std::vector<Matrix*> a1_idx_X;
	std::vector<Matrix*> d1;
	std::vector<Matrix*> z2_X;
	std::vector<Matrix*> z2_Y;

	std::vector<cudaStream_t> _streamNextBatch;
	cudaStream_t *STREAMS;
	double _dSumError;
	int _nCVErrorPeriodicity;
	int _nCVErrorLength;
	int _nMaxoutSize;
	float MOMENTUM;
	float _learningRate;
	int _totalNumberOfBatches;

	bool useRMSProp;
	bool useMaxout;



	void loadNextDataSet();
	void allocateNextBatch(bool isCV);
	void feedforward();
	void nesterov();
	double calculateError();
	void backprop();
	void weightUpdates();

};


#endif /* WIKIMAXOUTNET_H_ */
