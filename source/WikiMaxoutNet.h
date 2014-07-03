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
	Matrix *_currentBatchIdx_X;
	Matrix *_currentBatchIdx_Y;
	Matrix *_nextBatchIdx;
	ClusterNet gpu;
	Matrix *_X;
	Matrix *_CV_X;
	Matrix *_Vocab;
	Matrix *_Vocab_grad;
	Matrix *_MSVocab_grad;
	Matrix *_MSVocab_grad_Y;
	Matrix *M_VocabX;
	Matrix *M_VocabY;
	Matrix *_Vocab_grad_idx;
	Matrix *_batchX;
	Matrix *_batchY;
	Matrix *stackedVocabGrad_X;
	Matrix *stackedVocabGrad_Y;
	Matrix *stackedBatchIdx_X;
	Matrix *stackedBatchIdx_Y;

	Matrix *out;
	Matrix *pairwise_grad;
	Matrix *e1;
	Matrix *aB;
	Matrix *e2_partial;
	Matrix *e2;



	int _nCurrentDataSet;
	int _nNextBatchNumber;
	int _nNextBatchNumber_CV;
	float _RMS_multiplier;
	int _nBatchSize;
	int _batches;
	std::vector<int> _layers;
	std::vector<Matrix*> W;
	std::vector<Matrix*> B;
	std::vector<Matrix*> M;
	std::vector<Matrix*> M_B;
	std::vector<Matrix**> arrGRAD;
	std::vector<Matrix*> MSGRAD;
	std::vector<Matrix**> arrGRAD_B;
	std::vector<Matrix*> MSBGRAD;
	clock_t start,stop;

	Matrix *d0;
	Matrix *z1;
	Matrix *a1_Y;
	Matrix *a1_idx_Y;
	Matrix *a1_X;
	Matrix *a1_idx_X;
	Matrix *d1;
	Matrix *z2_X;
	Matrix *z2_Y;

	cudaStream_t _streamNextBatch;
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
