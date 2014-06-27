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

class WikiMaxoutNet
{
public:
	WikiMaxoutNet(ClusterNet gpu);
	void run();

private:
	Matrix *_currentBatchIdx_X;
	Matrix *_currentBatchIdx_Y;
	Matrix *_nextBatchIdx;
	ClusterNet _gpu;
	Matrix *_X;
	Matrix *_Vocab;
	Matrix *_W1;
	Matrix *_W2;
	Matrix *_B1;
	Matrix *_B2;
	Matrix *_grad1_X;
	Matrix *_grad2_X;
	Matrix *_grad1_Y;
	Matrix *_grad2_Y;
	Matrix *_grad3_X;
	Matrix *_grad3_Y;
	Matrix *_Vocab_grad;
	Matrix *_Vocab_grad_idx;
	Matrix *_gradB1;
	Matrix *_gradB2;
	Matrix *_batchX;
	Matrix *_batchY;
	int _nCurrentDataSet;
	int _nNextBatchNumber;
	int _nBatchSize;


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
	int _nTrainErrorPeriodicity;
	int _nTrainErrorLength;


	void loadNextDataSet();
	void allocateNextBatch();
	void feedforward();
	void calculateError();
};


#endif /* WIKIMAXOUTNET_H_ */
