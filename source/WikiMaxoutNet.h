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
	BatchAllocator _b;
	Matrix *_currentBatchIdx;
	Matrix *_nextBatchIdx;
	ClusterNet _gpu;
	Matrix *_X;
	Matrix *_Vocab;
	Matrix *_batchX;
	Matrix *_batchY;
	int _nCurrentDataSet;
	int _nNextBatchNumber;
	int _nBatchSize;

	cudaStream_t _streamNextBatch;


	void loadNextDataSet();
	void allocateNextBatch();
};


#endif /* WIKIMAXOUTNET_H_ */
