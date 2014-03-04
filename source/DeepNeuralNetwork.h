#ifndef DeepNeuralNetwork_H
#define DeepNeuralNetwork_H
#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <basicOps.cuh>
#include <batchAllocator.h>
#include <clusterNet.h>

class DeepNeuralNetwork
{
public:
	DeepNeuralNetwork(Matrix *X, Matrix *y, float cv_size, std::vector<int> lLayersSizes);
	void train();

	float LEARNING_RATE;
	float MOMENTUM;
	std::vector<float> lDropout;
	std::vector<int> lLayers;
private:
	BatchAllocator m_BA;
	ClusterNet m_gpus;
	std::vector<Matrix*> W;
	std::vector<Matrix*> M;
	std::vector<Matrix*> GRAD;
	std::vector<Matrix*> MS;
	std::vector<Matrix*> D;
	std::vector<Matrix*> Z;
	std::vector<Matrix*> E;

	void init_weights();

};

#endif
