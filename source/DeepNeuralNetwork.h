#ifndef DeepNeuralNetwork_H
#define DeepNeuralNetwork_H
#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <basicOps.cuh>
#include <batchAllocator.h>
#include <clusterNet.h>

typedef enum FeedForward_t
{
	Dropout = 0,
	Train_error = 1,
	CV_error = 2
} FeedForward_t;

typedef enum Batchtype_t
{
	Train = 0,
	CV = 1
} Batchtype_t;

typedef enum Unittype_t
{
	Logistic = 0,
	Rectified_Linear = 1,
	Softmax = 2,
	Linear = 4
} Unittype_t;

typedef enum Networktype_t
{
	Classification = 0,
	Regression = 1
} Networktype_t;

typedef enum Costfunction_t
{
	Cross_Entropy = 0,
	Squared_Error = 1,
	Root_Squared_Error = 2
} Costfunction_t;

class DeepNeuralNetwork
{
public:
	DeepNeuralNetwork(Matrix *X, Matrix *y, float cv_size, std::vector<int> lLayersSizes, Networktype_t net_type);
	DeepNeuralNetwork(Matrix *X, Matrix *y, float cv_size, std::vector<int> lLayerSizes, Networktype_t net_type, int argc, char *argv[]);
	void train();

	float LEARNING_RATE;
	float MOMENTUM;

private:
	BatchAllocator m_BA;
	ClusterNet m_gpus;
	Costfunction_t m_costFunction;
	std::vector<Matrix*> W;
	std::vector<Matrix*> M;
	std::vector<Matrix*> GRAD;
	std::vector<Matrix*> MS;
	std::vector<Matrix*> D;
	std::vector<Matrix*> Z;
	std::vector<Matrix*> E;

	std::vector<float> lDropout;
	std::vector<int> lLayers;
	std::vector<Unittype_t> lUnits;

	void init_weights();
	void nesterov_updates();
	void feedforward(FeedForward_t ff);
	void backprop();
	void weight_updates();
	void free_variables();
	int get_classification_errors(Batchtype_t batch_t);
	void cross_validation_error();
	void train_error();

	void activation_function(int layer, Matrix *A);
	void derivative_function(int layer, Matrix *A);
	void init(Matrix *X, Matrix *y, float cv_size, std::vector<int> lLayerSizes, Networktype_t net_type, int argc, char *argv[]);

};

#endif
