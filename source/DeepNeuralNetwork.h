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
	Linear = 4,
	Double_Rectified_Linear = 8
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

typedef enum WeightUpdateType_t
{
	RMSProp = 0,
	NoMomentum = 1
} WeightUpdateType_t;

class DeepNeuralNetwork
{
public:
	DeepNeuralNetwork(std::vector<int> lLayerSizes, Networktype_t net_type, ClusterNet gpus, BatchAllocator allocator, int categories);
	void train();
	Matrix* predict(Matrix *X);

	float LEARNING_RATE;
	float MOMENTUM;
	int EPOCHS;
	bool OUTPUT_IS_PROBABILITY;
	int TRANSITION_EPOCH;
    bool PRINT_MISSCLASSIFICATION;

private:
	Costfunction_t m_costFunction;
	std::vector<Matrix*> D;
	std::vector<Matrix*> D_B;
	std::vector<Matrix*> Z;
	std::vector<Matrix*> Z_B;
	std::vector<Matrix*> E;
	std::vector<float> lDropout;
	std::vector<Unittype_t> lUnits;
	BatchAllocator m_BA;
	std::vector<Matrix*> W;
	std::vector<Matrix*> B;
	std::vector<Matrix*> B_Activations;
	std::vector<Matrix*> B_Activations_CV;
	std::vector<Matrix*> M;
	std::vector<Matrix*> B_M;
	std::vector<Matrix*> GRAD;
	std::vector<Matrix*> B_GRAD;
	std::vector<Matrix*> MS;
	std::vector<Matrix*> B_MS;
	std::vector<int> m_lLayers;
	ClusterNet m_gpus;
	int m_output_dim;
	Networktype_t m_net_type;
	WeightUpdateType_t m_update_type;

	float missclassification_error;

	void init_network_layout(std::vector<int> lLayerSizes);
	void init_weights();
	void nesterov_updates();
	void feedforward(FeedForward_t ff);
	void backprop();
	void weight_updates();
	void free_variables();
	float get_errors(Batchtype_t batch_t);
	void cross_validation_error();
	void train_error();

	void activation_function(int layer, Matrix *A);
	void derivative_function(int layer, Matrix *A);

};

#endif
