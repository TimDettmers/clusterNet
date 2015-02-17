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

typedef enum Networktype_t
{
	Classification = 0,
	Regression = 1
} Networktype_t;





class DeepNeuralNetwork
{
public:
	DeepNeuralNetwork(std::vector<int> lLayerSizes, Networktype_t net_type, ClusterNet gpus, BatchAllocator allocator, int categories);
	void train();
	Matrix* predict(Matrix *X);

	float LEARNING_RATE;
	float LEARNING_RATE_DECAY;
	float MOMENTUM;
	float RMSPROP_MOMENTUM;
	int EPOCHS;
	bool OUTPUT_IS_PROBABILITY;
	int TRANSITION_EPOCH;
    bool PRINT_MISSCLASSIFICATION;
    Unittype_t MAIN_UNIT;
	std::vector<float> DROPOUT;

	WeightUpdateType_t UPDATE_TYPE;

private:
	Costfunction_t m_costFunction;
	std::vector<Matrix*> D;
	std::vector<Matrix*> D_B;
	std::vector<Matrix*> Z;
	std::vector<Matrix*> Z_B;
	std::vector<Matrix*> E;
	std::vector<Unittype_t> lUnits;
	BatchAllocator m_BA;
	std::vector<Matrix*> W;
	std::vector<float> max_values;
	std::vector<Matrix*> B;
	std::vector<Matrix*> B_Activations;
	std::vector<Matrix*> B_Activations_CV;
	std::vector<Matrix*> M;
	std::vector<Matrix*> B_M;
	std::vector<Matrix*> GRAD;
	std::vector<Matrix*> GRAD8bit;
	std::vector<Matrix*> GRAD_approx;
	std::vector<Matrix*> B_GRAD;
	std::vector<Matrix*> MS;
	std::vector<Matrix*> B_MS;
	std::vector<float> train_history;
	std::vector<float> cv_history;
	std::vector<int> m_lLayers;

	ClusterNet m_gpus;
	int m_output_dim;
	Networktype_t m_net_type;

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

	void save_history();

	void activation_function(int layer, Matrix *A);
	void derivative_function(int layer, Matrix *A);

};

#endif
