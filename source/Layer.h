#ifndef Layer_H
#define Layer_H
#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <basicOps.cuh>
#include <clusterNet.h>

class Layer
{
public:
	Matrix *w_grad_next;
	Matrix *b_grad_next;
	Layer *next;
	Layer *prev;
	Matrix *w_next;
	Matrix *b_next;

	Matrix *w_rms_next;
	Matrix *b_rms_next;

	Matrix *bias_activations;
	Matrix *out;
	Matrix *error;

	Matrix *out_offsize;
	Matrix *error_offsize;
	Matrix *bias_activations_offsize;
	Matrix *target_matrix_offsize;

	Matrix *target;
	Matrix *target_matrix;


	ClusterNet *GPU;

	float LEARNING_RATE;
	float MOMENTUM;
	float RMSPROP_MOMENTUM;
	float RUNNING_ERROR;
	float RUNNING_SAMPLE_SIZE;
    Unittype_t UNIT_TYPE;
	Costfunction_t COST;
	float DROPOUT;
	int UNITCOUNT;
	int BATCH_SIZE;

	WeightUpdateType_t UPDATE_TYPE;

	virtual void forward();
	virtual void running_error();
	virtual void backward();
	virtual void print_error(std::string message);
	virtual void weight_update();

	virtual void link_with_next_layer(Layer *next_layer);
	virtual void init(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu);
	virtual ~Layer();
	Layer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu);
	Layer(int unitcount, Unittype_t unit);
	Layer(int unitcount);

	Layer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet *gpu);
	Layer(int unitcount, Unittype_t unit, Layer *prev);
	Layer(int unitcount, Layer *prev);

private:
	virtual void activation(Matrix *input);
	virtual void activation_gradient();
	void handle_offsize();


};

#endif
