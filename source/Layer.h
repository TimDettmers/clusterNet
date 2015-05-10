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
	Matrix *b_grad_next;
	Layer *next;
	Layer *prev;
	Matrix *w_next;
	Matrix *b_next;

	std::vector<Matrix* > vec_w_grad_next;
	std::vector<Matrix* > vec_w_grad_next_8bit;
	Matrix *b_next_sync;
	Matrix *w_next_sync_recv;
	Matrix *b_next_sync_recv;
	Matrix *w_next_sync_send;
	Matrix *b_next_sync_send;

	Matrix *w_rms_next;
	Matrix *b_rms_next;

	Matrix *bias_activations;
	Matrix *out;
	Matrix *error;
	Matrix *activation;

	Matrix *out_offsize;
	Matrix *activation_offsize;
	Matrix *error_offsize;
	Matrix *bias_activations_offsize;
	Matrix *target_matrix_offsize;

	Matrix *target;
	Matrix *target_matrix;

	std::vector<MPI_Request* > send_request;
	std::vector<MPI_Request* > recv_request;

	std::vector<float> CV_errors;
	std::vector<float> Train_errors;

	ClusterNet *GPU;

	float LEARNING_RATE;
	float MOMENTUM;
	float RMSPROP_MOMENTUM;
	float RUNNING_ERROR;
	float RUNNING_SAMPLE_SIZE;
	float L2;
    Unittype_t UNIT_TYPE;
	Costfunction_t COST;
	float DROPOUT;
	int UNITCOUNT;
	int BATCH_SIZE;

	bool isSynchronizing;

	float MAX_GRAD_VALUE;
	float *max_grad_value_sync;

	Compression_t compression;

	WeightUpdateType_t UPDATE_TYPE;

	ParallelismType_t PARALLELISM;

	virtual ~Layer();
	Layer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu);
	Layer(int unitcount, Unittype_t unit);
	Layer(int unitcount);

	Layer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet *gpu);
	Layer(int unitcount, Unittype_t unit, Layer *prev);
	Layer(int unitcount, Layer *prev);

	virtual void forward();
	virtual void forward(bool useDropout);
	virtual void running_error();
	virtual void backward_errors();
	virtual void backward_grads();
	virtual void print_error(std::string message);
	virtual void weight_update();

	virtual void MPI_synchronization_async();
	virtual void wait_for_synchronization();

	virtual void limit_magnitude();

	virtual void link_with_next_layer(Layer *next_layer);
	virtual void init(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu);
	virtual void set_hidden_dropout(float dropout);

	virtual void dropout_decay();
	virtual void learning_rate_decay(float decay_rate);



private:
	virtual void unit_activation();
	virtual void unit_activation(bool useDropout);
	virtual void activation_gradient();
	void handle_offsize();


};

#endif
