#include <Layer.h>
#include <basicOps.cuh>



using std::cout;
using std::endl;
using std::string;
using std::vector;

Layer::Layer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu){ init(unitcount, start_batch_size,unit,gpu); }
Layer::Layer(int unitcount, Unittype_t unit){ init(unitcount, 0,unit, NULL); }
Layer::Layer(int unitcount){ init(unitcount, 0,Rectified_Linear, NULL); }

Layer::Layer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet *gpu)
{ init(unitcount, start_batch_size,unit,gpu); prev->link_with_next_layer(this); }
Layer::Layer(int unitcount, Unittype_t unit, Layer *prev){ init(unitcount, 0,unit, prev->GPU); prev->link_with_next_layer(this); }
Layer::Layer(int unitcount, Layer *prev){ init(unitcount, 0,Rectified_Linear, NULL); prev->link_with_next_layer(this); }

void Layer::init(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu)
{

	next = NULL;
	prev = NULL;
	w_next = NULL;
	b_next = NULL;
	b_next_sync = NULL;
	w_rms_next = NULL;
	b_rms_next = NULL;
	b_grad_next = NULL;

	w_next_sync_send = NULL;
	b_next_sync_send = NULL;
	w_next_sync_recv = NULL;
	b_next_sync_recv = NULL;

	isSynchronizing = false;

	compression = bits_32;

	target = NULL;
	target_matrix = NULL;
	error = NULL;

	LEARNING_RATE = 0.003;
	RMSPROP_MOMENTUM = 0.9f;
	UNIT_TYPE = unit;
	DROPOUT = 0.5f;
	UNITCOUNT = unitcount;
	BATCH_SIZE = start_batch_size;
	RUNNING_ERROR = 0.0f;
	RUNNING_SAMPLE_SIZE = 0.0f;
	L2 = 15.0f;

	MAX_GRAD_VALUE = 1.0f;

	UPDATE_TYPE = RMSProp;
	COST = Misclassification;
	PARALLELISM = None;

	GPU = gpu;

	for(int i = 0; i < GPU->MPI_SIZE; i++)
	{

		send_request.push_back(new MPI_Request);
		recv_request.push_back(new MPI_Request);
	}

	max_grad_value_sync = (float*)malloc(GPU->MPI_SIZE*sizeof(float));

	if(BATCH_SIZE > 0)
	{
		out = zeros(BATCH_SIZE, UNITCOUNT);
		bias_activations = ones(1, BATCH_SIZE);
		activation = zeros(BATCH_SIZE, UNITCOUNT);
	}
	else
	{
		out = NULL;
		bias_activations = NULL;
		activation = NULL;
	}


	mpi_buffer = (float*)malloc(GPU->MPI_SIZE*sizeof(float));
}

void Layer::link_with_next_layer(Layer *next_layer)
{
	next = next_layer;
	if(next->BATCH_SIZE == 0){ next->BATCH_SIZE = BATCH_SIZE; }
	if(!next->GPU){next->GPU = GPU;}

	w_rms_next = zeros(UNITCOUNT,next_layer->UNITCOUNT);
	if(PARALLELISM == DataParallelism)
	{
		for(int i = 0; i < GPU->MPI_SIZE; i++)
		{
			vec_w_grad_next.push_back(zeros(UNITCOUNT,next_layer->UNITCOUNT));
			vec_w_grad_next_8bit.push_back(empty_char(UNITCOUNT,next_layer->UNITCOUNT));
		}

		w_next_sync_send = empty_char(UNITCOUNT,next_layer->UNITCOUNT);
		w_next_sync_recv = empty_char(UNITCOUNT,next_layer->UNITCOUNT);
		b_next_sync = zeros(1,next_layer->UNITCOUNT);
		b_next_sync_send = empty_char(1,next_layer->UNITCOUNT);
		b_next_sync_recv = empty_char(1,next_layer->UNITCOUNT);
	}

	if(PARALLELISM == ModelParallelism)
	{
		for(int i = 0; i < GPU->MPI_SIZE; i++)
		{
			vec_w_grad_next.push_back(GPU->distributed_zeros(UNITCOUNT,next_layer->UNITCOUNT));
		}

		Matrix *w = GPU->distributed_uniformSqrtWeight(UNITCOUNT,next_layer->UNITCOUNT);
		w_next = w;
		b_grad_next = GPU->distributed_zeros(1,next_layer->UNITCOUNT);
		b_rms_next = GPU->distributed_zeros(1,next_layer->UNITCOUNT);

	}
	else
	{
		Matrix *w = GPU->uniformSqrtWeight(UNITCOUNT,next_layer->UNITCOUNT);
		w_next = w;
		b_grad_next = zeros(1,next_layer->UNITCOUNT);
		b_rms_next = zeros(1,next_layer->UNITCOUNT);

	}


	Matrix *b = zeros(1,next_layer->UNITCOUNT);
	b_next = b;
	next->out = zeros(BATCH_SIZE, next->UNITCOUNT);
	next->activation = zeros(BATCH_SIZE, next->UNITCOUNT);
	next->error = zeros(BATCH_SIZE, next->UNITCOUNT);
	next->bias_activations = ones(1, BATCH_SIZE);
	next->prev = this;


}


void Layer::unit_activation(){ unit_activation(true); }
void Layer::unit_activation(bool useDropout)
{
	switch(UNIT_TYPE)
	{
		case Logistic:
			logistic(out,activation);
			break;
		case Rectified_Linear:
			rectified_linear(out,activation);
			break;
		case Softmax:
			softmax(out,out);
			break;
		case Double_Rectified_Linear:
			doubleRectifiedLinear(out,activation);
			break;
		case Linear:
			LinearUnit(out, activation);
			break;
		case Input:
			break;
	}

	if(UNIT_TYPE != Softmax)
	{
		if(useDropout)
			GPU->dropout(activation,out,DROPOUT);
		else
			scalarMul(activation,1.0f-DROPOUT, out);
	}

}

void Layer::activation_gradient()
{

	switch(UNIT_TYPE)
	{
		case Logistic:
			logisticGrad(activation,out);
			break;
		case Rectified_Linear:
			rectified_linear_derivative(activation,out);
			break;
		case Double_Rectified_Linear:
			double_rectified_linear_derivative(activation,out);
			break;
		case Softmax:
			break;
		default:
			throw "Unknown unit";
			break;
	}

}

void Layer::handle_offsize()
{
	if(!prev)
	{
		if(!out){ out = empty(activation->rows, activation->cols); }
		else if(out->rows != activation->rows)
		{
			cudaFree(out->data);
			free(out);
			out = empty(activation->rows, activation->cols);
		}
	}
	else
	{
		if(prev->out->rows != out->rows && (!out_offsize || out_offsize->rows != prev->out->rows))
		{
			if(out_offsize)
			{
				cudaFree(out_offsize->data);
				cudaFree(activation_offsize->data);
				cudaFree(error_offsize->data);
				cudaFree(bias_activations_offsize->data);
				cudaFree(target_matrix_offsize->data);
			}

			out_offsize = empty(prev->out->rows, UNITCOUNT);
			activation_offsize = empty(prev->out->rows, UNITCOUNT);
			error_offsize = empty(prev->out->rows, UNITCOUNT);
			bias_activations_offsize = empty(1,prev->out->rows);
			target_matrix_offsize = zeros(prev->out->rows, UNITCOUNT);
		}


		if(prev->out->rows != out->rows)
		{
			Matrix *swap;
			swap = out; out = out_offsize; out_offsize = swap;
			swap = activation; activation = activation_offsize; activation_offsize = swap;
			swap = error; error = error_offsize; error_offsize = swap;
			swap = bias_activations; bias_activations = bias_activations_offsize; bias_activations_offsize = swap;
			swap = target_matrix; target_matrix = target_matrix_offsize; target_matrix_offsize = swap;
		}
	}

}

void Layer::forward(){ forward(true); }
void Layer::forward(bool useDropout)
{
	handle_offsize();
	if(!prev){  unit_activation(useDropout); next->forward(useDropout); return; }
	if(PARALLELISM == DataParallelism && useDropout){ prev->wait_for_synchronization(); prev->weight_update(); }

	GPU->dot(prev->out,prev->w_next,out);
	addMatrixVector(out,prev->b_next,out);
    unit_activation(useDropout);

    if(next){ next->forward(useDropout); }
}


void Layer::running_error(bool isCV, int epoch)
{
	if(!target){ next->running_error(isCV, epoch); return;}

	string text = "";

	Matrix *result;
	Matrix *eq;
	float sum_value = 0.0f;
	float size = 0.0f;

	if (!Train_errors.count(epoch))
	{
		Train_errors[epoch] = std::vector<float>();
		CV_errors[epoch] = std::vector<float>();
	}





	switch(COST)
	{
		case Misclassification:
			result = argmax(out);
			eq = equal(result,target);
			sum_value = sum(eq);
			sum_value = reduce_to_sum_root(sum_value);
			size = reduce_to_sum_root(out->rows);
			if(GPU->MYRANK == 0)
			{
				if(isCV)
					CV_errors[epoch].push_back(sum_value/size);
				else
					Train_errors[epoch].push_back(sum_value/size);
			}
			RUNNING_ERROR += (out->rows  - sum_value);
			RUNNING_SAMPLE_SIZE += out->rows;
			break;
		default:
			throw "Unknown cost function!";
			break;
	}

	cudaFree(result->data);
	cudaFree(eq->data);
}



void Layer::backward_errors()
{
	if(!target){ next->backward_errors(); }
	if(target)
	{
		if(out->cols != target->cols && !target_matrix){ target_matrix = zeros(BATCH_SIZE,out->cols); }
		if(out->cols != target->cols){ create_t_matrix(target,target_matrix); sub(out,target_matrix,error); return; }
		else{ sub(activation,target,error);  return;}
	}

	if(UNIT_TYPE == Input){ backward_grads(); return; }

	activation_gradient();
	GPU->dotT(next->error, w_next,error);
	mul(error, out, error);

}

void Layer::backward_grads()
{
	GPU->Tdot(activation, next->error, vec_w_grad_next[GPU->MYRANK]);
	MPI_synchronization_async();
	if(!next->target){ next->backward_grads(); }
	//GPU->dot(next->bias_activations, next->error,b_grad_next);

}

void Layer::MPI_synchronization_async()
{
	if(PARALLELISM != DataParallelism){ return; }

	int target = GPU->MYRANK +1 == GPU->MPI_SIZE ? 0 : GPU->MYRANK+1;
	int source = GPU->MYRANK-1 == -1 ? GPU->MPI_SIZE-1 : GPU->MYRANK-1;


	if(compression == bits_8)
	{

		//cout << 1.0f/((float)out->rows) << endl;
		scalarMul(vec_w_grad_next[GPU->MYRANK],1.0f/((float)out->rows),vec_w_grad_next[GPU->MYRANK]);

		abs(vec_w_grad_next[GPU->MYRANK],vec_w_grad_next[target]);


		MAX_GRAD_VALUE = max(vec_w_grad_next[target]);
		//MAX_GRAD_VALUE = 0.003f;
		MPI_Allgather(&MAX_GRAD_VALUE, 1, MPI_FLOAT, max_grad_value_sync, 1, MPI_FLOAT, MPI_COMM_WORLD);

		GPU->compression_8bit(vec_w_grad_next[GPU->MYRANK],MAX_GRAD_VALUE,vec_w_grad_next_8bit[GPU->MYRANK]);
		for (int i = 0; i < GPU->MPI_SIZE - 1; i++)
		{
			MPI_Isend(vec_w_grad_next_8bit[GPU->MYRANK]->char_data,vec_w_grad_next_8bit[GPU->MYRANK]->size,MPI_CHAR,target,999,MPI_COMM_WORLD, send_request[target]);
			MPI_Irecv(vec_w_grad_next_8bit[source]->char_data,vec_w_grad_next_8bit[source]->size,MPI_CHAR,source,999,MPI_COMM_WORLD,recv_request[source]);
			target = target +1 == GPU->MPI_SIZE ? 0 : target+1;
			source = source-1 == -1 ? GPU->MPI_SIZE-1 : source-1;
		}
	}
	else
	{
		for (int i = 0; i < GPU->MPI_SIZE - 1; i++)
		{
			MPI_Isend(vec_w_grad_next[GPU->MYRANK]->data,vec_w_grad_next[GPU->MYRANK]->size,MPI_FLOAT,target,999,MPI_COMM_WORLD, send_request[target]);
			MPI_Irecv(vec_w_grad_next[source]->data,vec_w_grad_next[source]->size,MPI_FLOAT,source,999,MPI_COMM_WORLD,recv_request[source]);
			target = target +1 == GPU->MPI_SIZE ? 0 : target+1;
			source = source-1 == -1 ? GPU->MPI_SIZE-1 : source-1;

		}
	}
	isSynchronizing = true;




}

void Layer::wait_for_synchronization()
{
	if(target){ return; }
	if(!isSynchronizing){ return; }
	if(PARALLELISM != DataParallelism){ return; }
	//GPU->tick();
	//MPI_Wait(next->send_request,MPI_STATUS_IGNORE);_w_next_sync

	for(int i = 0; i < GPU->MPI_SIZE; i++)
	{
		if(i== GPU->MYRANK){ continue; }
		MPI_Wait(send_request[i],MPI_STATUS_IGNORE);
		MPI_Wait(recv_request[i],MPI_STATUS_IGNORE);
	}

	//float secs = GPU->tock()/1000.0f;
	//cout << w_next_sync->bytes/1024./1024./1024./secs << " GB/s" << endl;
	//printdim(w_next_sync);
	//cout << "pre decomrpess" << endl;
	//GPU->decompression_8bit(w_next_sync_recv,0.001,w_next_sync);
	//cout << "post decompress" << endl;




	/*
	MPI_Barrier(MPI_COMM_WORLD);
	cout << GPU->MYRANK << " " << sum(vec_w_grad_next[0]) << " 0" << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	cout << GPU->MYRANK << " " << sum(vec_w_grad_next[1]) << " 1" << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	cout << GPU->MYRANK << " " << sum(vec_w_grad_next[2]) << " 2" << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	cout << GPU->MYRANK << " " << sum(vec_w_grad_next[3]) << " 3" << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	*/


	for(int i = 0; i < GPU->MPI_SIZE; i++)
	{
		if(i == GPU->MYRANK){ continue; }
		if(compression == bits_8){ GPU->decompression_8bit(vec_w_grad_next_8bit[i],max_grad_value_sync[i],vec_w_grad_next[i]); }
		add(vec_w_grad_next[GPU->MYRANK],vec_w_grad_next[i],vec_w_grad_next[GPU->MYRANK]);
	}
	isSynchronizing = false;
}

void Layer::weight_update()
{
	if(target){ return; }

	//next->weight_update();
	float *data = (float*)malloc(sizeof(float)*100);

	switch(UPDATE_TYPE)
	{
		case RMSProp:

			CUDA_CHECK_RETURN(cudaMemcpy(data,vec_w_grad_next[GPU->MYRANK]->data,10*sizeof(float),cudaMemcpyDefault));
			cout << "pre print" << endl;

			for(int i; i < 100; i++){ cout << data[i]  << endl;}
			RMSprop_with_weight_update(w_rms_next,vec_w_grad_next[GPU->MYRANK],w_next,w_next,RMSPROP_MOMENTUM,LEARNING_RATE,out->rows*GPU->MPI_SIZE,MOMENTUM);
			cout << "post print" << endl;
			//RMSprop_with_weight_update(b_rms_next,b_grad_next,b_next,b_next,RMSPROP_MOMENTUM,LEARNING_RATE/100.0f,out->rows,MOMENTUM);
			//scalarMul(b_grad_next, LEARNING_RATE/float(out->rows*GPU->MPI_SIZE) ,b_grad_next);
			//sub(b_next,b_grad_next,b_next);

			break;
		default:
			throw "Unknown update type!";
			break;
	}
	free(data);

	//limit_magnitude();

}

void Layer::limit_magnitude()
{

	square(w_next,vec_w_grad_next[GPU->MYRANK]);
	Matrix *temp = ones(vec_w_grad_next[GPU->MYRANK]->cols,1);
	Matrix *sums = GPU->dot(vec_w_grad_next[GPU->MYRANK],temp);
	renormalizeWeights(w_next,sums,L2);
	cudaFree(temp->data);
	cudaFree(sums->data);
	free(temp);
	free(sums);

}

void Layer::print_error(string message)
{
	if(!target){ next->print_error(message); return;}

	if(GPU->MPI_SIZE > 1)
	{
		RUNNING_ERROR =reduce_to_sum_root(RUNNING_ERROR);
		RUNNING_SAMPLE_SIZE = reduce_to_sum_root(RUNNING_SAMPLE_SIZE);
	}

	if(GPU->MYRANK == 0)
		cout << message << RUNNING_ERROR/RUNNING_SAMPLE_SIZE << endl;

	RUNNING_ERROR = 0.0f;
	RUNNING_SAMPLE_SIZE = 0.0f;
}


float Layer::reduce_to_sum_root(float value)
{

	MPI_Gather(&value, 1, MPI_FLOAT, mpi_buffer, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		for(int i = 1; i < GPU->MPI_SIZE; i++)
			mpi_buffer[0] += mpi_buffer[i];


	return mpi_buffer[0];

}

void Layer::set_hidden_dropout(float dropout)
{
	if(!next){ return; }
	next->DROPOUT = dropout;
	next->set_hidden_dropout(dropout);
}

void Layer::learning_rate_decay(float decay_rate)
{
	if(!next){ return; }
	next->LEARNING_RATE *= decay_rate;
	next->learning_rate_decay(decay_rate);
}

void Layer::dropout_decay()
{
	if(!prev){ cout << "Decaying dropout!" << endl; }
	if(!next){ return;}

	cout << "Setting dropout from " << DROPOUT << " to " << DROPOUT/2.0f << endl;
	DROPOUT /= 2.0f;
	next->dropout_decay();
}

Layer::~Layer()
{
	cout << "destruct" << endl;
}


