#ifndef clusterKernels
#define clusterKernels

#include "curand.h"
#include "curand_kernel.h"
__global__ void kGetNonZeroColumns(float *A, float *out, int rows, int cols);
__global__ void kGetNonZeroElements(float *A, float *out, int size);
__global__ void kFill_with(float *m, float fill_value, int size);
__global__ void kFill_with(int *m, int fill_value, int size);
__global__ void kAdd(float *A,float *B, float *out, int size);
__global__ void kSub(float *A,float *B, float *out, int size);
__global__ void kSub_Sparse(float *A, float *data, int *ptr_rows, int *idx_cols, float *out, int rows, int cols, int size);
__global__ void kMul(float *A,float *B, float *out, int size);
__global__ void kDiv(float *A,float *B, float *out, int size);
__global__ void kExp(float *A, float *out, int size);
__global__ void kLog(float *A, float *out, int size);
__global__ void kSqrt(float *A, float *out, int size);
__global__ void kSquare(float *A, float *out, int size);
__global__ void kScalarMul(float *A, float scalar, float *out, int size);
__global__ void kScalarAdd(float *A, float scalar, float *out, int size);
__global__ void kTranspose(float *A, float *out, int width, int height); 
__global__ void setup_kernel(curandState *state, int seed);
__global__ void generate_uniform_kernel(curandState *state, int size, float *out);
__global__ void generate_normal_kernel(curandState *state, int size, float *out);
__global__ void slice_rows(float *A, float *out, int size_out, int rows_A, int start, int end);
__global__ void slice_cols(float *A, float *out, int start, int rows, int size_out);
__global__ void vStack(float *A, float *B, float *out, int size_out, int rows_a, int rows, int cols);
__global__ void hStack(float *A, float *B, float *out, int size_out, int size_a);
__global__ void hStackN(float **arrA, int general_size, float *out, int size_out, int matrices_count);
__global__ void hStackN(Matrix **arrA, int general_size, float *out, int size_out, int matrices_count);
__global__ void vStackN(float **arrA, float *out, int rows, int cols);
__global__ void kSoftMax(float* A, float* out, unsigned int rows, unsigned int cols);
__device__ void reduceToMax(float* sdata, unsigned int tid);
__device__ void reduceToSumLocal(float* sdata, unsigned int tid);
__global__ void kSubMatrixVector(float *A, float *v, float *out, int rows, int size);
__global__ void kAddMatrixVector(float *A, float *v, float *out, int rows, int size);
__global__ void kArgmax(float* A, float* out, unsigned int height, unsigned int width);
__global__ void kCreate_t_matrix(float *labels, float *out, int rows, int size);
__global__ void kEqual(float *A, float *B, float *out, int size);
__global__ void kSum(float *v, float *out, int size);
__global__ void kLogistic(float *A, float *out, int size);
__global__ void kLogisticGrad(float *A, float *out, int size);
__global__ void kArange(float *out, int start, int rows, int cols, int size);
__global__ void kDropout(float *A, float *rdm, float dropout, int size);
__global__ void kRMSprop(float *RMS, float *grad, float RMS_multiplier, float learning_rate, int batch_size, int size);
__global__ void kRMSprop_with_nesterov_weight_update(float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum);
__global__ void kRMSprop_with_momentum_weight_update(float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum);
__global__ void kRMSprop_with_weight_update(float *RMS, float *grad, float *w, float RMS_multiplier, float learning_rate, int batch_size, int size);
__global__ void kCreateRdmSqrtWeight_Logistic(float *A, int in, int out, int size);
__global__ void kRandInt(float *A, int lower_limit, int upper_limit, int size);
__global__ void kCreateSparseRdmWeight(float *rdm, float* indicies, float *out, int rows, int cols, int connections);
__global__ void kRectifiedLinear(float *A, float *out, int size);
__global__ void kRectifiedLinear_Derivative(float *A, float *out, int size);
__global__ void kSquaredError(float *A, float *t, float *out, int size);
__global__ void kDoubleRectifiedLinear(float* A, float* out, int size);
__global__ void kDoubleRectifiedLinear_Derivative(float *A, float *out, int size);
__global__ void kSparseDot(int m, int n, int k, float *data, int* indptr, int* indices, float *dense_data, float* target, float beta, float alpha);
__global__ void kPrintData(float *A, int size);
__global__ void kHardTanH(float *A, float *out, int size);
__global__ void kHardTanH_Derivative(float *A, float *out, int size);
__global__ void kPairwise_ranking(float *A, float *B, float *out, int size);
__global__ void kPairwise_ranking_derivative(float *A, float *B, float *out, int size);
__global__ void kMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height);
__global__ void kMaxout(float *A, float *out, float *outargmax, int maxout_level, unsigned int cols, unsigned int rows);
__device__ void reduceToMaxAndArgMax(float* sdataMax, float* sdataArgMax, unsigned int tid, int threads);
__global__ void kExpandToMaxoutGrad(float* error, float* indexes, float *out, int error_size, int error_rows, int maxout_level);
__global__ void kConstructVocabMatrix(float *vocab_idx, float *vocab_idx_y, float* vocab, float *rdm_idx, float *batch_X, float *batch_Y);
__global__ void kExpandDoubleVocabGradient(float *gradX, float *gradY, float *vocab_idx_X, float *vocab_idx_Y, float* vocab,
										 float *vocab_grad, float *vocab_grad_idx, float learning_rate, int grad_size);
__global__ void kExpandVocabGradient(float *grad, float *vocab_idx, float *vocab_grad);
__global__ void kUpdateVocabWithGradient(float *grad, float *vocab_idx, float* vocab, float learning_rate);
__global__ void kUpdateVocabWithGradient_LearningRateMatrix(float *grad, float *vocab_idx, float* vocab, float *learning_rate);
__global__ void kNesterovVocabUpdate(float *M, float *vocab_idx, float* vocab, float* vocab_idx_placeholder, float momentum, int vocab_cols);
__global__ void kRMSpropVocab_with_nesterov_weight_update(float *RMS, float *grad, float *M, float *vocab_idx, float* vocab, float* vocab_idx_placeholder,
														  float RMS_multiplier, float learning_rate, int batch_size, int vocab_size, float momentum);
#endif
