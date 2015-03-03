#ifndef basicOps_H
#define basicOps_H

#define TILE_DIM (32)
#define BLOCK_ROWS (8)
#define COPY_BLOCK_SIZE 16

#define RDM_NUMBERS_PER_THREAD (1024)
#define THREADS_PER_BLOCKS (512)
#define BLOCKS (4096)

#define DOT_BLOCKS (128)
#define TILE_SIZE (32)
#define DOT_REPS (4)

#include <cublas_v2.h>
#include <stdio.h>
#include <vector>
#include <sstream>

//working
//128,16,8
//64,16,4

typedef struct Matrix
{
  int rows;
  int cols;
  size_t bytes;
  int size;
  float *data;
  int isDistributed;
  int cols_distributed;

  unsigned char *char_data;

  int isSparse;
  size_t ptr_bytes;
  size_t idx_bytes;
  int *ptr_rows;
  int *idx_cols;
} Matrix;

Matrix *fill_matrix(int rows, int cols, float fill_value);
void fill_matrix(Matrix *A, const float fill_value);
void fill_gpuarray(float *A, const float fill_value, int size);
void fill_gpuarray(int *A, const int fill_value, int size);
void fill_sparse_with_zeros(Matrix *A);
Matrix *ones(int rows, int cols);
Matrix *zeros(int rows, int cols);
Matrix *empty(int rows, int cols);
Matrix *empty_char(int rows, int cols);
Matrix *empty_pinned(int rows, int cols);
Matrix *empty_cpu(int rows, int cols);
Matrix *empty_pinned_sparse(int rows, int cols, float max_sparsity, float sparsity_buffer);
Matrix *empty_pinned_sparse(int rows, int cols, int nonzeros);
Matrix *empty_sparse(int rows, int cols, float max_sparsity, float sparsity_buffer);
Matrix *empty_sparse(int rows, int cols, int nonzeros);
Matrix *arange(int rows, int cols);
Matrix *arange(int start, int rows, int cols);

void uniformSqrtWeight(Matrix * uniform_rdm);
void uniformSqrtWeight(Matrix * uniform_rdm, int in, int out);
void sparseRdmWeight(Matrix *rdm, Matrix *idx, Matrix *out, int connections);
void rand_int(Matrix *int_values,int low, int high);

void add_to_z(Matrix *z, Matrix *z1, Matrix *y, int classes, Matrix *out);

Matrix *add(Matrix *A, Matrix *B);
void add(Matrix *A, Matrix *B, Matrix *out);
Matrix *sub(Matrix *A, Matrix *B);
void sub(Matrix *A, Matrix *B, Matrix *out);
Matrix *mul(Matrix *A, Matrix *B);
void mul(Matrix *A, Matrix *B, Matrix *out);
Matrix *div(Matrix *A, Matrix *B);
void div(Matrix *A, Matrix *B, Matrix *out);
float sum(Matrix *A);
float max(Matrix *A);
int getNonZeroElements(Matrix *A);
int getNonZeroColumns(Matrix *A);

Matrix *to_host(Matrix *A);
Matrix *to_host(Matrix *A, int is_row_major);
Matrix *to_gpu(Matrix *A);
Matrix *to_gpu(Matrix *A, int is_col_major);
Matrix *T(Matrix *A);
Matrix *to_col_major(Matrix *A);
void to_col_major(Matrix *A, Matrix *out);
Matrix *to_row_major(Matrix *A);

Matrix *scalarMul(Matrix *A, float a);
void scalarMul(Matrix *A, float a, Matrix *out);
Matrix *scalarAdd(Matrix *A, float a);
void scalarAdd(Matrix *A, float a, Matrix *out);
void dropout(Matrix *A, Matrix *out, float dropout_rate);
void dropout_cached(Matrix *A, Matrix *dropout, Matrix *out, int idx);
void RMSprop(Matrix *RMS, Matrix *grad, float RMS_multiplier, float learning_rate, int batch_size);
void RMSprop_with_momentum_update(Matrix *RMS, Matrix *grad, Matrix *w, Matrix *m, float RMS_multiplier, float learning_rate, int batch_size, float momentum);
void RMSprop_with_momentum_weight_update(Matrix *RMS, Matrix *grad, Matrix *w, Matrix *m, float RMS_multiplier, float learning_rate, int batch_size, float momentum);
void RMSprop_with_nesterov_weight_update(Matrix *RMS, Matrix *grad, Matrix *w, Matrix *m, float RMS_multiplier, float learning_rate, int batch_size, float momentum);
void RMSprop_with_weight_update(Matrix *RMS, Matrix *grad, Matrix *w, Matrix *m, float RMS_multiplier, float learning_rate, int batch_size, float momentum);
void RMSprop_with_weight_update_8bit(Matrix *RMS, Matrix *grad, Matrix *w, Matrix *m, float RMS_multiplier, float learning_rate, int batch_size, float momentum);
void Nesterov_weight_update(Matrix *RMS, Matrix *grad, Matrix *w, Matrix *m, float RMS_multiplier, float learning_rate, int batch_size, float momentum);

void LocalGrad(Matrix *z, Matrix *w, Matrix *y, float learning_rate, int batch_size, float momentum);
void compression_8bit(Matrix *tbl_flt, Matrix *A, float precision,  Matrix *out);
void compression_8bit_test(Matrix *tbl, Matrix *A, float precision,  Matrix *out);

void decompression_8bit(Matrix *tbl_flt, Matrix *A, float precision,  Matrix *out);

void renormalizeWeights(Matrix *w, Matrix *unit_sums, float limit);

Matrix *square(Matrix *A);
void square(Matrix *A, Matrix *out);
Matrix *abs(Matrix *A);
void abs(Matrix *A, Matrix *out);
Matrix *gpuExp(Matrix *A);
void gpuExp(Matrix *A, Matrix *out);
Matrix *logistic(Matrix *A);
void logistic(Matrix *A, Matrix *out);
Matrix *logisticGrad(Matrix *A);
void logisticGrad(Matrix *A, Matrix *out);
Matrix *gpuLog(Matrix *A);
void gpuLog(Matrix *A, Matrix *out);
Matrix *gpuSqrt(Matrix *A);
void gpuSqrt(Matrix *A, Matrix *out);
Matrix *doubleRectifiedLinear(Matrix *A);
void doubleRectifiedLinear(Matrix *A, Matrix *out);
Matrix *LinearUnit(Matrix *A);
void LinearUnit(Matrix *A, Matrix *out);
Matrix *hardTanH(Matrix *A);
void hardTanH(Matrix *A, Matrix *out);
Matrix *pairwise_ranking(Matrix *A, Matrix *B);
void pairwise_ranking(Matrix *A, Matrix *B, Matrix *out);
Matrix *pairwise_ranking_derivative(Matrix *A, Matrix *B);
void pairwise_ranking_derivative(Matrix *A, Matrix *B, Matrix *out);

Matrix *softmax(Matrix *A);
void softmax(Matrix *A, Matrix *out);
Matrix *subMatrixVector(Matrix *A, Matrix *v);
void subMatrixVector(Matrix *A, Matrix *v, Matrix *out);
Matrix *addMatrixVector(Matrix *A, Matrix *v);
void addMatrixVector(Matrix *A, Matrix *v, Matrix *out);
Matrix *addScaledMatrixVector(Matrix *A, Matrix *v, float weight);
void addScaledMatrixVector(Matrix *A, Matrix *v, float weight, Matrix *out);
Matrix *mulMatrixVector(Matrix *A, Matrix *v);
void mulMatrixVector(Matrix *A, Matrix *v, Matrix *out);
Matrix *argmax(Matrix *A);
void argmax(Matrix* A, Matrix* out);
Matrix *create_t_matrix(Matrix *labels, int max_label);
void create_t_matrix(Matrix *labels, Matrix *out);
Matrix *equal(Matrix *A, Matrix *B);
void equal(Matrix *A, Matrix *B, Matrix *out);
Matrix *maxColumnwise(Matrix *A);
void maxColumnwise(Matrix *A, Matrix *out);
Matrix **maxout(Matrix *A, int maxout_level);
void maxout(Matrix *A, Matrix *out, Matrix *outargmax, int maxout_level);

Matrix *rectified_linear(Matrix *A);
void rectified_linear(Matrix *A, Matrix *out);
Matrix *rectified_linear_derivative(Matrix *A);
void rectified_linear_derivative(Matrix *A, Matrix *out);
Matrix *double_rectified_linear_derivative(Matrix *A);
void double_rectified_linear_derivative(Matrix *A, Matrix *out);
Matrix *hardTanH_derivative(Matrix *A);
void hardTanH_derivative(Matrix *A, Matrix *out);
Matrix *squared_error(Matrix *A, Matrix *targets);
void squared_error(Matrix *A, Matrix *targets, Matrix *out);

void expand_to_maxout_grad(Matrix *error, Matrix *idx, Matrix *grad);

Matrix *rand_numbers(int rows, int cols, Matrix *seeds);
void rand_numbers(Matrix *out, Matrix *seeds);

int checkMatrixOperation(Matrix *A, Matrix *B, Matrix *C, cublasOperation_t T1, cublasOperation_t T2, int blnMatrixProduct);
int blnFaultySizes(Matrix *A, Matrix *B, Matrix *C);
int blnFaultyMatrixProductSizes(Matrix *A, Matrix *B, Matrix *C, cublasOperation_t T1, cublasOperation_t T2);
void printFaultySizeError(Matrix *A, Matrix *B, Matrix *C);
void printFaultyMatrixProductSizeError(Matrix *A, Matrix *B, Matrix *C, cublasOperation_t T1, cublasOperation_t T2);
void printData(Matrix *A);

Matrix *slice_rows(Matrix *A, int start, int end);
Matrix *slice_cols(Matrix *A, int start, int end);
void vStack(Matrix *A, Matrix *B, Matrix *out);
Matrix *vStack(Matrix *A, Matrix *B);
void hStack(Matrix *A, Matrix *B, Matrix *out);
Matrix *hStack(Matrix *A, Matrix *B);
void hStackN(float** arrA, int general_size, Matrix *out, int matrices_count);
void hStackN(Matrix** arrA, int general_size, Matrix *out, int matrices_count);
void vStackN(Matrix** arrA, Matrix *out, int matrices_count);

void addGradientsN(Matrix** arrA,  int myrank, int matrices_count, float multiplier);

void sparse_dot(Matrix *A, Matrix *B, Matrix *out);
void construct_vocab_matrix(Matrix *vocab_idx, Matrix *vocab_idx_y, Matrix *batch_X, Matrix *batch_y, Matrix *vocab, Matrix *rdm_idx);
//void update_vocab_with_gradient(Matrix *grad, Matrix *vocab_idx, Matrix *vocab, float learning_rate);
void expand_double_vocab_gradient(Matrix *gradX, Matrix *gradY, Matrix *vocab_idx_X, Matrix *vocab_idx_Y, Matrix *vocab, Matrix *vocab_grad, Matrix *vocab_grad_idx, float learning_rate);
void expand_vocab_gradient(Matrix *grad, Matrix *vocab_idx, Matrix *vocab_grad);
void expand_partial_vocab_gradient(Matrix *grad, Matrix *vocab_idx, Matrix *vocab_grad, int matrix_idx, int matrix_count);
void update_vocab_with_gradient(Matrix *grad, Matrix *vocab_idx, Matrix *vocab, float learning_rate);
void concatVocabBatchesN(Matrix** arrBatch_X, Matrix **arrBatch_Y, Matrix *out_X, Matrix *out_Y, int window_size, int matrices_count);
void expand_vocab_gradient_middle_word(Matrix *grad, Matrix *vocab_idx, Matrix *vocab_grad);
void matmul(Matrix *A, Matrix *B, Matrix *out, int T1, int T2);
void dot8bit(Matrix *charA, Matrix *charB, Matrix* out, Matrix *flt_tbl, float precisionA, float precisionB);
void dot8bit_shared(Matrix *charA, Matrix *charB, Matrix* out, Matrix *flt_tbl, float precisionA, float precisionB);
#endif
