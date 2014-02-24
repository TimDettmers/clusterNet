#ifndef basicOps
#define basicOps

#define TILE_DIM (32)
#define BLOCK_ROWS (8)
#define COPY_BLOCK_SIZE 16

#define RDM_NUMBERS_PER_THREAD (512)
#define THREADS_PER_BLOCKS (1024)

#define DOT_BLOCKS (128)
#define TILE_SIZE (32)
#define DOT_REPS (4)


//working
//128,16,8
//64,16,4

typedef struct Matrix
{
  int shape[2];
  size_t bytes;
  int size;
  float *data;
} Matrix;

Matrix fill_matrix(int rows, int cols, float fill_value);
Matrix ones(int rows, int cols);
Matrix zeros(int rows, int cols);
Matrix empty(int rows, int cols);

Matrix add(Matrix A, Matrix B);
void add(Matrix A, Matrix B, Matrix out);
Matrix sub(Matrix A, Matrix B);
void sub(Matrix A, Matrix B, Matrix out);
Matrix mul(Matrix A, Matrix B);
void mul(Matrix A, Matrix B, Matrix out);
Matrix div(Matrix A, Matrix B);
void div(Matrix A, Matrix B, Matrix out);

Matrix to_host(Matrix A);
Matrix to_host(Matrix A, int is_row_major);
Matrix to_gpu(Matrix A);
Matrix to_gpu(Matrix A, int is_col_major);
Matrix T(Matrix A);
Matrix to_col_major(Matrix A);
Matrix to_row_major(Matrix A);

Matrix scalarMul(Matrix A, float a);
void scalarMul(Matrix A, float a, Matrix out);

Matrix square(Matrix A);
void square(Matrix A, Matrix out);
Matrix gpuExp(Matrix A);
void gpuExp(Matrix A, Matrix out);
Matrix gpuLog(Matrix A);
void gpuLog(Matrix A, Matrix out);
Matrix gpuSqrt(Matrix A);
void gpuSqrt(Matrix A, Matrix out);

int checkMatrixOperation(Matrix A, Matrix B, Matrix C, int blnMatrixProduct);
int blnFaultySizes(Matrix A, Matrix B, Matrix C);
int blnFaultyMatrixProductSizes(Matrix A, Matrix B, Matrix C);
void printFaultySizeError(Matrix A, Matrix B, Matrix C);
void printFaultyMatrixProductSizeError(Matrix A, Matrix B, Matrix C);

Matrix slice_rows(Matrix A, int start, int end);
Matrix slice_cols(Matrix A, int start, int end);
void vStack(Matrix A, Matrix B, Matrix out);
Matrix vStack(Matrix A, Matrix B);
void hStack(Matrix A, Matrix B, Matrix out);
Matrix hStack(Matrix A, Matrix B);
#endif
