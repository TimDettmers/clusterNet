#ifndef basicOps
#define basicOps
typedef struct Matrix
{
  int shape[2];
  size_t bytes;
  int size;
  float *data;
} Matrix;


Matrix allocate(Matrix m);
Matrix dot(Matrix A, Matrix B);
Matrix fill_matrix(int rows, int cols, float fill_value);
Matrix ones(int rows, int cols);
Matrix zeros(int rows, int cols);
Matrix add(Matrix A, Matrix B);
void add(Matrix A, Matrix B, Matrix out);
Matrix sub(Matrix A, Matrix B);
void sub(Matrix A, Matrix B, Matrix out);
Matrix mul(Matrix A, Matrix B);
void mul(Matrix A, Matrix B, Matrix out);
Matrix div(Matrix A, Matrix B);
void div(Matrix A, Matrix B, Matrix out);
Matrix to_host(Matrix m);

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

#endif
