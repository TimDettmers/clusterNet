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
Matrix add(Matrix A, Matrix B, Matrix out);
Matrix sub(Matrix A, Matrix B);
Matrix sub(Matrix A, Matrix B, Matrix out);
Matrix mul(Matrix A, Matrix B);
Matrix mul(Matrix A, Matrix B, Matrix out);
Matrix div(Matrix A, Matrix B);
Matrix div(Matrix A, Matrix B, Matrix out);
Matrix to_host(Matrix m);
#endif
