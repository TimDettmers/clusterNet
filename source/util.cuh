#ifndef util_H
#define util_H
#include <basicOps.cuh>
#include <string>

#define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (false)

Matrix *read_csv(const char* filename);
void write_csv(const char* filename, Matrix *X, const char* header, Matrix *ids);
Matrix *read_hdf5(const char * filepath);
Matrix *read_sparse_hdf5(const char * filepath);
Matrix *read_hdf5(const char *filepath, const char *tag);
void write_hdf5(const char * filepath, Matrix *A);

cudaEvent_t* tick();
void tock(cudaEvent_t* startstop);
void tock(cudaEvent_t* startstop, std::string text);
void tock(std::string text, float tocks);
float tock(cudaEvent_t* startstop, float tocks);
int test_eq(float f1, float f2, char* message);
int test_eq(float f1, float f2, int idx1, int idx2, char* message);
int test_eq(int i1, int i2, char* message);
int test_eq(int i1, int i2, int idx1, int idx2, char* message);
int test_matrix(Matrix *A, int rows, int cols);
void printmat(Matrix *A);
void printmat(Matrix *A, int end_rows, int end_cols);
void printmat(Matrix *A, int start_row, int end_row, int start_col, int end_col);
void print_matrix(Matrix *A, int end_rows, int end_cols);
void print_matrix(Matrix *A, int start_row, int end_row, int start_col, int end_col);
bool replace(std::string& str, const std::string& from, const std::string& to);
void slice_sparse_to_dense(Matrix *X, Matrix *out, int start, int length);
float determine_max_sparsity(Matrix *X, int batch_size);
#endif

