#ifndef util
#define util
#include <basicOps.cuh>
Matrix read_csv(char* filename);
cudaEvent_t* tick();
void tock(cudaEvent_t *startstop);
#endif
