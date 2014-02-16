
__global__ void kFill_with(float *m, float fill_value, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m[i] = fill_value;
  }
}

__global__ void kAdd(float *m1, float *m2, float *m_out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m_out[i] = m1[i] + m2[i];
  }
}

__global__ void kMul(float *m1, float *m2, float *m_out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m_out[i] = m1[i] * m2[i];
  }
}

__global__ void kSub(float *m1, float *m2, float *m_out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m_out[i] = m1[i] - m2[i];
  }
}

__global__ void kDiv(float *m1, float *m2, float *m_out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m_out[i] = m1[i] / m2[i];
  }
}

__global__ void kExp(float *m1, float *m_out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m_out[i] = __expf(m1[i]);
  }
}

__global__ void kSqrt(float *m1, float *m_out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m_out[i] = sqrt(m1[i]);
  }
}

__global__ void kLog(float *m1, float *m_out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m_out[i] = __logf(m1[i]);
  }
}

__global__ void kSquare(float *m1, float *m_out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m_out[i] = __powf(m1[i], 2);
  }
}

__global__ void kScalarMul(float *m1, float scalar, float *m_out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       m_out[i] = scalar*m1[i];
  }
}
