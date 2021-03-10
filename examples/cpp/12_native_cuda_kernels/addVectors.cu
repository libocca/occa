extern "C" __global__ void addVectors(const int entries,
                                      const float *a,
                                      const float *b,
                                      float *ab) {
  const int N = threadIdx.x + (16 * blockIdx.x);

  if (N < entries) {
    ab[N] = a[N] + b[N];
  }
}
