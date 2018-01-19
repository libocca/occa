extern "C"
__global__ void reduction(const int entries,
                          const float *vec,
                          float *blockSum) {

  int group = blockIdx.x;
  int item = threadIdx.x;

  __shared__ float s_vec[block];

  if ((group * block + item) < entries) {
    s_vec[item] = vec[group * block + item];
  } else {
    s_vec[item] = 0;
  }

  for (int alive = ((block + 1) / 2); 0 < alive; alive /= 2) {
    __syncthreads();
    if (item < alive) {
      s_vec[item] += s_vec[item + alive];
    }
  }

  if (item == 0) {
    blockSum[group] = s_vec[0];
  }
}
