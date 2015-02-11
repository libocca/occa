__global__ void reduction(const int entries, float *a, float *aRed){

  int group = blockIdx.x;
  int item = threadIdx.x;

  __shared__ float s_a[p_Nred];

  if((group*p_Nred + item) < entries)
    s_a[item] = a[group*p_Nred + item];
  else
    s_a[item] = 0;

  for(int alive = ((p_Nred + 1) / 2); 0 < alive; alive /= 2){

    __syncthreads();

    if(item < alive)
      s_a[item] += s_a[item+alive];
  }

  if(item == 0){
    aRed[group] = s_a[0];
  }
}
