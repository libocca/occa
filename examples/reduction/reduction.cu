__global__ void reduction(const int entries, float *a, float *aRed){

  int group = blockIdx.x;
  int item = threadIdx.x;

  __shared__ float s_a[p_Nred];

  s_a[item] = a[group*p_Nred + item];

  for(int alive=p_Nred/2;alive>=1;alive/=2){

    __syncthreads();

    if(item<alive)
      s_a[item] += s_a[item+alive];
  }

  if(item==0){
    aRed[group] = s_a[0];
  }
}
