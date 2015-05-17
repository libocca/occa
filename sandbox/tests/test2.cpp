kernel void k(const int n, int *A){
  for(int j = 0; j < n; ++j){
    for(int o = 0; o < j; ++o; outer0){
      for(int i = 0; i < 16; ++i; inner0){
        A[i] = 0;
      }
    }
  }

  if(n < 10){
    for(int o = 0; o < j; ++o; outer0){
      for(int i = 0; i < 16; ++i; inner0){
        A[i] = 0;
      }
    }
  }
}