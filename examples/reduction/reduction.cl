__kernel void reduction(const int entries, __global float *a, __global float *aRed){

  int group = get_group_id(0);
  int item  = get_local_id(0);

  __local float s_a[p_Nred];

  if((group*p_Nred + item) < entries)
    s_a[item] = a[group*p_Nred + item];
  else
    s_a[item] = 0;

  for(int alive = ((p_Nred + 1) / 2); 0 < alive; alive /= 2){

    barrier(CLK_LOCAL_MEM_FENCE);

    if(item < alive)
      s_a[item] += s_a[item+alive];
  }

  if(item == 0){
    aRed[group] = s_a[0];
  }
}
