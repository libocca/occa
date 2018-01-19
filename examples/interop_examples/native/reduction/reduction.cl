__kernel void reduction(const int entries,
                        __global const float *vec,
                        __global float *blockSum) {

  int group = get_group_id(0);
  int item  = get_local_id(0);

  __local float s_vec[block];

  if ((group * block + item) < entries) {
    s_vec[item] = vec[group * block + item];
  } else {
    s_vec[item] = 0;
  }

  for (int alive = ((block + 1) / 2); 0 < alive; alive /= 2) {

    barrier(CLK_LOCAL_MEM_FENCE);

    if (item < alive)
      s_vec[item] += s_vec[item+alive];
  }

  if (item == 0) {
    blockSum[group] = s_vec[0];
  }
}
