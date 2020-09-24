__kernel void addVectors(const int entries,
                         __global const float *a,
                         __global const float *b,
                         __global       float *ab) {
  const int N = get_local_id(0) + (16 * get_group_id(0));

  if (N < entries) {
    ab[N] = a[N] + b[N];
  }
}
