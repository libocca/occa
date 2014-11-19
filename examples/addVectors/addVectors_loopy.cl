#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))

__kernel void __attribute__ ((reqd_work_group_size(16, 1, 1))) addVectors(int const entries,
                                                                          __global float const *restrict a,
                                                                          __global float const *restrict b,
                                                                          __global float *restrict ab)
{

  if ((-1 + -16 * gid(0) + -1 * lid(0) + entries) >= 0)
    ab[lid(0) + gid(0) * 16] = a[lid(0) + gid(0) * 16] + b[lid(0) + gid(0) * 16];
}
