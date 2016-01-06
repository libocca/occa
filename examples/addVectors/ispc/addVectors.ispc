#define WGDIM0 16
#define WGDIM1 1
#define WGDIM2 1

task void addVectors_kernel(const uniform int dims[],
                            const uniform int &entries,
                            const uniform float a[],
                            const uniform float b[],
                                  uniform float ab[]) {
  const int ntasks = dims[2] * dims[1] * dims[0];
  const int wgsize = WGDIM0 * WGDIM1 * WGDIM2;
  const int taskidx = taskIndex0 + dims[2] * taskIndex1 + dims[2] * dims[3] * taskIndex2;

  foreach (i2 = 0 ... WGDIM2, i1 = 0 ... WGDIM1, i0 = 0 ... WGDIM0)
  {
    int wgindex = i0 + WGDIM0 * i1 + WGDIM1 * WGDIM0 * i2;

    for (int i = taskidx * wgsize + wgindex; i < entries; i += ntasks * wgsize)
    {
      // print("%\n", i);
      ab[i] = a[i] + b[i];
    }
  }
}

export void addVectors(const uniform int dims[],
                       const uniform int o1,
                       const uniform int o2,
                       const uniform int o3,
                       const uniform int &entries,
                       const uniform float a[],
                       const uniform float b[],
                             uniform float ab[])
{
  assert(dims[5] == WGDIM0);
  assert(dims[4] == WGDIM1);
  assert(dims[3] == WGDIM2);

  launch[dims[2], dims[1], dims[0]] addVectors_kernel(dims, entries, a, b, ab);
}
