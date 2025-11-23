#define block_size 32
extern "C" __global__ __launch_bounds__(block_size) void reduce(
  float *g_idata,
  float *res)
{
 {
   int bid = (0) + blockIdx.x;
   extern __shared__ float sdata[];
   {
     int tid = (0) + threadIdx.x;
     int i = bid * block_size + tid;
     sdata[tid] = g_idata[i];
     __syncthreads();
     for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
       if (tid < s) {
         sdata[tid] += sdata[tid + s];
       }
       __syncthreads();
     }
     if (tid == 0)
       atomicAdd(res, sdata[0]);
   }
 }
}