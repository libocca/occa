//#pragma OPENCL_EXTENSION  cl_nv_pragma_unroll enable
// http://openvidia.sourceforge.net/index.php/OpenCL
#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define UNROLL      1
#define UNROLLOUTER 1
#define datafloat   float
#define Nq          1
#define PAD         1
#define BSIZE       1

__kernel void PCGpart1(const int K,
		       const datafloat lambda,
		       __global   const int     * restrict galnums,
		       __global   const datafloat * restrict geo,
		       __constant       datafloat * restrict D,
		       __global const datafloat * restrict u,
		       __global       datafloat * restrict NL){

#if 0
  0<=i,j,k,m<=N AND 0<=k<K

  ur(i,j,k,e) = D(i,m)*u(m,j,k,e)
  us(i,j,k,e) = D(j,m)*u(i,m,k,e)
  ut(i,j,k,e) = D(k,m)*u(i,j,m,e)

  // (grad phi, grad u) = (Dr' Ds' Dt')*(G)*(Dr; Ds; Dt)
  lap(i,j,k,e)  =  D(m,i)*(G(0,m,j,k,e)*ur(m,j,k,e) + G(1,m,j,k,e)*us(m,j,k,e) + G(2,m,j,k,e)*ut(m,j,k,e));
  lap(i,j,k,e) +=  D(m,j)*(G(1,i,m,k,e)*ur(i,m,k,e) + G(3,i,m,k,e)*us(i,m,k,e) + G(4,i,m,k,e)*ut(i,m,k,e));
  lap(i,j,k,e) +=  D(m,k)*(G(2,i,j,m,e)*ur(i,j,m,e) + G(4,i,j,m,e)*us(i,j,m,e) + G(5,i,j,m,e)*ut(i,j,m,e));

#endif

  /* advects (u.grad)(u,v,w,T) */
  /* shared register for 'r,s' plane */
  __local datafloat LD[Nq][Nq+PAD];
  __local datafloat Lu[Nq][Nq+PAD];
  __local datafloat Lv[Nq][Nq+PAD];
  __local datafloat Lw[Nq][Nq+PAD];

  // u[:][j][i] -> uk[:]
  datafloat   uk[Nq];
  datafloat lapu[Nq]; // use shared ?

  const unsigned int e = get_group_id(0);
  const unsigned int i = get_local_id(0);
  const unsigned int j = get_local_id(1);
  unsigned int k;

  /* load D into local memory */
  LD[i][j] = D[Nq*j+i]; // D is column major

  /* *** Change load here to use scatter array *** */
  /* load pencil of u into register */
  unsigned int id = e*BSIZE+j*Nq+i;

#if UNROLL==1
#pragma unroll 16
#endif
  for(k=0;k<Nq;++k) {
    id = e*BSIZE+k*Nq*Nq+j*Nq+i;
    int gid = galnums[id];

    datafloat tmp = 0.0;

    if(gid >= 0)
      tmp = u[gid]; // this becomes uncoalesced but ** maybe ** cached

    uk[k] = tmp;
    lapu[k] = 0.f;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

#if UNROLLOUTER==1
#pragma unroll 16
#endif
  for(k=0;k<Nq;++k){
    // prefetch geometric factors
    id = 7*e*BSIZE+k*Nq*Nq+j*Nq+i;

    const datafloat G00 = geo[id]; id+= BSIZE;
    const datafloat G01 = geo[id]; id+= BSIZE;
    const datafloat G02 = geo[id]; id+= BSIZE;
    const datafloat G11 = geo[id]; id+= BSIZE;
    const datafloat G12 = geo[id]; id+= BSIZE;
    const datafloat G22 = geo[id]; id+= BSIZE;
    const datafloat J   = geo[id];

    Lu[j][i] = uk[k];

    datafloat ur = 0.f;
    datafloat us = 0.f;
    datafloat ut = 0.f;

#if UNROLL==1
#pragma unroll 16
#endif
    for(int m=0;m<Nq;++m) ut += LD[k][m]*uk[m];

    barrier(CLK_LOCAL_MEM_FENCE);

#if UNROLL==1
#pragma unroll 16
#endif
    for(int m=0;m<Nq;++m) ur += LD[i][m]*Lu[j][m];

#if UNROLL==1
#pragma unroll 16
#endif
    for(int m=0;m<Nq;++m) us += LD[j][m]*Lu[m][i];

    Lw[j][i] = G01*ur + G11*us + G12*ut;
    Lv[j][i] = G00*ur + G01*us + G02*ut;

    // put this here for a performance bump
    const datafloat GDut = G02*ur + G12*us + G22*ut;

    datafloat lapuk = J*(lambda*uk[k]);

    barrier(CLK_LOCAL_MEM_FENCE);

#if UNROLL==1
#pragma unroll 16
#endif
    for(int m=0;m<Nq;++m)
      lapuk += LD[m][j]*Lw[m][i];

#if UNROLL==1
#pragma unroll 16
#endif
    for(int m=0;m<Nq;++m)
      lapu[m] += LD[k][m]*GDut; // DT(m,k)*ut(i,j,k,e)

#if UNROLL==1
#pragma unroll 16
#endif
    for(int m=0;m<Nq;++m)
      lapuk+= LD[m][i]*Lv[j][m];

    lapu[k] += lapuk;
  }

#if UNROLL==1
#pragma unroll 16
#endif
  for(k=0;k<Nq;++k){
    id = e*BSIZE+k*Nq*Nq+j*Nq+i;
    NL[id] = lapu[k];
  }

}
