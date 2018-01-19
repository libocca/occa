#define PI   3.14159265358
#define PI_4 0.78539816339           // PI/4

#define FD_STENCIL_1(D)                         \
  {1.0/(D*D), -2.0/(D*D), 1.0/(D*D)}

#define FD_STENCIL_2(D)                         \
  {-0.0833333/(D*D), 1.33333/(D*D), -2.5/(D*D), \
      1.33333/(D*D), -0.0833333/(D*D)}

#define FD_STENCIL_3(D)                         \
  {0.0111111/(D*D), -0.15/(D*D), 1.5/(D*D),     \
      -2.72222/(D*D), 1.5/(D*D), -0.15/(D*D),   \
      0.0111111/(D*D)}

#define FD_STENCIL_4(D)                                 \
  {-0.00178571/(D*D), 0.0253968/(D*D), -0.2/(D*D),      \
      1.6/(D*D), -2.84722/(D*D), 1.6/(D*D),             \
      -0.2/(D*D), 0.0253968/(D*D), -0.00178571/(D*D)}

#define FD_STENCIL_5(D)                                         \
  {0.00031746/(D*D), -0.00496032/(D*D), 0.0396825/(D*D),        \
      -0.238095/(D*D), 1.66667/(D*D), -2.92722/(D*D),           \
      1.66667/(D*D), -0.238095/(D*D), 0.0396825/(D*D),          \
      -0.00496032/(D*D), 0.00031746/(D*D)}

#define FD_STENCIL_6(D)                                         \
  {-6.01251e-05/(D*D), 0.00103896/(D*D), -0.00892857/(D*D),     \
      0.0529101/(D*D), -0.267857/(D*D), 1.71429/(D*D),          \
      -2.98278/(D*D), 1.71429/(D*D), -0.267857/(D*D),           \
      0.0529101/(D*D), -0.00892857/(D*D), 0.00103896/(D*D),     \
      -6.01251e-05/(D*D)}

#define FD_STENCIL_7(D)                                         \
  {1.18929e-05/(D*D), -0.000226625/(D*D), 0.00212121/(D*D),     \
      -0.0132576/(D*D), 0.0648148/(D*D), -0.291667/(D*D),       \
      1.75/(D*D), -3.02359/(D*D), 1.75/(D*D),                   \
      -0.291667/(D*D), 0.0648148/(D*D), -0.0132576/(D*D),       \
      0.00212121/(D*D), -0.000226625/(D*D), 1.18929e-05/(D*D)}

#define FD_STENCIL2(N,D) FD_STENCIL_##N(D)
#define FD_STENCIL(N,D)  FD_STENCIL2(N,D) // Unwraps N and D

#ifndef tFloat
#  define tFloat float
#endif

__constant tFloat tStencil[] = FD_STENCIL(1 , dt);
__constant tFloat xStencil[] = FD_STENCIL(sr, dx);

// 0.9899*sqrt(8.0*log(10.0))/(PI*freq);
__constant tFloat hat_t0 = 1.3523661426929/freq;

tFloat hatWavelet(tFloat t);

tFloat hatWavelet(tFloat t) {
  const tFloat pift  = PI*freq*(t - hat_t0);
  const tFloat pift2 = pift*pift;

  return (1.0 - 2.0*pift2)*exp(-pift2);
}

__kernel void fd2d(__global       tFloat *u1,
                   __global const tFloat *u2,
                   __global const tFloat *u3,
                   const tFloat currentTime) {

  __local tFloat Lu[By + 2*sr][Bx + 2*sr];
  tFloat r_u2, r_u3;

  const int bx = (get_group_id(0) * Bx);
  const int by = (get_group_id(1) * By);

  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  const int tx = bx + lx;
  const int ty = by + ly;

  const int id = ty*w + tx;

  r_u2 = u2[id];
  r_u3 = u3[id];

  const int nX1 = (tx - sr + w) % w;
  const int nY1 = (ty - sr + h) % h;

  const int nX2 = (tx + Bx - sr + w) % w;
  const int nY2 = (ty + By - sr + h) % h;

  Lu[ly][lx] = u2[nY1*w + nX1];

  if (lx < 2*sr) {
    Lu[ly][lx + Bx] = u2[nY1*w + nX2];

    if (ly < 2*sr)
      Lu[ly + By][lx + Bx] = u2[nY2*w + nX2];
  }

  if (ly < 2*sr)
    Lu[ly + By][lx] = u2[nY2*w + nX1];

  barrier(CLK_LOCAL_MEM_FENCE);

  tFloat lap = 0.0;

  for (int i = 0; i < (2*sr + 1); i++)
    lap += xStencil[i]*Lu[ly + sr][lx + i] + xStencil[i]*Lu[ly + i][lx + sr];

  const tFloat u_n1 = (-tStencil[1]*r_u2 - tStencil[2]*r_u3 + lap)/tStencil[0];

  if ((tx == mX) && (ty == mY))
    u1[id] = u_n1 + hatWavelet(currentTime)/tStencil[0];
  else
    u1[id] = u_n1;
}
