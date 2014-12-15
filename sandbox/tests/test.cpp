#if
#  define sr     1
#  define tFloat bool
#elif (1.2f < (+1e0f + 0.3)) && (1 & 3) && (true || false)
#  define sr     2
#  define tFloat char
#elif false
#  define sr     3
#  define tFloat int
#else
#  define sr     4
#  define tFloat float
#endif

#define freq 3.0
#define dt 0.1
#define dx 0.1
#define w 10
#define h 10
#define mX 0
#define mY 0

int a23[1][2][3][4][5][6];

int  a = 0, *b = 1;
int *c = 0,  d = 1;

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

#define FD_STENCIL_4(D)                               \
  {-0.00178571/(D*D), 0.0253968/(D*D), -0.2/(D*D),    \
      1.6/(D*D), -2.84722/(D*D), 1.6/(D*D),           \
      -0.2/(D*D), 0.0253968/(D*D), -0.00178571/(D*D)}

#define FD_STENCIL_5(D)                                   \
  {0.00031746/(D*D), -0.00496032/(D*D), 0.0396825/(D*D),  \
      -0.238095/(D*D), 1.66667/(D*D), -2.92722/(D*D),     \
      1.66667/(D*D), -0.238095/(D*D), 0.0396825/(D*D),    \
      -0.00496032/(D*D), 0.00031746/(D*D)}

#define FD_STENCIL_6(D)                                     \
  {-6.01251e-05/(D*D), 0.00103896/(D*D), -0.00892857/(D*D), \
      0.0529101/(D*D), -0.267857/(D*D), 1.71429/(D*D),      \
      -2.98278/(D*D), 1.71429/(D*D), -0.267857/(D*D),       \
      0.0529101/(D*D), -0.00892857/(D*D), 0.00103896/(D*D), \
      -6.01251e-05/(D*D)}

#define FD_STENCIL_7(D)                                         \
  {1.18929e-05/(D*D), -0.000226625/(D*D), 0.00212121/(D*D),     \
      -0.0132576/(D*D), 0.0648148/(D*D), -0.291667/(D*D),       \
      1.75/(D*D), -3.02359/(D*D), 1.75/(D*D),                   \
      -0.291667/(D*D), 0.0648148/(D*D), -0.0132576/(D*D),       \
      0.00212121/(D*D), -0.000226625/(D*D), 1.18929e-05/(D*D)}

#define FD_STENCIL2(N,D) FD_STENCIL_##N(D)
#define FD_STENCIL(N,D)  FD_STENCIL2(N,D) // Unwraps N and D

const tFloat tStencil[] = FD_STENCIL(1 , dt);
const tFloat xStencil[] = FD_STENCIL(sr, dx);
const tFloat blah[1 + (2*3)];

/* Testing 1 */

/*
  Testing 2
*/

gotoTest:

// 0.9899*sqrt(8.0*log(10.0))/(PI*freq);
const tFloat hat_t0 = 1.3523661426929/freq; /* Testing 3 */
const tFloat &hat_t1 = hat_t0;

occaFunction tFloat dummyFunction(tFloat t){
  return 0;
}

occaFunction tFloat hatWavelet(tFloat t);
occaFunction tFloat hatWavelet(tFloat t){
  const tFloat pift  = PI*freq*(t - hat_t0);
  const tFloat pift2 = pift*pift;

  return (1.0 - 2.0*pift2)*exp(-pift2);
}

const int2 * const a34;

#if 1
occaKernel void fd2d(tFloat *u1,
                     tFloat *u2,
                     tFloat *u3,
                     const texture tFloat tex1[],
                     texture tFloat tex2[][],
                     texture tFloat **tex3,
                     const tFloat currentTime){
  for(int by = 0; by < bDimY; ++by; outer0){
    for(int bx = 0; bx < bDimX; ++bx; outer1){
      shared tFloat Lu[bDimY + 2*sr][bDimX + 2*sr];
      exclusive tFloat r_u2 = 2, r_u3 = 3, r_u4[3], *r_u5, *r_u6[3];

      for(int ly = 0; ly < lDimY; ++ly; inner1){
        for(int lx = 0; lx < lDimX; ++lx; inner0){
          const int tx = bx * lDimX + lx;
          const int ty = by * lDimY + ly;

          int y1, y2;
          y1 = y2 = 0;

          const int id = ty*w + tx;

          float data = tex1[0][0];
          tex1[0][0] = data;

          r_u2 = u2[id];
          r_u3 = u3[id];

          const int nX1 = (tx - sr + w) % w;
          const int nY1 = (ty - sr + h) % h;

          const int nX2 = (tx + bDimX - sr + w) % w;
          const int nY2 = (ty + bDimY - sr + h) % h;

          Lu[ly][lx] = u2[nY1*w + nX1];

          if(lx < 2*sr){
            Lu[ly][lx + bDimX] = u2[nY1*w + nX2];

            if(ly < 2*sr)
              Lu[ly + bDimY][lx + bDimX] = u2[nY2*w + nX2];
          }

          if(ly < 2*sr)
            Lu[ly + bDimY][lx] = u2[nY2*w + nX1];

          a.b = 3;
        }
      }

      // barrier(localMemFence);

      for(int ly = 0; ly < lDimY; ++ly; inner1){
        for(int lx = 0; lx < lDimX; ++lx; inner0){
          const int tx = bx * lDimX + lx;
          const int ty = by * lDimY + ly;

          const int id = ty*w + tx;

          tFloat lap = 0.0;

          if(true)
            blah;
          else if(true)
            blah;
          else
            blah;

          for(int i = 0; i < (2*sr + 1); i++)
            lap += xStencil[i]*Lu[ly + sr][lx + i] + xStencil[i]*Lu[ly + i][lx + sr];

          for(int i = 0; i < (2*sr + 1); i++){
            lap += xStencil[i]*Lu[ly + sr][lx + i] + xStencil[i]*Lu[ly + i][lx + sr];
          }

          const tFloat u_n1 = (-tStencil[1]*r_u2 - tStencil[2]*r_u3 + lap)/tStencil[0];

          if((tx == mX) && (ty == mY))
            u1[id] = u_n1 + hatWavelet(currentTime)/tStencil[0];
          else
            u1[id] = u_n1;
        }
      }
    }
  }
}
#endif
