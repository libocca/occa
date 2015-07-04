#include "~/gitRepos/night/sandbox/tests/testHeader.hpp"

/*
  /* Testing " '
*/

#define BOXIFY(x)                               \
  {                                             \
    if ((x) >= 0.5)                             \
      x -= 1.;                                  \
    else if ((x) < -0.5)                        \
      x += 1.;                                  \
  }

typedef int blah234;

// 0.9899*sqrt(8.0*log(10.0))/(PI*freq);
const tFloat hat_t0 = 1.3523661426929/freq; /* Testing 3 */
const tFloat &hat_t1 = hat_t0;

namespace occa {
  int x;
};

int     (*f1)(int);
int   (*(*f2)(int))(double);
int (*(*(*f3)(int))(double))(float);


occaFunction tFloat hatWavelet(tFloat t);

occaFunction tFloat dummyFunction(shared tFloat t){
  return 0;
}

occaFunction tFloat hatWavelet(tFloat t);
occaFunction tFloat hatWavelet(tFloat t){
  const tFloat pift  = PI*freq*(t - hat_t0);
  const tFloat pift2 = pift*pift;

  double *psi @(dim(X,Y,Z), idxOrder(2,1,0));
  psi(a,b,c);

  double *phi @dim(D);
  phi(blah + 1);

  double *delta @(dim(BJ,BI,BJ,AI), idxOrder(3,2,1,0));

  for(int ai = 0; ai < 10; ++ai;       loopOrder("a", 0)) {
    for(int bj = 0; bj < 10; ++bj;     loopOrder("b", 0)) {
      for(int bi = 0; bi < 10; ++bi;   loopOrder("b", 1)) {
        for(int aj = 0; aj < 10; ++aj; loopOrder("a", 1)) {
            delta(aj,bi,bj,ai) = 0;
        }
      }
    }
  }

  return (1. - 2.0*pift2)*exp(-pift2);
}

const int2 * const a34;

float numberSamples[] = {1, +1, -1,
                         1., +1., -1.,
                         1.0, -1.0, +1.0,
                         1.0f, -1.0f, +1.0f,
                         1.01F, -1.01F, +1.01F,
                         1.0e1f, -1.0e-11f, +1.0e+111f,
                         1.0E1, -1.0E-11, +1.0E+111,
                         1.01F, -1.01F, +1.01F,
                         1l, -1l, +1l,
                         1.01L, -1.01L, +1.01L,
                         0b001, -0b010, +0b011,
                         0B100, -0B101, +0B110,
                         00001, -00010, +00011,
                         00100, -00101, +00110,
                         0x001, -0x010, +0x011,
                         0X100, -0X101, +0X110};

#if 1
occaKernel void fd2d(tFloat *u1,
                     tFloat *u2,
                     tFloat *u3,
                     const texture tFloat tex1[],
                     texture tFloat tex2[][],
                     texture tFloat **tex3,
                     const tFloat currentTime){

  const int bDimX = 16;
  const int bDimY = 16 + bDimX;
  const int bDimZ = 16 + bDimY;

  const int lDimX = 16 + bDimY;
  int lDimY = lDimX;
  int lDimZ = lDimX + lDimY;

  double2 s[2];

  BOXIFY(s[i].x);

  for(int i = 0; i < 10; ++i; loopOrder(0)){
    for(int j = 0; j < 10; ++j; loopOrder(1)){
      for(int n = 0; n < bDimX; ++n; tile(lDimX)){
      }
    }
  }

  // for(int2 i(0,1); i.x < bDimX, i.y < bDimY; i.y += 2, ++i.x; tile(lDimX,lDimY)){
  // }

  // for(int2 i(0,1); i.x < bDimX, i.y < bDimY; ++i; tile(lDimX,lDimY)){
  // }

  shared tFloat Lu[bDimY + 2*sr][bDimX + 2*sr];

  for(int by = 0; by < bDimY; by += 16; outer0){
    for(int bx = 0; bx < bDimX; bx += 16; outer1){
      exclusive tFloat r_u2 = 2, r_u3 = 3, r_u4[3], *r_u5, *r_u6[3];

      for(int ly = by; ly < (by + lDimY); ++ly; inner1){
        for(int lx = bx; lx < (by + lDimX); ++lx; inner0){
          const int tx = bx * lDimX + lx;
          const int ty = by * lDimY + ly;

          float2 sj, si;
          float2 s = sj - si;

          Lu[0] = WR_MIN(Lu[tx], Lu[tx+512]);

          int y1, y2;

          {
            y1 = y2 = 0;
          }

          int tmpin = *u1;

          switch(y1){
          case 0 : printf("0\n"); break;
          case 1 : {printf("1\n");}
          default: printf("default\n"); break;
          }

          switch(y2)
          case 0: printf("0\n");

          const int id = ty*w + tx;

          float *__u1 = &u1[bDimX];
          float *__u2 = (float*) (&(u1[bDimX]));

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

          for(int i = 0; i < (2*sr + 1); i++){
            lap += xStencil[i]*Lu[ly + sr][lx + i] + xStencil[i]*Lu[ly + i][lx + sr];
            if(i < 2)
              continue;
            break;
          }

          continue;

          for(int i = 0; i < (2*sr + 1); i++){
            lap += xStencil[i]*Lu[ly + sr][lx + i] + xStencil[i]*Lu[ly + i][lx + sr];
          }

          const tFloat u_n1 = (-tStencil[1]*r_u2 - tStencil[2]*r_u3 + lap)/tStencil[0];

          if((tx == mX) && (ty == mY))
            u1[id] = u_n1 + hatWavelet(currentTime, 1, 2)/tStencil[0];
          else
            u1[id] = u_n1;
        }
      }
    }
  }
}
#endif
