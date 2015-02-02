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

#define WR_MIN(X,Y) ((X) < (Y) ? (X) : (Y))

int a23[1][2][3][4][5][6];

int  a = 0, *b = 1;
int *c = 0,  d = 1;

#define PI   3.14159265358
#define THREE 3
#define FOUR THREE
#define FOUR 4
#define PI_##THREE #<PI / 3.0#>           // PI/3
#define PI_#<THREE + 1#> 0.78539816339   // PI/4

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

const double pi_3 = PI_3;
const double pi_4 = PI_4;

/* Testing 1 */

/*
  Testing 2
*/

gotoTest: