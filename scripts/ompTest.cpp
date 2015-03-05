#include <omp.h>

int main(int argc, char **argv){
#ifdef _OPENMP
  return 0;
#else
  return 1;
#endif
}