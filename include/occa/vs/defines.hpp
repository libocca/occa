#if OCCA_USING_VS
#  ifndef OCCA_VS_DEFINES_HEADER
#  define OCCA_VS_DEFINES_HEADER

//---[ Mode Support ]-------------------
#define OCCA_PTHREADS_ENABLED  1
#define OCCA_OPENMP_ENABLED    1
#define OCCA_OPENCL_ENABLED    0
#define OCCA_CUDA_ENABLED      0
//======================================

//---[ Build Config ]-------------------
#define OCCA_CHECK_ENABLED     1
#define OCCA_GL_ENABLED        1

#ifdef _DEBUG
#  define OCCA_DEBUG_ENABLED   1
#else
#  define OCCA_DEBUG_ENABLED   0
#endif
//======================================

#  endif
#endif