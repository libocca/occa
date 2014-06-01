#ifndef OCCA_DEFINES_HEADER
#define OCCA_DEFINES_HEADER

#include "ocl_preprocessor.hpp"

//---[ Checks and Info ]----------------
#if OCCA_DEBUG
#  define OCCA_CHECK2( _expr , file , line , func )                     \
  do {                                                                  \
    size_t expr = _expr;                                                \
    if( !expr ){                                                        \
      std::cout << '\n'                                                 \
                << "---[ Error ]--------------------------------------------\n" \
                << "    File     : " << file << '\n'                    \
                << "    Function : " << func << '\n'                    \
                << "    Line     : " << line << '\n'                    \
        std::cout << "========================================================\n"; \
      throw 1;                                                          \
    }                                                                   \
  } while(0)
#  define OCCA_CHECK( _expr ) OCCA_CHECK2( _expr , __FILE__ , __LINE__ , __PRETTY_FUNCTION__)
#else
#  define OCCA_CHECK( _expr )
#endif

#define OCCA_MEM_ALIGN  16
#define OCCA_SIMD_WIDTH 8
//======================================


//---[ Base ]---------------------------
#define OCCA_EXTRACT_DATA(MODE, CLASS)                          \
  MODE##CLASS##Data_t &data_ = *((MODE##CLASS##Data_t*) data);

#define OCCA_KERNEL_ARG(N) , const kernelArg &arg##N
#define OCCA_KERNEL_ARGS(N)  const kernelArg &arg1 OCL_FOR(2, N, OCCA_KERNEL_ARG)

#define OCCA_INPUT_KERNEL_ARG(N) , arg##N
#define OCCA_INPUT_KERNEL_ARGS(N)  arg1 OCL_FOR(2, N, OCCA_INPUT_KERNEL_ARG)

#define OCCA_VIRTUAL_KERNEL_OPERATOR_DECLARATION(N)   \
  virtual void operator() (OCCA_KERNEL_ARGS(N)) = 0;

#define OCCA_VIRTUAL_KERNEL_OPERATOR_DECLARATIONS                       \
  OCL_FOR_2(1, OCL_MAX_FOR_LOOPS, OCCA_VIRTUAL_KERNEL_OPERATOR_DECLARATION)

#define OCCA_KERNEL_OPERATOR_DECLARATION(N)     \
  void operator() (OCCA_KERNEL_ARGS(N));

#define OCCA_KERNEL_OPERATOR_DECLARATIONS                           \
  OCL_FOR_2(1, OCL_MAX_FOR_LOOPS, OCCA_KERNEL_OPERATOR_DECLARATION)

#define OCCA_KERNEL_OPERATOR_DEFINITION(N)        \
  void kernel::operator() (OCCA_KERNEL_ARGS(N)){  \
    (*kHandle)(OCCA_INPUT_KERNEL_ARGS(N));        \
  }

#define OCCA_KERNEL_OPERATOR_DEFINITIONS                            \
  OCL_FOR_2(1, OCL_MAX_FOR_LOOPS, OCCA_KERNEL_OPERATOR_DEFINITION)
//======================================


//---[ OpenMP ]-------------------------
#  define OCCA_OPENMP_FUNCTION_ARG(N) , void *arg##N
#  define OCCA_OPENMP_FUNCTION_ARGS(N)  void *occaKernelInfoArgs OCL_FOR(1, N, OCCA_OPENMP_FUNCTION_ARG)

#  define OCCA_OPENMP_FUNCTION_POINTER_TYPEDEF(N) typedef void (*functionPointer##N)(OCCA_OPENMP_FUNCTION_ARGS(N));
#  define OCCA_OPENMP_FUNCTION_POINTER_TYPEDEFS                         \
    OCL_FOR_2(1, OCL_MAX_FOR_LOOPS, OCCA_OPENMP_FUNCTION_POINTER_TYPEDEF)

#  define OCCA_OPENMP_INPUT_FUNCTION_ARG(N) , arg##N.data()
#  define OCCA_OPENMP_INPUT_FUNCTION_ARGS(N)  occaKernelInfoArgs.data OCL_FOR(1, N, OCCA_OPENMP_INPUT_FUNCTION_ARG)

#  define OCCA_OPENMP_KERNEL_OPERATOR_DEFINITION(N)                     \
    template <>                                                         \
    void kernel_t<OpenMP>::operator() (OCCA_KERNEL_ARGS(N)){            \
      OCCA_EXTRACT_DATA(OpenMP, Kernel);                                \
      functionPointer##N tmpKernel = (functionPointer##N) data_.handle; \
                                                                        \
        occaArgs_t occaKernelInfoArgs(inner,outer);                     \
                                                                        \
        tmpKernel(OCCA_OPENMP_INPUT_FUNCTION_ARGS(N));                  \
    }

#  define OCCA_OPENMP_KERNEL_OPERATOR_DEFINITIONS                       \
  OCL_FOR_2(1, OCL_MAX_FOR_LOOPS, OCCA_OPENMP_KERNEL_OPERATOR_DEFINITION)
//======================================


//---[ OpenCL ]-------------------------
#if OCCA_DEBUG_ENABLED
#  define OCCA_CL_CHECK( _str , _statement ) OCCA_CL_CHECK2( _str , _statement , __FILE__ , __LINE__ )
#  define OCCA_CL_CHECK2( _str , _statement , file , line )              \
  do {                                                                  \
    cl_int _error = _statement;                                         \
    if(_error){                                                         \
      _error = _error < 0  ? _error : -_error;                          \
      _error = _error < 65 ? _error : 15;                               \
                                                                        \
      std::cout << "Error\n"                                            \
                << "    File    : " << file << '\n'                     \
                << "    Line    : " << line << '\n'                     \
                << "    Error   : OpenCL Error [ " << _error << " ]: " << openclError(_error) << '\n' \
                << "    Message : " << _str << '\n';                    \
      throw 1;                                                          \
    }                                                                   \
  } while(0);
#else
#  define OCCA_CL_CHECK( _str , _statement ) do { _statement; } while(0);
#endif

#  define OCCA_OPENCL_SET_KERNEL_ARG(N)                                 \
  OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [ " + #N + " ]", \
               clSetKernelArg(kernel_, N, arg##N.size, arg##N.data()));

#  define OCCA_OPENCL_SET_KERNEL_ARGS(N)                                \
  OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [ " + #N + " ]", \
                clSetKernelArg(kernel_, 0, sizeof(cl_int), &occaKernelInfoArgs)); \
                                                                        \
  OCL_FOR(1, N, OCCA_OPENCL_SET_KERNEL_ARG)

#  define OCCA_OPENCL_KERNEL_OPERATOR_DEFINITION(N)                     \
  template <>                                                           \
  void kernel_t<OpenCL>::operator() (OCCA_KERNEL_ARGS(N)){              \
    OCCA_EXTRACT_DATA(OpenCL, Kernel);                                  \
    cl_kernel kernel_   = data_.kernel;                                 \
    occa::dim fullOuter = outer*inner;                                  \
                                                                        \
    cl_int occaKernelInfoArgs = 0;                                      \
                                                                        \
    OCCA_OPENCL_SET_KERNEL_ARGS(N);                                     \
                                                                        \
    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",          \
                 clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream), \
                                        kernel_,                        \
                                        (cl_int) dims,                  \
                                        NULL,                           \
                                        (size_t*) &fullOuter,           \
                                        (size_t*) &inner,               \
                                        0, NULL, NULL));                \
  }

#  define OCCA_OPENCL_KERNEL_OPERATOR_DEFINITIONS                       \
  OCL_FOR_2(1, OCL_MAX_FOR_LOOPS, OCCA_OPENCL_KERNEL_OPERATOR_DEFINITION)
//======================================


//---[ CUDA ]---------------------------
#if OCCA_DEBUG_ENABLED
#  define OCCA_CUDA_CHECK( _str , _statement ) OCCA_CUDA_CHECK2( _str , _statement , __FILE__ , __LINE__ )
#  define OCCA_CUDA_CHECK2( _str , _statement , file , line )           \
  do {                                                                  \
    CUresult errorCode = _statement;                                    \
    if(errorCode){                                                      \
      std::cout << "Error\n"                                            \
                << "    File    : " << file << '\n'                     \
                << "    Line    : " << line << '\n'                     \
                << "    Error   : CUDA Error [ " << errorCode << " ]: " << occa::cudaError(errorCode) << '\n' \
                << "    Message : " << _str << '\n';                    \
      throw 1;                                                          \
    }                                                                   \
  } while(0);
#else
#  define OCCA_CUDA_CHECK( _str , _statement ) do { _statement; } while(0);
#endif

#  define OCCA_CUDA_KERNEL_ARG(N) , arg##N.data()
#  define OCCA_CUDA_KERNEL_ARGS(N) &occaKernelInfoArgs OCL_FOR(1, N, OCCA_CUDA_KERNEL_ARG)

#  define OCCA_CUDA_KERNEL_OPERATOR_DEFINITION(N)         \
  template <>                                             \
  void kernel_t<CUDA>::operator() (OCCA_KERNEL_ARGS(N)){  \
    OCCA_EXTRACT_DATA(CUDA, Kernel);                      \
    CUfunction function_ = data_.function;                \
    int occaKernelInfoArgs = 0;                           \
                                                          \
    void *args[N+1] = {OCCA_CUDA_KERNEL_ARGS(N)};         \
                                                          \
    cuLaunchKernel(function_,                             \
                   outer.x, outer.y, outer.z,             \
                   inner.x, inner.y, inner.z,             \
                   0,                                     \
                   *((CUstream*) dev->currentStream),     \
                   args, 0);                              \
  }

#  define OCCA_CUDA_KERNEL_OPERATOR_DEFINITIONS                         \
  OCL_FOR_2(1, OCL_MAX_FOR_LOOPS, OCCA_CUDA_KERNEL_OPERATOR_DEFINITION)
//======================================

#endif
