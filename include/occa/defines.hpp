#ifndef OCCA_DEFINES_HEADER
#define OCCA_DEFINES_HEADER

#ifndef LINUX_OS
#  define LINUX_OS 1
#endif

#ifndef OSX_OS
#  define OSX_OS 2
#endif

#ifndef WINDOWS_OS
#  define WINDOWS_OS 4
#endif

#ifndef WINUX_OS
#  define WINUX_OS (LINUX_OS | WINDOWS_OS)
#endif

#ifndef OCCA_USING_VS
#  ifdef _MSC_VER
#    define OCCA_USING_VS 1
#  else
#    define OCCA_USING_VS 0
#  endif
#endif

#ifndef OCCA_OS
#  if defined(WIN32) || defined(WIN64)
#    if OCCA_USING_VS
#      define OCCA_OS WINDOWS_OS
#    else
#      define OCCA_OS WINUX_OS
#    endif
#  elif __APPLE__
#    define OCCA_OS OSX_OS
#  else
#    define OCCA_OS LINUX_OS
#  endif
#endif

#if OCCA_USING_VS
#  define OCCA_VS_VERSION _MSC_VER
#  include "vs/defines.hpp"
#endif

#ifndef __PRETTY_FUNCTION__
#  define __PRETTY_FUNCTION__ __FUNCTION__
#endif

#define OCCA_STRINGIFY2(macro) #macro
#define OCCA_STRINGIFY(macro) OCCA_STRINGIFY2(macro)

#ifdef __cplusplus
#  define OCCA_START_EXTERN_C extern "C" {
#  define OCCA_END_EXTERN_C   }
#else
#  define OCCA_START_EXTERN_C
#  define OCCA_END_EXTERN_C
#endif

#if   (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
#  define OCCA_INLINE inline __attribute__ ((always_inline))
#elif (OCCA_OS == WIN_OS)
#  define OCCA_INLINE __forceinline
#endif

#if defined __arm__
#  define OCCA_ARM 1
#else
#  define OCCA_ARM 0
#endif

#if defined(__x86_64__) || defined(_M_X64) // 64 Bit
#  define OCCA_64_BIT 1
#  define OCCA_32_BIT 0
#elif defined(__i386) || defined(_M_IX86) // 32 Bit
#  define OCCA_64_BIT 0
#  define OCCA_32_BIT 1
#elif defined(__ia64) || defined(__itanium__) || defined(_A_IA64) // Itanium
#  define OCCA_64_BIT 1
#  define OCCA_32_BIT 0
#endif

#if OCCA_ARM
#  define OCCA_LFENCE __asm__ __volatile__ ("dmb")
#else
#  define OCCA_LFENCE __asm__ __volatile__ ("lfence")
#endif

//---[ Checks and Info ]----------------
#ifndef OCCA_COMPILED_FOR_JULIA
#  define OCCA_THROW abort()
#else
#  define OCCA_THROW exit(1)
#endif

#define OCCA_EMPTY_FORCE_CHECK2( _expr , file , line , func )           \
  do {                                                                  \
    intptr_t expr = (_expr);                                            \
    if( !expr ){                                                        \
      std::cout << '\n'                                                 \
                << "---[ Error ]--------------------------------------------\n" \
                << "    File     : " << file << '\n'                    \
                << "    Function : " << func << '\n'                    \
                << "    Line     : " << line << '\n'                    \
                << "========================================================\n"; \
      OCCA_THROW;                                                       \
    }                                                                   \
  } while(0)

#define OCCA_FORCE_CHECK2( _expr , _msg , file , line , func )          \
  do {                                                                  \
    intptr_t expr = (_expr);                                            \
    if( !expr ){                                                        \
      std::cout << '\n'                                                 \
                << "---[ Error ]--------------------------------------------\n" \
                << "    File     : " << file << '\n'                    \
                << "    Function : " << func << '\n'                    \
                << "    Line     : " << line << '\n'                    \
                << "    Error    : " << _msg << '\n'                    \
                << "========================================================\n"; \
      OCCA_THROW;                                                       \
    }                                                                   \
  } while(0)

#define OCCA_EMPTY_FORCE_CHECK( _expr )  OCCA_EMPTY_FORCE_CHECK2( _expr , __FILE__ , __LINE__ , __PRETTY_FUNCTION__)
#define OCCA_FORCE_CHECK( _expr , _msg ) OCCA_FORCE_CHECK2( _expr , _msg , __FILE__ , __LINE__ , __PRETTY_FUNCTION__)

#if OCCA_CHECK_ENABLED
#  define OCCA_EMPTY_CHECK( _expr )  OCCA_EMPTY_FORCE_CHECK( _expr )
#  define OCCA_CHECK( _expr , _msg ) OCCA_FORCE_CHECK( _expr , _msg )
#else
#  define OCCA_EMPTY_CHECK( _expr )
#  define OCCA_CHECK( _expr , _msg )
#endif

#define OCCA_DEFAULT_MEM_BYTE_ALIGN 32

//---[ Compiler ]-------------
#define OCCA_GNU_COMPILER       (1 << 0)
#define OCCA_LLVM_COMPILER      (1 << 1)
#define OCCA_INTEL_COMPILER     (1 << 2)
#define OCCA_PATHSCALE_COMPILER (1 << 3)
#define OCCA_IBM_COMPILER       (1 << 4)
#define OCCA_PGI_COMPILER       (1 << 5)
#define OCCA_HP_COMPILER        (1 << 6)
#define OCCA_VS_COMPILER        (1 << 7)
#define OCCA_CRAY_COMPILER      (1 << 8)
#define OCCA_UNKNOWN_COMPILER   (1 << 9)

#ifndef OCCA_COMPILED_WITH
#if defined(__clang__)
#  define OCCA_COMPILED_WITH OCCA_LLVM_COMPILER
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#  define OCCA_COMPILED_WITH OCCA_INTEL_COMPILER
#elif defined(__GNUC__) || defined(__GNUG__)
#  define OCCA_COMPILED_WITH OCCA_GNU_COMPILER
#elif defined(__HP_cc) || defined(__HP_aCC)
#  define OCCA_COMPILED_WITH OCCA_HP_COMPILER
#elif defined(__IBMC__) || defined(__IBMCPP__)
#  define OCCA_COMPILED_WITH OCCA_IBM_COMPILER
#elif defined(__PGI)
#  define OCCA_COMPILED_WITH OCCA_PGI_COMPILER
#elif defined(_CRAYC)
#  define OCCA_COMPILED_WITH OCCA_CRAY_COMPILER
#elif defined(__PATHSCALE__) || defined(__PATHCC__)
#  define OCCA_COMPILED_WITH OCCA_PATHSCALE_COMPILER
#elif defined(_MSC_VER)
#  define OCCA_COMPILED_WITH OCCA_VS_COMPILER
#else
#  define OCCA_COMPILED_WITH OCCA_UNKNOWN_COMPILER
#endif
#endif

//---[ Vectorization ]--------
#ifdef __MIC__
#  define OCCA_MIC 1
#else
#  define OCCA_MIC 0
#endif

#ifdef __AVX2__
#  define OCCA_AVX2 1
#else
#  define OCCA_AVX2 0
#endif

#ifdef __AVX__
#  define OCCA_AVX 1
#else
#  define OCCA_AVX 0
#endif

#ifdef __SSE4_2__
#  define OCCA_SSE4_2 1
#else
#  define OCCA_SSE4_2 0
#endif

#ifdef __SSE4_1__
#  define OCCA_SSE4_1 1
#else
#  define OCCA_SSE4_1 0
#endif

#ifndef OCCA_SSE4
#  if OCCA_SSE4_1 || OCCA_SSE4_2
#    define OCCA_SSE4 1
#  else
#    define OCCA_SSE4 0
#  endif
#endif

#ifdef __SSE3__
#  define OCCA_SSE3 1
#else
#  define OCCA_SSE3 0
#endif

#ifndef OCCA_SSE3
#  ifdef __SSE3__
#    define OCCA_SSE3 1
#  else
#    define OCCA_SSE3 0
#  endif
#endif

#ifndef OCCA_SSE2
#ifdef __SSE2__
#  define OCCA_SSE2 1
#else
#  define OCCA_SSE2 0
#endif
#endif

#ifndef OCCA_SSE
#ifdef __SSE__
#  define OCCA_SSE 1
#else
#  define OCCA_SSE 0
#endif
#endif

#ifndef OCCA_MMX
#ifdef __MMX__
#  define OCCA_MMX 1
#else
#  define OCCA_MMX 0
#endif
#endif

#ifndef OCCA_VECTOR_SET
#if OCCA_MIC
#  define OCCA_VECTOR_SET "MIC AVX-512"
#elif OCCA_AVX2
#  define OCCA_VECTOR_SET "AVX2"
#elif OCCA_AVX
#  define OCCA_VECTOR_SET "AVX"
#elif OCCA_SSE4
#  define OCCA_VECTOR_SET "SSE4"
#elif OCCA_SSE3
#  define OCCA_VECTOR_SET "SSE3"
#elif OCCA_SSE2
#  define OCCA_VECTOR_SET "SSE2"
#elif OCCA_SSE
#  define OCCA_VECTOR_SET "SSE"
#elif OCCA_MMX
#  define OCCA_VECTOR_SET "MMX"
#else
#  define OCCA_VECTOR_SET "[Vector Instruction Set Not Found]"
#endif
#endif

#ifndef OCCA_SIMD_WIDTH
#if   OCCA_MIC
#  define OCCA_SIMD_WIDTH 16
#elif OCCA_AVX | OCCA_AVX2
#  define OCCA_SIMD_WIDTH 8
#elif OCCA_SSE | OCCA_SSE2 | OCCA_SSE3 | OCCA_SSE4
#  define OCCA_SIMD_WIDTH 4
#elif OCCA_MMX
#  define OCCA_SIMD_WIDTH 2
#else
#  define OCCA_SIMD_WIDTH 1
#endif
#endif
//============================

#define OCCA_MAX_ARGS 50
//======================================


//---[ Base ]---------------------------
#define OCCA_KERNEL_ARG_CONSTRUCTOR(TYPE)         \
  template <>                                     \
  inline kernelArg::kernelArg(const TYPE &arg_){  \
    argc                 = 1;                     \
    args[0].data.TYPE##_ = arg_;                  \
    args[0].size         = sizeof(TYPE);          \
  }

#define OCCA_KERNEL_ARG_CONSTRUCTOR_ALIAS(TYPE, ALIAS)  \
  template <>                                           \
  inline kernelArg::kernelArg(const TYPE &arg_){        \
    argc                  = 1;                          \
    args[0].data.ALIAS##_ = arg_;                       \
    args[0].size          = sizeof(TYPE);               \
  }

#define OCCA_EXTRACT_DATA(MODE, CLASS)                          \
  MODE##CLASS##Data_t &data_ = *((MODE##CLASS##Data_t*) data);
//======================================


//---[ OpenCL ]-------------------------
#if OCCA_CHECK_ENABLED
#  define OCCA_CL_CHECK( _str , _statement ) OCCA_CL_CHECK2( _str , _statement , __FILE__ , __LINE__ )
#  define OCCA_CL_CHECK2( _str , _statement , file , line )             \
  do {                                                                  \
    cl_int _error = _statement;                                         \
    if(_error){                                                         \
      _error = _error < 0  ? _error : -_error;                          \
      _error = _error < 65 ? _error : 15;                               \
                                                                        \
      std::cout << "Error\n"                                            \
                << "    File    : " << file << '\n'                     \
                << "    Line    : " << line << '\n'                     \
                << "    Error   : OpenCL Error [ " << _error << " ]: " << occa::openclError(_error) << '\n' \
                << "    Message : " << _str << '\n';                    \
      OCCA_THROW;                                                       \
    }                                                                   \
  } while(0)

#else
#  define OCCA_CL_CHECK( _str , _statement ) do { _statement; } while(0)
#endif
//======================================


//---[ CUDA ]---------------------------
#if OCCA_CHECK_ENABLED
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
      OCCA_THROW;                                                       \
    }                                                                   \
  } while(0)
#else
#  define OCCA_CUDA_CHECK( _str , _statement ) do { _statement; } while(0)
#endif
//======================================

//---[ HSA ]----------------------------
#if OCCA_CHECK_ENABLED
#  define OCCA_HSA_CHECK( _str , _statement ) OCCA_HSA_CHECK2( _str , _statement , __FILE__ , __LINE__ )
#  define OCCA_HSA_CHECK2( _str , _statement , file , line )            \
  do {                                                                  \
    hsa_status_t _error = _statement;                                   \
    if(_error != HSA_STATUS_SUCCESS){					\
      std::cout << "Error\n"                                            \
                << "    File    : " << file << '\n'                     \
                << "    Line    : " << line << '\n'                     \
                << "    Error   : HSA Error [ " << _error << " ]: " << occa::hsaError(_error) << '\n' \
                << "    Message : " << _str << '\n';                    \
      OCCA_THROW;                                                       \
    }                                                                   \
  } while(0)

#else
#  define OCCA_CL_CHECK( _str , _statement ) do { _statement; } while(0)
#endif
//======================================

#endif
