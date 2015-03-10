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

#if (OCCA_OS & OSX_OS)
#  define OCCA_MEM_ALIGN 16
#else
#  define OCCA_MEM_ALIGN 16
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

#ifdef __SSSE3__
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

#ifdef __SSE2__
#  define OCCA_SSE2 1
#else
#  define OCCA_SSE2 0
#endif

#ifdef __SSE__
#  define OCCA_SSE 1
#else
#  define OCCA_SSE 0
#endif

#ifdef __MMX__
#  define OCCA_MMX 1
#else
#  define OCCA_MMX 0
#endif

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

#if   OCCA_MIC
#  define OCCA_SIMD_WIDTH 16
#elif OCCA_AVX
#  define OCCA_SIMD_WIDTH 8
#elif OCCA_SSE
#  define OCCA_SIMD_WIDTH 4
#elif OCCA_MMX
#  define OCCA_SIMD_WIDTH 2
#else
#  define OCCA_SIMD_WIDTH 1
#endif
//============================

#define OCCA_MAX_ARGS 50
//======================================


//---[ Base ]---------------------------
#define OCCA_KERNEL_ARG_CONSTRUCTOR(TYPE)         \
  template <>                                     \
  inline kernelArg::kernelArg(const TYPE &arg_){  \
    dHandle = NULL;                               \
    mHandle = NULL;                               \
                                                  \
    arg.TYPE##_ = arg_;                           \
    size      = sizeof(TYPE);                     \
                                                  \
    pointer    = false;                           \
    hasTwoArgs = false;                           \
  }

#define OCCA_KERNEL_ARG_CONSTRUCTOR_ALIAS(TYPE, ALIAS)  \
  template <>                                           \
  inline kernelArg::kernelArg(const TYPE &arg_){        \
    dHandle = NULL;                                     \
    mHandle = NULL;                                     \
                                                        \
    arg.ALIAS##_ = arg_;                                \
    size         = sizeof(TYPE);                        \
                                                        \
    pointer    = false;                                 \
    hasTwoArgs = false;                                 \
  }

#define OCCA_EXTRACT_DATA(MODE, CLASS)                          \
  MODE##CLASS##Data_t &data_ = *((MODE##CLASS##Data_t*) data);
//======================================


//---[ OpenCL ]-------------------------
#if OCCA_CHECK_ENABLED
;
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


//---[ COI ]----------------------------
#if OCCA_CHECK_ENABLED
#  define OCCA_COI_CHECK( _str , _statement ) OCCA_COI_CHECK2( _str , _statement , __FILE__ , __LINE__ )
#  define OCCA_COI_CHECK2( _str , _statement , file , line )            \
  do {                                                                  \
    COIRESULT errorCode = _statement;                                   \
    if(errorCode != COI_SUCCESS){                                       \
      std::cout << "Error\n"                                            \
                << "    File    : " << file << '\n'                     \
                << "    Line    : " << line << '\n'                     \
                << "    Error   : " << occa::coiError(errorCode) << '\n' \
                << "    Message : " << _str << '\n';                    \
      throw 1;                                                          \
    }                                                                   \
  } while(0);
#else
#  define OCCA_COI_CHECK( _str , _statement ) do { _statement; } while(0);
#endif
//======================================
#endif
