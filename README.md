OCCA2 [![Build Status](https://travis-ci.org/tcew/OCCA2.svg?branch=master)](https://travis-ci.org/tcew/OCCA2)
=====

OCCA 2.0: extensible multi-threading programming API.

See web-page: http://libocca.org

See white paper on OCCA 1.0 here: http://arxiv.org/abs/1403.0968

See tutorial slides here: http://github.com/tcew/OCCA2Tutorial

```
+---[ (0) README ]------------------------------------------
|   Installing:
|      Using a terminal, go to your OCCA directory
|         You should see: this README, include, src, lib
|         Set OCCA_DIR (check (2) below)
|         Type "make" to compile libocca
|
|   Running Examples:  (After compiling libocca)
|      Setup your LD_LIBRARY PATH (check (3) below)
|      cd examples/addVectors
|      make
|      ./main (C++), ./main_c (C), python main.py, etc
|
|   Further options can be seen in (4-7)
|
|   OS Status:
|      Linux and OSX are fully supported
|      Windows is partially supported
|        - Code is up-to-date for windows
|        - Missing compilation project/scripts
|        - Visual Studio project is out of date
|
|  OKL Status:
|    Supports most of C (send bugs =))
|    Preprocessor is missing variadic functions
|
|  OFL Status:
|    Currently only supports a subset of Fortran:
|       - integer, real, character, logical, double precision
|       - function, subroutine
|       - DO, WHILE, IF, IF ELSE, ELSE
|
+===========================================================


+---[ (1) ENVIRONMENT VARIABLES ]---------------------------
|
|    (2) OCCA_DIR (Required!)
|
|    (3) Useful environment variables:
|            LD_LIBRARY_PATH
|            OCCA_CACHE_DIR
|            OCCA_INCLUDE_PATH
|            OCCA_LIBRARY_PATH
|            OCCA_CXX
|            OCCA_CXXFLAGS
|
|    (4) Pthreads Options
|            OCCA_PTHREADS_COUNT
|
|    (5) OpenCL Options
|            OCCA_OPENCL_COMPILER_FLAGS
|
|    (6) CUDA Options
|            OCCA_CUDA_COMPILER, OCCA_CUDA_COMPILER_FLAGS
|
|    (7) COI Options
|            OCCA_COI_COMPILER, OCCA_COI_COMPILER_FLAGS
|
|    (A) Running OCCA on Matlab
|
|    (B) Running OCCA on Fortran
+===========================================================



+---[ (2) OCCA_DIR (Required!) ]----------------------------
|  Info:
|    Absolute path to the OCCA2, used to find:
|      $OCCA_DIR/include
|      $OCCA_DIR/src
|      $OCCA_DIR/lib
|
|  Setting it:
|    export OCCA_DIR=/absolute/path/to/dir
+===========================================================


+---[ (3) Useful environment variables ]--------------------
|
|     +---[ LD_LIBRARY_PATH ]---------------------
|     |  Info:
|     |    Append directory where libocca is stored:
|     |      $OCCA_DIR/lib
|     |    to link libocca to your application
|     |
|     |  Setting it:
|     |    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH":${OCCA_DIR}/lib"
|     +===========================================
|
|     +---[ OCCA_CACHE_DIR ]----------------------
|     |  Info:
|     |    Directory where compile kernels are
|     |      stored (caching binaries). This defaults
|     |      to:
|     |    ~/._occa
|     |
|     |  Setting it:
|     |    export OCCA_CACHE_DIR=/absolute/path/to/dir
|     +===========================================
|
|     +---[ OCCA_INCLUDE_PATH ]-------------------
|     |  Info:
|     |    Append paths for OCCA-mode headers, such as:
|     |      /path/to/CL     (for #include <CL/cl.h>)
|     |      /path/to/cuda.h (for #include <cuda.h>)
|     |
|     |  Setting it:
|     |    export OCCA_INCLUDE_PATH="/path/to/CL:/path/to/cuda.h"
|     +===========================================
|
|     +---[ OCCA_LIBRARY_PATH ]-------------------
|     |  Info:
|     |    Append paths for OCCA-mode libraries, such as:
|     |      /path/to/libOpenCL
|     |      /path/to/libcuda
|     |
|     |  Setting it:
|     |    export OCCA_LIBRARY_PATH="/path/to/libOpenCL:/path/to/libcuda"
|     +===========================================
|
|     +---[ OCCA_CXX ]----------------------------
|     |  Info:
|     |    C++ compiler used at runtime
|     |
|     |  Setting it:
|     |    export OCCA_CXX=clang++
|     +===========================================
|
|     +---[ OCCA_CXX_FLAGS ]----------------------
|     |  Info:
|     |    C++ compiler flags used at runtime
|     |
|     |  Setting it:
|     |    export OCCA_CXXFLAGS=-g
|     +===========================================
+===========================================================


+---[ (4) Pthreads Options ]--------------------------------
|  Info:
|    OCCA_PTHREAD_COUNT : Threads spawned at each kernel
|
|  Setting it:
|    export OCCA_PTHREAD_COUNT="8"
+===========================================================


+---[ (5) OpenCL Options ]----------------------------------
|  Info:
|    OCCA_OPENCL_COMPILER_FLAGS: OpenCL Compiler Flags
|
|  Setting it:
|    export OCCA_OPENCL_COMPILER_FLAGS="-cl-mad-enable -cl-finite-math-only"
+===========================================================


+---[ (6) CUDA Options ]------------------------------------
|  Info:
|    OCCA_CUDA_COMPILER      : CUDA Compiler
|    OCCA_CUDA_COMPILER_FLAGS: CUDA Compiler Flags
|
|  Setting it:
|    export OCCA_CUDA_COMPILER="nvcc"
|    export OCCA_CUDA_COMPILER_FLAGS="-O3"
+===========================================================


+---[ (7) COI Options ]-------------------------------------
|  Info:
|    OCCA_COI_COMPILER      : COI Compiler
|    OCCA_COI_COMPILER_FLAGS: COI Compiler Flags
|
|  Setting it:
|    export OCCA_COI_COMPILER="icpc"
|    export OCCA_COI_COMPILER_FLAGS="-O3"
+===========================================================


+--[ (A) MATLAB ]-------------------------------------------
|
|  path(strcat(getenv('OCCA_DIR'), '/lib'), path)
|
+===========================================================


+--[ (B) Fortran ]-------------------------------------------
|
|  Info:
|    OCCA_FORTRAN_ENABLED: Build Fortran wrappers
|    FC                  : Fortran Compiler
|    FCFLAGS             : Fortran Compiler Flags
|
|  Setting it:
|    export OCCA_FORTRAN_ENABLED="1"
|    export FC="gfortran"
|    export FCFLAGS="-O3 -wAll"
|
+===========================================================
```