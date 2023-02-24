# !/bin/bash
export OCCA_CC=clang
export OCCA_CXX=clang++
export SYCL_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda"

export OCCA_DPCPP_COMPILER=clang++
export OCCA_DPCPP_COMPILER_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda"

# Override default values here
CC=clang
CXX=clang++
#FC=

# Default build parameters
: ${BUILD_DIR:=`pwd`/build}
: ${INSTALL_DIR:=`pwd`/install}
: ${BUILD_TYPE:="RelWithDebInfo"}

: ${CC:="gcc"}
: ${CXX:="g++"}
: ${FC:="gfortran"}

: ${ENABLE_DPCPP:="ON"}
: ${ENABLE_OPENCL:="OFF"}
: ${ENABLE_CUDA:="ON"}
: ${ENABLE_HIP="OFF"}
: ${ENABLE_OPENMP="OFF"}
: ${ENABLE_METAL="OFF"}
: ${ENABLE_FORTRAN="OFF"}
: ${ENABLE_TESTS="ON"}
: ${ENABLE_EXAMPLES="ON"}

cmake -S . -B ${BUILD_DIR} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DCMAKE_Fortran_COMPILER=${FC} \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
  -DCMAKE_C_FLAGS="${CFLAGS}" \
  -DCMAKE_Fortran_FLAGS="${FFLAGS}" \
  -DENABLE_OPENMP=${ENABLE_OPENMP} \
  -DENABLE_OPENCL=${ENABLE_OPENCL} \
  -DENABLE_DPCPP=${ENABLE_DPCPP} \
  -DENABLE_CUDA=${ENABLE_CUDA} \
  -DENABLE_HIP=${ENABLE_HIP} \
  -DENABLE_METAL=${ENABLE_METAL} \
  -DENABLE_FORTRAN=${ENABLE_FORTRAN} \
  -DENABLE_TESTS=${ENABLE_TESTS} \
  -DENABLE_EXAMPLES=${ENABLE_EXAMPLES}
