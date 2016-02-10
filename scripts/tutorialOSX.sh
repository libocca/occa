
# set up OCCA env
export OCCA_DIR=`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OCCA_DIR}/lib

# build library
make clean
CC=gcc-mp-4.9 CXX=g++-mp-4.9  OCCA_FORTRAN_ENABLED=0 FC=gfortran make -j 16

# build example
cd examples
cd addVectors
cd cpp
make clean
CC=gcc-mp-4.9 CXX=g++-mp-4.9 OCCA_FORTRAN_ENABLED=0 FC=gfortran make -j 16
