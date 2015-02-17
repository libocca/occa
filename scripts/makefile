#---[ Paths ]-------------------------------------
bPath = bin
oPath = obj
sPath = src
iPath = include
#=================================================


#---[ Default Variables ]-------------------------
debugEnabled   = 0
checkEnabled   = 1

# CXX      : C++ Compiler
# CXXFLAGS : C++ Compiler Flags

# FC       : Fortran Compiler
# FCFLAGS  : Fortran Compiler Flags

# OCCA_INCLUDE_PATH : Extra include paths
# OCCA_LIBRARY_PATH : Extra library paths
#=================================================


#---[ OS Detection ]------------------------------
LINUX_OS   = 1
OSX_OS     = 2
WINDOWS_OS = 4

flags = -D LINUX_OS=$(LINUX_OS) -D OSX_OS=$(OSX_OS) -D WINDOWS_OS=$(WINDOWS_OS)

ifdef SystemRoot
	OS = WINDOWS
else ifeq ($(shell uname -s),Darwin)
	OS = OSX
else
	OS = LINUX
endif
#=================================================


#---[ Variables ]---------------------------------
ifdef OCCA_DEVELOPER
 ifeq ($(OCCA_DEVELOPER), 1)
  ifeq ($(DEBUG), 0)
     debugEnabled = 0
  else
     debugEnabled = 1
  endif
 else
  ifeq ($(DEBUG), 1)
     debugEnabled = 1
  else
     debugEnabled = 0
  endif
 endif
endif

ifndef CXX
  ifdef OCCA_CXX
    CXX = ${OCCA_CXX}
  else
    CXX = g++
  endif
endif

ifndef FC
  FC = gfortran
endif

ifndef CXXFLAGS
  ifeq ($(DEBUG),1)
    CXXFLAGS = -g
  else
    CXXFLAGS = -O3
  endif
endif

ifndef FCFLAGS
  ifeq ($(DEBUG),1)
    FCFLAGS = -g
  else
    FCFLAGS = -O3
  endif
endif

compiler      = $(CXX)
compilerFlags = $(CXXFLAGS)

fCompiler      = $(FC)
fCompilerFlags = $(FCFLAGS)
#=================================================


#---[ Paths/Flags/Links ]-------------------------
paths = -I${OCCA_DIR}/lib -I${OCCA_DIR}/include -L${OCCA_DIR}/lib
paths += $(foreach path, $(subst :, ,$(OCCA_INCLUDE_PATH)), -I$(path))
paths += $(foreach path, $(subst :, ,$(OCCA_LIBRARY_PATH)), -L$(path))

ifneq (,$(wildcard ./$(iPath)/*))
  paths += -I./$(iPath)
endif

links = -locca
#=================================================


#---[ C++ Compiler Info ]-------------------------
# C/C++ Compiler
ifneq (,$(findstring $(compiler),g++))
  cCompilerType = GCC
else ifneq (,$(findstring $(compiler),gcc))
  cCompilerType = GCC
else ifneq (,$(findstring $(compiler),clang))
  cCompilerType = CLANG
else ifneq (,$(findstring $(compiler),icc))
  cCompilerType = INTEL
else ifneq (,$(findstring $(compiler),icpc))
  cCompilerType = INTEL
else ifneq (,$(findstring $(compiler),xlc))
  cCompilerType = IBM
else ifneq (,$(findstring $(compiler),pgcc))
  cCompilerType = PGI
else ifneq (,$(findstring $(compiler),pgc++))
  cCompilerType = PGI
else ifneq (,$(findstring $(compiler),cc))
  cCompilerType = CRAY
else ifneq (,$(findstring $(compiler),CC))
  cCompilerType = CRAY
else ifneq (,$(findstring $(compiler),pathcc))
  cCompilerType = PATHSCALE
else ifneq (,$(findstring $(compiler),pathCC))
  cCompilerType = PATHSCALE
else ifneq (,$(findstring $(compiler),cl.exe))
  cCompilerType = VS
else
  cCompilerType = NOT_FOUND
endif

# Debug and Release Flags
ifneq (,$(filter $(cCompilerType),GCC CLANG))
  debugFlags   = -g
  releaseFlags = -O3 -D __extern_always_inline=inline
  Shared_Flags
else ifneq (,$(filter $(cCompilerType),INTEL))
  debugFlags   = -g
  releaseFlags = -O3 -xHost
else ifneq (,$(filter $(cCompilerType),CRAY))
  debugFlags   = -g
  releaseFlags = -O3 -h intrinsics -fast
else ifneq (,$(filter $(cCompilerType),IBM))
  debugFlags   = -g
  releaseFlags = -O3 -qhot=simd
else ifneq (,$(filter $(cCompilerType),PGI))
  debugFlags   = -g
  releaseFlags = -O3 -fast -Mipa=fast,inline -Msmartalloc
else ifneq (,$(filter $(cCompilerType),PATHSCALE))
  debugFlags   = -g
  releaseFlags = -O3 -march=auto
else ifneq (,$(filter $(cCompilerType),VS))
  debugFlags   =      # [-] Missing
  releaseFlags =      # [-] Missing
endif
#=================================================


#---[ Fortran Compiler Info ]---------------------
ifneq (,$(findstring $(fCompiler),gfortran))
  fCompilerType = GCC
else ifneq (,$(findstring $(fCompiler),ifort))
  fCompilerType = INTEL
else ifneq (,$(findstring $(fCompiler),ftn))
  fCompilerType = CRAY
else ifneq (,$(findstring $(fCompiler),xlf))
  fCompilerType = IBM
else ifneq (,$(findstring $(fCompiler),pgfortran))
  fCompilerType = PGI
else ifneq (,$(findstring $(fCompiler),pathf95))
  fCompilerType = PATHSCALE
else
  fCompilerType = NOT_FOUND
endif

# Fortran Compiler
ifneq (,$(filter $(fCompilerType),GCC))
  fCompilerFlags =      # [-] Missing
  fModDirFlag    = -J
else ifneq (,$(filter $(fCompilerType),INTEL))
  fCompilerFlags =      # [-] Missing
  fModDirFlag    = -module
else ifneq (,$(filter $(fCompilerType),CRAY))
  fCompilerFlags =      # [-] Missing
  fModDirFlag    = -J
else ifneq (,$(filter $(fCompilerType),IBM))
  fCompilerFlags =      # [-] Missing
  fModDirFlag    = -qmoddir
else ifneq (,$(filter $(fCompilerType),PGI))
  fCompilerFlags =      # [-] Missing
  fModDirFlag    = -module
else ifneq (,$(filter $(fCompilerType),PATHSCALE))
  fCompilerFlags =      # [-] Missing
  fModDirFlag    = -module
endif
#=================================================


#---[ Flags and Libraries ]-----------------------
ifeq ($(OS),OSX)
  flags      += -D OCCA_OS=OSX_OS -Wno-deprecated-declarations
  openclFlags =
  cudaFlags   =
  coiFlags    =

  links      += -framework accelerate -framework CoreServices
  openclLinks = -framework OpenCL
  cudaLinks   = -framework CUDA
  coiLinks    =
else ifeq ($(OS), LINUX)
  flags       += -D OCCA_OS=LINUX_OS
  openclFlags  =
  cudaFlags    =
  coiFlags     =

  links       += -lm -lrt -ldl
  openclLinks  = -lOpenCL
  cudaLinks    = -lcuda
  coiLinks     = -lcoi_host
else ifeq ($(OS), WINDOWS)
  flags       += -D OCCA_OS=WINDOWS_OS
  openclFlags  =
  cudaFlags    =
  coiFlags     =

  links       +=
  openclLinks  =
  cudaLinks    =
  coiLinks     =
endif

findInLdConfig = $(shell ldconfig -p | grep --quiet $(1) && echo 1 || echo 0)
findInDydlPath = $(shell for dydir in ${DYLD_LIBRARY_PATH//:/ }; do if ls $dydir/$(1) >/dev/null 2>&1; then echo 1; fi; done)
findFramework  = $(shell if ls /System/Library/Frameworks/$(1).framework >/dev/null 2>&1; then echo 1; fi)

findLinuxLibrary = $(call findInLdConfig, $(1))
findOSXLibrary   = $(or, $(call findInDydlPath, $(1)), $(call findFramework, $(2)))
#=================================================


#---[ Variable Dependencies ]---------------------
ifdef OCCA_OPENCL_ENABLED
  openclEnabled = $(OCCA_OPENCL_ENABLED)
else
  ifeq ($(OS), OSX)
    openclEnabled = $(call findFramework,OpenCL)
  else ifeq ($(OS), LINUX)
    openclEnabled = $(call findLinuxLibrary,libOpenCL.so)
  endif
endif

ifdef OCCA_CUDA_ENABLED
  cudaEnabled = $(OCCA_CUDA_ENABLED)
else
  ifeq ($(OS), OSX)
    cudaEnabled = $(call findOsxLibrary,CUDA,libcuda.so)
  else ifeq ($(OS), LINUX)
    cudaEnabled = $(call findLinuxLibrary,libcuda.so)
  endif
endif

ifdef OCCA_COI_ENABLED
  coiEnabled = $(OCCA_COI_ENABLED)
else ifeq ($(OS), LINUX)
  coiEnabled = $(call findLinuxLibrary,coi_host.so)
endif

ifeq ($(debugEnabled), 1)
  flags += $(debugFlags)   -DOCCA_DEBUG_ENABLED=1
else
  flags += $(releaseFlags) -DOCCA_DEBUG_ENABLED=0 -DNDEBUG=1 -DOCCA_SHOW_WARNINGS=0
endif

ifeq ($(checkEnabled), 1)
  flags += -DOCCA_CHECK_ENABLED=1
else
  flags += -DOCCA_CHECK_ENABLED=0
endif

ifeq ($(openclEnabled), 1)
  flags += $(openclFlags) -DOCCA_OPENCL_ENABLED=1
  links += $(openclLinks)
else
  flags +=                 -DOCCA_OPENCL_ENABLED=0
endif


ifeq ($(cudaEnabled), 1)
  flags += $(cudaFlags) -DOCCA_CUDA_ENABLED=1
  links += $(cudaLinks)
else
  flags +=               -DOCCA_CUDA_ENABLED=0
endif

ifeq ($(coiEnabled), 1)
  flags += $(coiFlags) -DOCCA_COI_ENABLED=1
  links += $(coiLinks)
else
  flags += -DOCCA_COI_ENABLED=0
endif

ifdef COMPILING_WITH_JULIA
  flags += -DOCCA_COMPILED_FOR_JULIA=1
endif
#=================================================
