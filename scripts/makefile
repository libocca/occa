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


#---[ Shell Tools ]-------------------------------
ifeq (,$(findstring bash,${SHELL}))
  ifneq (,$(wildcard /bin/bash))
    SHELL := /bin/bash
  else
    $(error /bin/bash not found, set SHELL environment variable to compile [export SHELL=/path/to/bash])
  endif
endif

libraryFlagsFor = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; libraryFlags $1)
includeFlagsFor = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; headerFlags  $1)

compilerReleaseFlags      = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerReleaseFlags      $(compiler))
compilerDebugFlags        = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerDebugFlags        $(compiler))
compilerSharedBinaryFlags = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerSharedBinaryFlags $(compiler))

compilerSupportsOpenMP = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerSupportsOpenMP $(compiler))
compilerOpenMPFlags    = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerOpenMPFlags    $(compiler))
#=================================================


#---[ C++ Compiler Info ]-------------------------
debugFlags        = $(call compilerDebugFlags)
releaseFlags      = $(call compilerReleaseFlags)
sharedBinaryFlags = $(call compilerSharedBinaryFlags)
#=================================================


#---[ Fortran Compiler Info ]---------------------
ifneq (,$(findstring gfortran,$(fCompiler)))
  fCompilerType = GCC
else ifneq (,$(findstring ifort,$(fCompiler)))
  fCompilerType = INTEL
else ifneq (,$(findstring ftn,$(fCompiler)))
  fCompilerType = CRAY
else ifneq (,$(findstring xlf,$(fCompiler)))
  fCompilerType = IBM
else ifneq (,$(findstring pgfortran,$(fCompiler)))
  fCompilerType = PGI
else ifneq (,$(findstring pathf95,$(fCompiler)))
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
  flags += -D OCCA_OS=OSX_OS -Wno-deprecated-declarations
  links += -framework accelerate -framework CoreServices
else ifeq ($(OS), LINUX)
  flags += -D OCCA_OS=LINUX_OS
  links += -lm -lrt -ldl
else ifeq ($(OS), WINDOWS)
  flags += -D OCCA_OS=WINDOWS_OS
  links +=
endif
#=================================================


#---[ Variable Dependencies ]---------------------
openmpEnabled = 0
openclEnabled = 0
cudaEnabled   = 0
coiEnabled    = 0

ifdef OCCA_OPENMP_ENABLED
  openmpEnabled = $(OCCA_OPENMP_ENABLED)
else
  openmpEnabled = $(call compilerSupportsOpenMP)
  flags        += $(call compilerOpenMPFlags)
endif

ifdef OCCA_OPENCL_ENABLED
  openclEnabled = $(OCCA_OPENCL_ENABLED)

  ifeq ($(openclEnabled), 1)
    ifeq ($(OS), OSX)
      links += -framework OpenCL
    else ifeq ($(OS), LINUX)
      links += -lOpenCL
    endif
  endif
else
  ifeq ($(OS), OSX)
    # OpenCL includes are embedded in the framework
    openclLibFlags = $(call libraryFlagsFor,OpenCL)

    ifneq (,$(openclLibFlags))
      openclEnabled = 1
      links += $(openclLibFlags)
    endif
  else ifeq ($(OS), LINUX)
    openclLibFlags = $(call libraryFlagsFor,OpenCL)
    ifneq (,$(openclLibFlags))

      openclIncFlags = $(call includeFlagsFor,CL/cl.h)
      ifneq (,$(openclIncFlags))
        openclEnabled = 1
        paths += $(openclIncFlags)
        links += $(openclLibFlags)
      endif
    endif
  endif
endif

ifdef OCCA_CUDA_ENABLED
  cudaEnabled = $(OCCA_CUDA_ENABLED)

  ifeq ($(cudaEnabled), 1)
    ifeq ($(OS), OSX)
      links += -framework CUDA
    else ifeq ($(OS), LINUX)
      links += -lcuda
    endif
  endif
else

  cudaIncFlags = $(call includeFlagsFor,cuda.h)

  ifneq (,$(cudaIncFlags))

    ifeq ($(OS), OSX)
      cudaLibFlags = $(call libraryFlagsFor,CUDA)
    else ifeq ($(OS), LINUX)
      cudaLibFlags = $(call libraryFlagsFor,cuda)
    endif

    ifneq (,$(cudaLibFlags))
      cudaEnabled = 1
      paths += $(cudaIncFlags)
      links += $(cudaLibFlags)
    endif
  endif
endif

ifdef OCCA_COI_ENABLED
  coiEnabled = $(OCCA_COI_ENABLED)

  ifeq ($(coiEnabled), 1)
    ifeq ($(OS), LINUX)
      links += -lcoi_host
    endif
  endif
else
  ifeq ($(OS), LINUX)
    coiLibFlags = $(call libraryFlagsFor,coi_host)
    ifneq (,$(coiLibFlags))

      coiIncFlags = $(call includeFlagsFor,intel-coi/source/COIProcess_source.h)
      ifneq (,$(coiIncFlags))
        coiEnabled = 1
        paths += $(coiIncFlags)
        links += $(coiLibFlags)
      endif
    endif
  endif
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

flags += -DOCCA_OPENMP_ENABLED=$(openmpEnabled)
flags += -DOCCA_OPENCL_ENABLED=$(openclEnabled)
flags += -DOCCA_CUDA_ENABLED=$(cudaEnabled)
flags += -DOCCA_COI_ENABLED=$(coiEnabled)

ifdef COMPILING_WITH_JULIA
  flags += -DOCCA_COMPILED_FOR_JULIA=1
endif
#=================================================
