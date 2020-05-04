rmSlash = $(patsubst %/,%,$(1))

PROJ_DIR := $(call rmSlash,$(PROJ_DIR))

ifdef OCCA_DIR
  OCCA_DIR := $(call rmSlash,${OCCA_DIR})
endif

ifeq ($(strip $(OCCA_DIR)),)
  OCCA_DIR := $(call rmSlash,$(abspath $(dir $(abspath $(lastword $(MAKEFILE_LIST))))/..))
endif

binPath  = $(PROJ_DIR)/bin
libPath  = $(PROJ_DIR)/lib
objPath  = $(PROJ_DIR)/obj
srcPath  = $(PROJ_DIR)/src
incPath  = $(PROJ_DIR)/include
testPath = $(PROJ_DIR)/tests


#---[ Default Variables ]-------------------------
debugEnabled = 0
checkEnabled = 1

ifeq ($(OCCA_DEVELOPER),1)
  ifeq ($(DEBUG),0)
    debugEnabled = 0
  else
    debugEnabled = 1
  endif
else
  ifeq ($(DEBUG),1)
    debugEnabled = 1
  else
    debugEnabled = 0
  endif
endif

flags =

# CXX      : C++ Compiler
# CXXFLAGS : C++ Compiler Flags

# OCCA_INCLUDE_PATH : Extra include paths
# OCCA_LIBRARY_PATH : Extra library paths

OCCA_COVERAGE ?= 0
#=================================================


#---[ OS Detection ]------------------------------
OCCA_LINUX_OS   = 1
OCCA_MACOS_OS   = 2
OCCA_WINDOWS_OS = 4
OCCA_WINUX_OS   = 5 # (OCCA_LINUX_OS | OCCA_WINDOWS_OS)

usingLinux   = 0
usingMacOS   = 0
usingWinux   = 0
usingWindows = 0

UNAME := $(shell uname)

ifeq ($(UNAME),Linux)
  usingLinux   = 1
else ifeq ($(UNAME),Darwin)
  usingMacOS   = 1
else ifneq (,$(findstring CYGWIN,$(UNAME)))
  usingLinux   = 1
  usingWinux   = 1
else ifneq (,$(findstring MINGW,$(UNAME)))
  usingWinux   = 1
  usingWindows = 1
else
  usingWindows = 1
endif
#=================================================


#---[ Variables ]---------------------------------
ifndef CXX
  ifdef OCCA_CXX
    CXX = ${OCCA_CXX}
  else
    CXX = g++
  endif
endif

ifndef CXXFLAGS
  ifeq ($(debugEnabled),1)
    compilerFlags = $(debugFlags)
  else
    compilerFlags = $(releaseFlags)
  endif
else
  compilerFlags = $(CXXFLAGS)
endif

CC ?= gcc

ifndef CFLAGS
  ifeq ($(debugEnabled),1)
    cCompilerFlags = $(debugFlags)
  else
    cCompilerFlags = $(releaseFlags)
  endif
else
  cCompilerFlags = $(CFLAGS)
endif

ifeq ($(OCCA_COVERAGE),1)
  coverageFlags = --coverage -fno-inline -fno-inline-small-functions -fno-default-inline -Wno-ignored-optimization-argument
  compilerFlags += $(coverageFlags)
  cCompilerFlags += $(coverageFlags)
endif

compiler  = $(CXX)
cCompiler = $(CC)

linkerFlags = $(LDFLAGS)
#=================================================


#---[ Paths/Flags ]-------------------------------
# OCCA (global) include and library paths
paths  = -I$(OCCA_DIR)/include
paths += -L$(OCCA_DIR)/lib

# Extra (user-supplied) include and library paths
paths += $(foreach path, $(subst :, ,$(OCCA_INCLUDE_PATH)), -I$(path))
paths += $(foreach path, $(subst :, ,$(OCCA_LIBRARY_PATH)), -L$(path))

# Project (local) include path
ifneq (,$(strip $(wildcard $(incPath)/*)))
  paths += -I$(incPath)
endif

linkerFlags += -locca
#=================================================


#---[ Shell Tools ]-------------------------------
ifeq (,$(findstring bash,$(SHELL)))
  SHELL := $(shell which bash)
  ifeq (,$(SHELL))
    $(error Could not find [bash], set SHELL manually with [export SHELL=/path/to/bash] or compile with [make SHELL=/path/to/bash])
  endif
endif

libraryFlagsFor = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; libraryFlags "$1")
includeFlagsFor = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; headerFlags  "$1")

compilerVendor       = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; compilerVendor       "$(compiler)")
compilerReleaseFlags = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; compilerReleaseFlags "$(compiler)")
compilerDebugFlags   = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; compilerDebugFlags   "$(compiler)")
compilerCpp11Flags   = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; compilerCpp11Flags   "$(compiler)")
compilerPicFlag      = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; compilerPicFlag      "$(compiler)")
compilerSharedFlag   = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; compilerSharedFlag   "$(compiler)")
compilerPthreadFlag  = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; compilerPthreadFlag  "$(compiler)")

compilerSupportsMPI    = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; compilerSupportsMPI    "$(compiler)")
compilerSupportsOpenMP = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; compilerSupportsOpenMP "$(compiler)")
compilerOpenMPFlag     = $(shell . $(OCCA_DIR)/include/occa/scripts/shellTools.sh; compilerOpenMPFlag     "$(compiler)")
#=================================================


#---[ Compiler Info ]-----------------------------
vendor       = $(call compilerVendor)
debugFlags   = $(call compilerDebugFlags)
releaseFlags = $(call compilerReleaseFlags)
cpp11Flags   = $(call compilerCpp11Flags)
picFlag      = $(call compilerPicFlag)
sharedFlag   = $(call compilerSharedFlag)
pthreadFlag  = $(call compilerPthreadFlag)

# Workaround for GitHub Actions CI tests on Ubuntu using GCC.
ifdef GITHUB_ACTIONS
  ifeq ($(usingLinux),1)
    ifeq ($(vendor),GCC)
      releaseFlags := $(filter-out -march=native,$(releaseFlags))
    endif
  endif
endif
#=================================================


#---[ Flags and Libraries ]-----------------------
OCCA_USING_VS := 0
OCCA_UNSAFE   ?= 0
soNameFlag=
soExt=

ifeq ($(usingLinux),1)

  ifeq ($(usingWinux),0)
    OCCA_OS := OCCA_LINUX_OS
  else
    OCCA_OS := OCCA_WINUX_OS
  endif

  linkerFlags += -lm -lrt -ldl
  soNameFlag = -Wl,-soname,libocca.so
  soExt = so

else ifeq ($(usingMacOS),1)

  OCCA_OS := OCCA_MACOS_OS
  flags += -Wno-deprecated-declarations
  linkerFlags += -framework accelerate -framework CoreServices
  soNameFlag = -Wl,-install_name,@rpath/libocca.dylib
  soExt = dylib

else ifeq ($(usingWindows),1)

  ifeq ($(usingWinux),0)
    OCCA_OS := OCCA_WINDOWS_OS
    OCCA_USING_VS := 1
  else
    OCCA_OS := OCCA_WINDOWS_OS
  endif

  linkerFlags +=
  soExt = so

endif
#=================================================


#---[ Variable Dependencies ]---------------------
mpiEnabled    = 0
openmpEnabled = 0
cudaEnabled   = 0
hipEnabled    = 0
openclEnabled = 0
metalEnabled  = 0


#---[ MPI ]-----------------------------
ifdef OCCA_MPI_ENABLED
  mpiEnabled = $(OCCA_MPI_ENABLED)
else
  mpiEnabled = $(call compilerSupportsMPI)
endif


#---[ OpenMP ]--------------------------
ifdef OCCA_OPENMP_ENABLED
  openmpEnabled  = $(OCCA_OPENMP_ENABLED)
else
  openmpEnabled  = $(call compilerSupportsOpenMP)
endif
ifeq ($(openmpEnabled),1)
  flags += $(call compilerOpenMPFlag)
endif


#---[ CUDA ]----------------------------
ifdef OCCA_CUDA_ENABLED
  cudaEnabled = $(OCCA_CUDA_ENABLED)

  ifeq ($(cudaEnabled),1)
    ifeq ($(usingLinux),1)
      linkerFlags += -lcuda
    else ifeq ($(usingMacOS),1)
      linkerFlags += -framework CUDA
    endif
  endif
else
  cudaIncFlags = $(call includeFlagsFor,cuda.h)

  ifneq (,$(cudaIncFlags))

    ifeq ($(usingLinux),1)
      cudaLibFlags = $(call libraryFlagsFor,cuda)
    else ifeq ($(usingMacOS),1)
      cudaLibFlags = $(call libraryFlagsFor,CUDA)
    endif

    ifneq (,$(cudaLibFlags))
      cudaEnabled = 1
      paths += $(cudaIncFlags)
      linkerFlags += $(cudaLibFlags)
    endif
  endif
endif


#---[ HIP ]-----------------------------
hipEnabled = 0
hipIncFlags = $(call includeFlagsFor,hip/hip_runtime_api.h)
ifneq (,$(hipIncFlags))
  HIP_PATH = ${hipIncFlags:-I%=%}/..
  hipEnabled = 1
endif
ifdef OCCA_HIP_ENABLED
  hipEnabled = $(OCCA_HIP_ENABLED)
endif

ifeq ($(hipEnabled),1)
  ifneq (,$(HIP_PATH))
    hipPlatform = $(shell $(HIP_PATH)/bin/hipconfig --platform)
    hipConfig = $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)
    ifeq ($(hipPlatform),nvcc)
      linkerFlags += -lcuda -lcudart
      hipLibFlags  = $(call libraryFlagsFor,cuda)
      hipLibFlags += $(call libraryFlagsFor,cudart)
    else ifeq ($(hipPlatform),hcc)
      linkerFlags += -lhip_hcc
      hipLibFlags = $(call libraryFlagsFor,hip_hcc)
    endif
    paths += $(hipConfig)
    paths += $(hipIncFlags)
    linkerFlags += $(hipLibFlags)
  endif
endif


#---[ OpenCL ]--------------------------
ifdef OCCA_OPENCL_ENABLED
  openclEnabled = $(OCCA_OPENCL_ENABLED)

  ifeq ($(openclEnabled),1)
    ifeq ($(usingLinux),1)
      linkerFlags += -lOpenCL
    else ifeq ($(usingMacOS),1)
      linkerFlags += -framework OpenCL
    endif
  endif
else
  ifeq ($(usingLinux),1)
    openclLibFlags = $(call libraryFlagsFor,OpenCL)
    ifneq (,$(openclLibFlags))

      openclIncFlags = $(call includeFlagsFor,CL/cl.h)
      ifneq (,$(openclIncFlags))
        openclEnabled = 1
        paths += $(openclIncFlags)
        linkerFlags += $(openclLibFlags)
      endif
    endif
  else ifeq ($(usingMacOS),1)
    # OpenCL includes are embedded in the framework
    openclLibFlags = $(call libraryFlagsFor,OpenCL)

    ifneq (,$(openclLibFlags))
      openclEnabled = 1
      linkerFlags += $(openclLibFlags)
    endif
  endif
endif


#---[ Metal ]---------------------------
# Metal is only supported with
# - MacOS
# - clang++ compiler
# - XCode version >= 10.2.1
ifeq ($(usingMacOS),1)
  ifdef OCCA_METAL_ENABLED
    metalEnabled := $(OCCA_METAL_ENABLED)
  else
    xcodeVersion := $(shell xcodebuild -version | head -n 1 | sed 's/Xcode //g')
    minXcodeVersion := $(shell echo -e "$(xcodeVersion)\n10.2.1" | sort -V | head -n 1)
    ifeq ($(minXcodeVersion),$(xcodeVersion))
      # Check for Metal Framework
      metalLibFlags = $(call libraryFlagsFor,Metal)

      ifneq (,$(metalLibFlags))
        metalEnabled = 1
        linkerFlags += $(metalLibFlags) -framework Foundation -lobjc
      endif
    endif
  endif
endif

ifeq ($(cudaEnabled),1)
  compilerFlags += -Wno-c++11-long-long
endif
compilerFlags += $(cpp11Flags)

ifeq ($(checkEnabled),1)
  OCCA_CHECK_ENABLED := 1
else
  OCCA_CHECK_ENABLED := 0
endif

OCCA_MPI_ENABLED    := $(mpiEnabled)
OCCA_OPENMP_ENABLED := $(openmpEnabled)
OCCA_CUDA_ENABLED   := $(cudaEnabled)
OCCA_HIP_ENABLED    := $(hipEnabled)
OCCA_OPENCL_ENABLED := $(openclEnabled)
OCCA_METAL_ENABLED  := $(metalEnabled)
#=================================================
