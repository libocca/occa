rmSlash = $(patsubst %/,%,$(1))

PROJ_DIR := $(call rmSlash,$(PROJ_DIR))

ifdef OCCA_DIR
  OCCA_DIR := $(call rmSlash,${OCCA_DIR})
endif

ifeq ($(strip $(OCCA_DIR)),)
  OCCA_DIR := $(call rmSlash,$(abspath $(dir $(abspath $(lastword $(MAKEFILE_LIST))))/../..))
endif

binPath  = $(PROJ_DIR)/bin
libPath  = $(PROJ_DIR)/lib
modPath  = $(PROJ_DIR)/lib/fortran/modules
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

FC ?= gfortran

ifndef FFLAGS
  ifeq ($(debugEnabled),1)
    fCompilerFlags = $(debugFlags)
  else
    fCompilerFlags = $(releaseFlags)
  endif
else
  fCompilerFlags = $(FFLAGS)
endif

ifeq ($(OCCA_COVERAGE),1)
  coverageFlags = --coverage
  compilerFlags += $(coverageFlags)
  cCompilerFlags += $(coverageFlags)
  fCompilerFlags += $(coverageFlags)
endif

compiler  = $(CXX)
cCompiler = $(CC)
fCompiler = $(FC)

linkerFlags = $(LDFLAGS)
fLinkerFlags = $(LDFLAGS)
#=================================================


#---[ Paths/Flags ]-------------------------------
# Include the public API
paths  = -I$(OCCA_DIR)/include
# Include the internal API
paths += -I$(OCCA_DIR)/src

# Link with the library for our examples
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

libraryFlagsFor = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; libraryFlags "$1")
includeFlagsFor = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; headerFlags  "$1")

compilerVendor       = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; compilerVendor       "$(compiler)")
compilerReleaseFlags = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; compilerReleaseFlags "$(compiler)")
compilerDebugFlags   = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; compilerDebugFlags   "$(compiler)")
compilerCpp11Flags   = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; compilerCpp11Flags   "$(compiler)")
compilerPicFlag      = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; compilerPicFlag      "$(compiler)")
compilerSharedFlag   = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; compilerSharedFlag   "$(compiler)")
compilerPthreadFlag  = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; compilerPthreadFlag  "$(compiler)")

fCompilerModuleDirFlag = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; fCompilerModuleDirFlag "$(fCompiler)")
fCompilerCppFlag       = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; fCompilerCppFlag       "$(fCompiler)")

compilerSupportsOpenMP = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; compilerSupportsOpenMP "$(compiler)")
compilerOpenMPFlag     = $(shell . $(OCCA_DIR)/scripts/build/shellTools.sh; compilerOpenMPFlag     "$(compiler)")
#=================================================


#---[ Compiler Info ]-----------------------------
vendor       = $(call compilerVendor)
debugFlags   = $(call compilerDebugFlags)
releaseFlags = $(call compilerReleaseFlags)
cpp11Flags   = $(call compilerCpp11Flags)
picFlag      = $(call compilerPicFlag)
sharedFlag   = $(call compilerSharedFlag)
pthreadFlag  = $(call compilerPthreadFlag)

fModuleDirFlag = $(call fCompilerModuleDirFlag)
fCppFlag       = $(call fCompilerCppFlag)

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
fortranEnabled = 0
mpiEnabled     = 0
openmpEnabled  = 0
cudaEnabled    = 0
hipEnabled     = 0
openclEnabled  = 0
metalEnabled   = 0
dpcppEnabled	 = 0


#---[ Fortran ]-------------------------
ifdef OCCA_FORTRAN_ENABLED
  fortranEnabled = $(OCCA_FORTRAN_ENABLED)
endif
include $(OCCA_DIR)/scripts/build/Make.fortran

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
ifdef OCCA_HIP_ENABLED
  hipEnabled = $(OCCA_HIP_ENABLED)

  ifeq ($(hipEnabled),1)
    #set HIP_PATH if not supplied by user
    ifeq (,$(HIP_PATH))
      hipIncFlags = $(call includeFlagsFor,hip/hip_runtime_api.h)
      ifneq (,$(hipIncFlags))
        HIP_PATH = ${hipIncFlags:-I%=%}/..
      endif
    endif

    #set HIP_PLATFORM if not supplied by user
    ifeq (,$(HIP_PLATFORM))
      hipPlatform = $(shell $(HIP_PATH)/bin/hipconfig --platform)
    else
      hipPlatform = $(HIP_PLATFORM)
    endif

    ifeq ($(hipPlatform),nvcc)
      linkerFlags += -lcuda
      linkerFlags += -lcudart
    else ifeq ($(hipPlatform),$(filter $(hipPlatform),hcc amd))
      #set HIP_COMPILER if not supplied by user
      ifeq (,$(HIP_COMPILER))
        hipCompiler = $(shell $(HIP_PATH)/bin/hipconfig --compiler)
      else
        hipCompiler = $(HIP_COMPILER)
      endif

      #Select HCC or HIP Runtime
      ifeq ($(hipCompiler),hcc)
        linkerFlags += -lhip_hcc
      else ifeq ($(hipCompiler),clang)
        linkerFlags += -lamdhip64
      endif
    endif
  endif
else
  #find HIP_PATH if not supplied by user
  ifeq (,$(HIP_PATH))
    hipIncFlags = $(call includeFlagsFor,hip/hip_runtime_api.h)
    ifneq (,$(hipIncFlags))
      HIP_PATH = ${hipIncFlags:-I%=%}/..
    endif
  endif

  ifneq (,$(HIP_PATH))
    #set HIP_PLATFORM if not supplied by user
    ifeq (,$(HIP_PLATFORM))
      hipPlatform = $(shell $(HIP_PATH)/bin/hipconfig --platform)
    else
      hipPlatform = $(HIP_PLATFORM)
    endif

    ifeq ($(hipPlatform),nvcc)
      hipLibFlags  = $(call libraryFlagsFor,cuda)
      hipLibFlags += $(call libraryFlagsFor,cudart)
    else ifeq ($(hipPlatform),$(filter $(hipPlatform),hcc amd))
      #set HIP_COMPILER if not supplied by user
      ifeq (,$(HIP_COMPILER))
        hipCompiler = $(shell $(HIP_PATH)/bin/hipconfig --compiler)
      else
        hipCompiler = $(HIP_COMPILER)
      endif

      #Select HCC or HIP Runtime
      ifeq ($(hipCompiler),hcc)
        hipLibFlags = $(call libraryFlagsFor,hip_hcc)
      else ifeq ($(hipCompiler),clang)
        hipLibFlags = $(call libraryFlagsFor,amdhip64)
      endif
    endif

    ifneq (,$(hipLibFlags))
      hipEnabled = 1
      paths += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)
      paths += $(call includeFlagsFor,hip/hip_runtime_api.h)
      linkerFlags += $(hipLibFlags)
    endif
  endif
endif

#---[ SYCL/DPC++ ]--------------------------
ifdef OCCA_DPCPP_ENABLED
  dpcppEnabled = $(OCCA_DPCPP_ENABLED)

  ifeq ($(dpcppEnabled),1)
    ifeq ($(usingLinux),1)
      linkerFlags += -lsycl
    else ifeq ($(usingMacOS),1)
      linkerFlags += -framework sycl
    endif
  endif
else
  ifeq ($(usingLinux),1)
    dpcppLibFlags = $(call libraryFlagsFor,sycl)
    ifneq (,$(dpcppLibFlags))

      dpcppIncFlags = $(call includeFlagsFor,CL/sycl.hpp)
      ifneq (,$(dpcppIncFlags))
        dpcppEnabled = 1
        paths += $(dpcppIncFlags)
        linkerFlags += $(dpcppLibFlags)
      endif
    endif
  else ifeq ($(usingMacOS),1)
    dpcppLibFlags = $(call libraryFlagsFor,sycl)

    ifneq (,$(dpcppLibFlags))
      dpcppEnabled = 1
      linkerFlags += $(dpcppLibFlags)
    endif
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

OCCA_FORTRAN_ENABLED := $(fortranEnabled)
OCCA_OPENMP_ENABLED  := $(openmpEnabled)
OCCA_CUDA_ENABLED    := $(cudaEnabled)
OCCA_HIP_ENABLED     := $(hipEnabled)
OCCA_OPENCL_ENABLED  := $(openclEnabled)
OCCA_METAL_ENABLED   := $(metalEnabled)
OCCA_DPCPP_ENABLED   := $(dpcppEnabled)
#=================================================
