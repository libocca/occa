#---[ Paths ]-------------------------------------
bPath = bin
lPath = lib
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
WINUX_OS   = 5 # (LINUX_OS | WINDOWS_OS)

usingLinux   = 0
usingOSX     = 0
usingWinux   = 0
usingWindows = 0

flags = -D LINUX_OS=$(LINUX_OS) -D OSX_OS=$(OSX_OS) -D WINDOWS_OS=$(WINDOWS_OS) -D WINUX_OS=$(WINUX_OS)

UNAME = $(shell uname)

ifeq ($(UNAME),Linux)
  usingLinux   = 1
else ifeq ($(UNAME),Darwin)
  usingOSX     = 1
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

links = $(LDFLAGS)
#=================================================


#---[ Paths/Flags/Links ]-------------------------
paths = -I${OCCA_DIR}/lib -I${OCCA_DIR}/include -L${OCCA_DIR}/lib
paths += $(foreach path, $(subst :, ,$(OCCA_INCLUDE_PATH)), -I$(path))
paths += $(foreach path, $(subst :, ,$(OCCA_LIBRARY_PATH)), -L$(path))

ifneq (,$(wildcard ./$(iPath)/*))
  paths += -I./$(iPath)
endif

links += -locca
#=================================================


#---[ Shell Tools ]-------------------------------
ifeq (,$(findstring bash,$(SHELL)))
  SHELL := $(shell which bash)
  ifeq (,$(SHELL))
    $(error Could not find [bash], set SHELL manually with [export SHELL=/path/to/bash] or compile with [make SHELL=/path/to/bash])
  endif
endif

libraryFlagsFor = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; libraryFlags $1)
includeFlagsFor = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; headerFlags  $1)

compilerReleaseFlags = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerReleaseFlags $(compiler))
compilerDebugFlags   = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerDebugFlags   $(compiler))
compilerPicFlag      = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerPicFlag      $(compiler))
compilerSharedFlag   = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerSharedFlag   $(compiler))

fCompilerModuleDirFlag = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; fCompilerModuleDirFlag $(fCompiler))

compilerSupportsOpenMP = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerSupportsOpenMP $(compiler))
compilerOpenMPFlags    = $(shell . ${OCCA_DIR}/scripts/shellTools.sh; compilerOpenMPFlags    $(compiler))
#=================================================


#---[ Compiler Info ]-----------------------------
debugFlags   = $(call compilerDebugFlags)
releaseFlags = $(call compilerReleaseFlags)
picFlag      = $(call compilerPicFlag)
sharedFlag   = $(call compilerSharedFlag)
fModDirFlag  = $(call fCompilerModuleDirFlag)
#=================================================


#---[ Flags and Libraries ]-----------------------
ifeq ($(usingLinux),1)

  ifeq ($(usingWinux),0)
    flags += -D OCCA_OS=LINUX_OS
  else
    flags += -D OCCA_OS=WINUX_OS
  endif

  links += -lm -lrt -ldl

else ifeq ($(usingOSX),1)

  flags += -D OCCA_OS=OSX_OS -Wno-deprecated-declarations
  links += -framework accelerate -framework CoreServices

else ifeq ($(usingWindows),1)

  ifeq ($(usingWinux),0)
    flags += -D OCCA_OS=WINDOWS_OS -D OCCA_USING_VS=1
  else
    flags += -D OCCA_OS=WINDOWS_OS -D OCCA_USING_VS=0
  endif

  links +=

endif
#=================================================


#---[ Variable Dependencies ]---------------------
openmpEnabled = 0
openclEnabled = 0
cudaEnabled   = 0
hsaEnabled    = 0
coiEnabled    = 0

ifdef OCCA_OPENMP_ENABLED
  openmpEnabled = $(OCCA_OPENMP_ENABLED)
else
  openmpEnabled = $(call compilerSupportsOpenMP)

  ifeq ($(openmpEnabled), 1)
    flags += $(call compilerOpenMPFlags)
  endif
endif

ifdef OCCA_OPENCL_ENABLED
  openclEnabled = $(OCCA_OPENCL_ENABLED)

  ifeq ($(openclEnabled), 1)
    ifeq ($(usingLinux),1)
      links += -lOpenCL
    else ifeq ($(usingOSX),1)
      links += -framework OpenCL
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
        links += $(openclLibFlags)
      endif
    endif
  else ifeq ($(usingOSX),1)
    # OpenCL includes are embedded in the framework
    openclLibFlags = $(call libraryFlagsFor,OpenCL)

    ifneq (,$(openclLibFlags))
      openclEnabled = 1
      links += $(openclLibFlags)
    endif
  endif
endif

ifdef OCCA_CUDA_ENABLED
  cudaEnabled = $(OCCA_CUDA_ENABLED)

  ifeq ($(cudaEnabled), 1)
    ifeq ($(usingLinux),1)
      links += -lcuda
    else ifeq ($(usingOSX),1)
      links += -framework CUDA
    endif
  endif
else
  cudaIncFlags = $(call includeFlagsFor,cuda.h)

  ifneq (,$(cudaIncFlags))

    ifeq ($(usingLinux),1)
      cudaLibFlags = $(call libraryFlagsFor,cuda)
    else ifeq ($(usingOSX),1)
      cudaLibFlags = $(call libraryFlagsFor,CUDA)
    endif

    ifneq (,$(cudaLibFlags))
      cudaEnabled = 1
      paths += $(cudaIncFlags)
      links += $(cudaLibFlags)
    endif
  endif
endif

# ifdef OCCA_HSA_ENABLED
#   hsaEnabled = $(OCCA_HSA_ENABLED)

#   ifeq ($(hsaEnabled), 1)
#     ifeq ($(usingLinux),1)
#       links += -lhsa                              # ?
#     endif
#   endif
# else
#   hsaIncFlags = $(call includeFlagsFor,hsa.h)     # ?

#   ifneq (,$(hsaIncFlags))

#     ifeq ($(usingLinux),1)
#       hsaLibFlags = $(call libraryFlagsFor,hsa)   # ?
#     endif

#     ifneq (,$(hsaLibFlags))
#       hsaEnabled = 1
#       paths += $(hsaIncFlags)
#       links += $(hsaLibFlags)
#     endif
#   endif
# endif

ifdef OCCA_COI_ENABLED
  coiEnabled = $(OCCA_COI_ENABLED)

  ifeq ($(coiEnabled), 1)
    ifeq ($(usingLinux),1)
      links += -lcoi_host
    endif
  endif
else
  ifeq ($(usingLinux),1)
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

cfFlags =

ifeq ($(debugEnabled), 1)
  cfFlags += $(debugFlags)
  cfFlags += -DOCCA_DEBUG_ENABLED=1
else
  cfFlags += $(releaseFlags)
  cfFlags += -DOCCA_DEBUG_ENABLED=0 -DNDEBUG=1 -DOCCA_SHOW_WARNINGS=0
endif

ifeq ($(checkEnabled), 1)
  cfFlags  += -DOCCA_CHECK_ENABLED=1
else
  cfFlags  += -DOCCA_CHECK_ENABLED=0
endif

cfFlags += -DOCCA_OPENMP_ENABLED=$(openmpEnabled)
cfFlags += -DOCCA_OPENCL_ENABLED=$(openclEnabled)
cfFlags += -DOCCA_CUDA_ENABLED=$(cudaEnabled)
cfFlags += -DOCCA_HSA_ENABLED=$(hsaEnabled)
cfFlags += -DOCCA_COI_ENABLED=$(coiEnabled)

ifdef COMPILING_WITH_JULIA
  cfFlags += -DOCCA_COMPILED_FOR_JULIA=1
endif

flags  += $(cfFlags)
fFlags += $(cfFlags)
#=================================================
