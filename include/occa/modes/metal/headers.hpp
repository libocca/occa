#include <occa/defines.hpp>

#if   (OCCA_OS & OCCA_LINUX_OS)
#  include <CL/cl.h>
#  include <CL/cl_gl.h>
#elif (OCCA_OS & OCCA_MACOS_OS)
#  include <OpenCL/OpenCl.h>
#else
#  include "CL/opencl.h"
#endif
