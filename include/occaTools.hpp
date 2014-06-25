#ifndef OCCA_TOOLS_HEADER
#define OCCA_TOOLS_HEADER

#include <iostream>
#include <stdlib.h>

#if   OCCA_OS == LINUX_OS
#  include <sys/time.h>
#elif OCCA_OS == OSX_OS
#  include <CoreServices/CoreServices.h>
#  include <mach/mach_time.h>
#else
#  include <windows.h>
#endif

namespace occa {
  class kernelInfo;

  double currentTime();

  std::string fnv(const std::string &filename);

  std::string readFile(const std::string &filename);

  std::string binaryIsCached(const std::string &filename,
                             const std::string &salt);

  std::string createIntermediateSource(const std::string &filename,
                                       const std::string &cachedBinary,
                                       const kernelInfo &info);
};

#endif
