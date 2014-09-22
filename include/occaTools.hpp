#ifndef OCCA_TOOLS_HEADER
#define OCCA_TOOLS_HEADER

#include <iostream>
#include <stdlib.h>

#include <sys/stat.h>
#include <string.h>
#include <errno.h>

#if   OCCA_OS == LINUX_OS
#  include <sys/time.h>
#  include <unistd.h>
#elif OCCA_OS == OSX_OS
#  include <CoreServices/CoreServices.h>
#  include <mach/mach_time.h>
#else
#  include <windows.h>
#endif

namespace occa {
  class kernelInfo;

  double currentTime();

  void getFilePrefixAndName(const std::string &fullFilename,
                            std::string &prefix,
                            std::string &filename);

  std::string getFileLock(const std::string &filename);

  bool haveFile(const std::string &filename);

  void waitForFile(const std::string &filename);

  void releaseFile(const std::string &filename);

  std::string fnv(const std::string &filename);

  std::string readFile(const std::string &filename);

  std::string getCachedName(const std::string &filename,
                            const std::string &salt);

  std::string createIntermediateSource(const std::string &filename,
                                       const std::string &cachedBinary,
                                       const kernelInfo &info);
};

#endif
