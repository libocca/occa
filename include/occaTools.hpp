#ifndef OCCA_TOOLS_HEADER
#define OCCA_TOOLS_HEADER

#include <iostream>
#include <stdlib.h>
#include <stdint.h>

#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include "occaDefines.hpp"
#include "occaParser.hpp"

#if   OCCA_OS == LINUX_OS
#  include <sys/time.h>
#  include <unistd.h>
#elif OCCA_OS == OSX_OS
#  include <CoreServices/CoreServices.h>
#  include <mach/mach_time.h>
#else
#  undef UNICODE
#  include <windows.h>
#  include <string>
#endif

namespace occa {
  class kernelInfo;

  class mutex_t {
  public:
#if (OCL_OS == OCL_LINUX_OS) || (OCL_OS == OCL_OSX_OS)
    pthread_mutex_t mutexHandle;
#else
    HANDLE mutexHandle;
#endif

    mutex_t();
    void free();

    void lock();
    void unlock();
  };

  double currentTime();

  std::string getFileExtension(const std::string &filename);

  void getFilePrefixAndName(const std::string &fullFilename,
                            std::string &prefix,
                            std::string &filename);

  std::string getMidCachedBinaryName(const std::string &cachedBinary,
                                     const std::string &namePrefix);

  std::string getFileLock(const std::string &filename);

  bool haveFile(const std::string &filename);
  void waitForFile(const std::string &filename);
  void releaseFile(const std::string &filename);

  parsedKernelInfo parseFileForFunction(const std::string &filename,
                                        const std::string &cachedBinary,
                                        const std::string &functionName,
                                        const kernelInfo &info);

  std::string fnv(const std::string &filename);

  bool fileExists(const std::string &filename);
  std::string readFile(const std::string &filename);

  std::string getOCCADir();
  std::string getCachePath();

  std::string getCachedName(const std::string &filename,
                            const std::string &salt);

  std::string createIntermediateSource(const std::string &filename,
                                       const std::string &cachedBinary,
                                       const kernelInfo &info);
};

#endif
