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

  inline double currentTime(){
#if OCCA_OS == LINUX_OS

    timespec ct;
    clock_gettime(CLOCK_MONOTONIC, &ct);

    return (double) (ct.tv_sec + (1.0e-9 * ct.tv_nsec));

#elif OCCA_OS == OSX_OS

    uint64_t ct;
    ct = mach_absolute_time();

    const Nanoseconds ct2 = AbsoluteToNanoseconds(*(AbsoluteTime *) &ct);

    return ((double) 1.0e-9) * ((double) ( *((uint64_t*) &ct2) ));

#elif OCCA_OS == WINDOWS_OS
#  warning "currentTime is not supported in Windows"
#endif
  }

  // inline lockFile(const std::string &filename){
  // }

  // inline unlockFile(const std::string &filename){
  //   flock (lockFd, LOCK_UN);
  //   close (lockFd);
  // }

  std::string fnv(const std::string &filename);

  std::string readFile(const std::string &filename);

  std::string getCachedName(const std::string &filename,
                            const std::string &salt);

  void getFilePrefixAndName(const std::string &fullFilename,
                            std::string &prefix,
                            std::string &filename);

  std::string createIntermediateSource(const std::string &filename,
                                       const std::string &cachedBinary,
                                       const kernelInfo &info);
};

#endif
