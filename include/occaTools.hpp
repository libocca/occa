#ifndef OCCA_TOOLS_HEADER
#define OCCA_TOOLS_HEADER

#include <iostream>
#include <stdlib.h>

#include <sys/stat.h>
#include <string.h>
#include <errno.h>

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

  inline void getFilePrefixAndName(const std::string &fullFilename,
                                   std::string &prefix,
                                   std::string &filename){
    int lastSlash = 0;
    const int chars = fullFilename.size();

    for(int i = 0; i < chars; ++i)
      if(fullFilename[i] == '/')
        lastSlash = i;

    ++lastSlash;

    prefix   = fullFilename.substr(0, lastSlash);
    filename = fullFilename.substr(lastSlash, chars - lastSlash);
  }

  inline std::string getFileLock(const std::string &filename){
    std::string prefix, name;
    getFilePrefixAndName(filename, prefix, name);

    return (prefix + "._occa_dir_" + name);
  }

  inline bool haveFile(const std::string &filename){
    std::string lockDir = getFileLock(filename);

    int mkdirStatus = mkdir(lockDir.c_str(), 0755);

    // Someone else is making it
    if(mkdirStatus && (errno == EEXIST))
      return false;

    return true;
  }

  inline void waitForFile(const std::string &filename){
    struct stat buffer;

    std::string lockDir   = getFileLock(filename);
    const char *c_lockDir = lockDir.c_str();

    while(stat(c_lockDir, &buffer) == 0)
      /* Do Nothing */;
  }

  inline void releaseFile(const std::string &filename){
    std::string lockDir = getFileLock(filename);

    rmdir(lockDir.c_str());
  }

  std::string fnv(const std::string &filename);

  std::string readFile(const std::string &filename);

  std::string getCachedName(const std::string &filename,
                            const std::string &salt);

  std::string createIntermediateSource(const std::string &filename,
                                       const std::string &cachedBinary,
                                       const kernelInfo &info);
};

#endif
