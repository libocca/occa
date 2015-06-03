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
#include "occaParserTypes.hpp"

#if   (OCCA_OS & LINUX_OS)
#  include <sys/time.h>
#  include <unistd.h>
#elif (OCCA_OS & OSX_OS)
#  ifdef __clang__
#    include <CoreServices/CoreServices.h>
#    include <mach/mach_time.h>
#  else
#    include <mach/clock.h>
#    include <mach/mach.h>
#  endif
#else
#  ifndef NOMINMAX
#    define NOMINMAX     // NBN: clear min/max macros
#  endif
#  include <windows.h>
#  include <string>
#  include <direct.h>    // NBN: rmdir _rmdir
#endif

namespace occa {
  class kernelInfo;

  //---[ Helper Info ]----------------
  namespace env {
    extern bool isInitialized;

    extern std::string HOME;
    extern std::string PATH, LD_LIBRARY_PATH;

    void initialize();
    std::string echo(const std::string &var);
  };
  //==================================

  class mutex_t {
  public:
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    pthread_mutex_t mutexHandle;
#else
    HANDLE mutexHandle;
#endif

    mutex_t();
    void free();

    void lock();
    void unlock();
  };

  class fnvOutput_t {
  public:
    int h[8];

    fnvOutput_t();

    bool operator == (const fnvOutput_t &fo);
    bool operator != (const fnvOutput_t &fo);

    void mergeWith(const fnvOutput_t &fo);

    operator std::string ();
  };

  double currentTime();

  std::string getFilePrefix(const std::string &filename);
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

  fnvOutput_t fnv(const void *ptr, uintptr_t bytes);

  template <class TM>
  fnvOutput_t fnv(const TM &t){
    return fnv(&t, sizeof(TM));
  }

  template <>
  fnvOutput_t fnv(const std::string &saltedString);

  bool fileExists(const std::string &filename);

  std::string readFile(const std::string &filename,
                       const bool readingBinary = false);

  void writeToFile(const std::string &filename,
                   const std::string &content);

  std::string getOCCADir();
  std::string getCachePath();

  std::string removeSlashes(const std::string &str);

  bool fileNeedsParser(const std::string &filename);

  std::string getCacheHash(const std::string &content,
                           const std::string &salt);

  std::string getCachedName(const std::string &filename,
                            const std::string &salt);

  std::string getContentCachedName(const std::string &content,
                                   const std::string &salt);

  std::string createIntermediateSource(const std::string &filename,
                                       const std::string &cachedBinary,
                                       const kernelInfo &info,
                                       const bool useParser = true);

  template <class TM>
  TM strTo(const std::string &str){
    std::stringstream ss;
    TM ret;

    ss << str;
    ss >> ret;

    return ret;
  }

  template <class TM>
  void ignoreResult(const TM &t){}
};

#endif
