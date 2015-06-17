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

    extern std::string HOME, PWD;
    extern std::string PATH, LD_LIBRARY_PATH;

    extern std::string OCCA_DIR, OCCA_CACHE_DIR;

    void initialize();

    std::string getAndInitCachePath();

    inline void endDirWithSlash(std::string &dir){
      if((0 < dir.size()) &&
         (dir[dir.size() - 1] != '/')){

        dir += '/';
      }
    }
  };

  namespace sys {
    std::string echo(const std::string &var);

    int rmdir(const std::string &dir);
    int mkdir(const std::string &dir);
    int mkpath(const std::string &dir);

    bool dirExists(const std::string &dir);
    bool fileExists(const std::string &filename,
                    const int flags = 0);

    std::string getFilename(const std::string &filename);

    void absolutePathVec(const std::string &dir_,
                         stringVector_t &pathVec);

    inline stringVector_t absolutePathVec(const std::string &dir){
      stringVector_t pathVec;

      absolutePathVec(dir, pathVec);

      return pathVec;
    }
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

  //---[ File Functions ]-------------------------
  std::string getFileDirectory(const std::string &filename);
  std::string getFileExtension(const std::string &filename);

  std::string readFile(const std::string &filename,
                       const bool readingBinary = false);

  void writeToFile(const std::string &filename,
                   const std::string &content);

  std::string getFileLock(const std::string &filename, const int n);

  bool haveHash(const std::string &hash, const int depth);
  void waitForHash(const std::string &hash, const int depth);
  void releaseHash(const std::string &hash, const int depth);

  bool fileNeedsParser(const std::string &filename);

  parsedKernelInfo parseFileForFunction(const std::string &filename,
                                        const std::string &cachedBinary,
                                        const std::string &functionName,
                                        const kernelInfo &info);

  std::string removeSlashes(const std::string &str);
  //==============================================


  //---[ Hash Functions ]-------------------------
  fnvOutput_t fnv(const void *ptr, uintptr_t bytes);

  template <class TM>
  fnvOutput_t fnv(const TM &t){
    return fnv(&t, sizeof(TM));
  }

  template <>
  fnvOutput_t fnv(const std::string &saltedString);

  std::string getContentHash(const std::string &content,
                             const std::string &salt);

  std::string getFileContentHash(const std::string &content,
                                 const std::string &salt);

  std::string getOccaLibraryName(const std::string &filename);

  std::string hashDirFor(const std::string &filename,
                         const std::string &hash);
  //==============================================

  template <class TM>
  std::string strFrom(const TM &t){
    std::stringstream ss;

    ss << t;

    return ss.str();
  }

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
