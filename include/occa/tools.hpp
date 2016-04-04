#ifndef OCCA_TOOLS_HEADER
#define OCCA_TOOLS_HEADER

#include <iostream>

#include <stdlib.h>
#include <stdint.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <assert.h>

#include "occa/defines.hpp"
#include "occa/parser/types.hpp"

#if   (OCCA_OS & LINUX_OS)
#  include <sys/time.h>
#  include <unistd.h>
#  include <sys/types.h>
#  include <sys/dir.h>
#elif (OCCA_OS & OSX_OS)
#  ifdef __clang__
#    include <CoreServices/CoreServices.h>
#    include <mach/mach_time.h>
#  else
#    include <mach/clock.h>
#    include <mach/mach.h>
#  endif
#  include <sys/types.h>
#  include <sys/dir.h>
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

  extern strToBoolMap_t fileLocks;

  //---[ Helper Info ]----------------
  namespace env {
    extern bool isInitialized;

    extern std::string HOME, PWD;
    extern std::string PATH, LD_LIBRARY_PATH;

    extern std::string OCCA_DIR, OCCA_CACHE_DIR;
    extern size_t OCCA_MEM_BYTE_ALIGN;
    extern stringVector_t OCCA_INCLUDE_PATH;

    void initialize();
    void initCachePath();
    void initIncludePath();

    void signalExit(int sig);

    std::string var(const std::string &var);

    void endDirWithSlash(std::string &dir);

    class envInitializer_t {
    public: envInitializer_t();
    };
    extern envInitializer_t envInitializer;
  }

  namespace sys {
    int call(const std::string &cmdline);
    int call(const std::string &cmdline, std::string &output);

    std::string expandEnvVariables(const std::string &str);

    void rmdir(const std::string &dir);
    int mkdir(const std::string &dir);
    void mkpath(const std::string &dir);

    bool dirExists(const std::string &dir_);
    bool fileExists(const std::string &filename_,
                    const int flags = 0);

    std::string getFilename(const std::string &filename);

    void absolutePathVec(const std::string &path_,
                         stringVector_t &pathVec);

    stringVector_t absolutePathVec(const std::string &path);
  }

  // Kernel Caching
  namespace kc {
    extern std::string sourceFile;
    extern std::string binaryFile;
  }
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
  std::string getOnlyFilename(const std::string &filename);
  std::string getPlainFilename(const std::string &filename);
  std::string getFileDirectory(const std::string &filename);
  std::string getFileExtension(const std::string &filename);

  std::string compressFilename(const std::string &filename);

  std::string readFile(const std::string &filename,
                       const bool readingBinary = false);

  void writeToFile(const std::string &filename,
                   const std::string &content);

  std::string getFileLock(const std::string &filename, const int n);
  void clearLocks();

  bool haveHash(const std::string &hash, const int depth = 0);
  void waitForHash(const std::string &hash, const int depth = 0);
  void releaseHash(const std::string &hash, const int depth = 0);
  void releaseHashLock(const std::string &lockDir);

  bool fileNeedsParser(const std::string &filename);

  parsedKernelInfo parseFileForFunction(const std::string &deviceMode,
                                        const std::string &filename,
                                        const std::string &cachedBinary,
                                        const std::string &functionName,
                                        const kernelInfo &info);

  std::string removeSlashes(const std::string &str);

  void setupOccaHeaders(const kernelInfo &info);

  void cacheFile(const std::string &filename,
                 std::string source,
                 const std::string &hash);

  void cacheFile(const std::string &filename,
                 const char *source,
                 const std::string &hash,
                 const bool deleteSource = true);

  void createSourceFileFrom(const std::string &filename,
                            const std::string &hashDir,
                            const kernelInfo &info);
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

  std::string getLibraryName(const std::string &filename);

  std::string hashFrom(const std::string &filename);

  std::string hashDirFor(const std::string &filename,
                         const std::string &hash);
  //==============================================


  //---[ String Functions ]-----------------------
  uintptr_t atoi(const char *c);
  uintptr_t atoiBase2(const char *c);
  uintptr_t atoi(const std::string &str);

  double atof(const char *c);
  double atof(const std::string &str);

  double atod(const char *c);
  double atod(const std::string &str);

  std::string stringifyBytes(uintptr_t bytes);
  //==============================================


  //---[ Misc Functions ]-------------------------
  int maxBase2Bit(const int value);
  int maxBase2(const int value);

  uintptr_t ptrDiff(void *start, void *end);
  void* ptrOff(void *ptr, uintptr_t offset);
  //==============================================

  template <class TM>
  void ignoreResult(const TM &t){
    (void) t;
  }
}

#endif
