#ifndef OCCA_TOOLS_HEADER
#define OCCA_TOOLS_HEADER

#include <iostream>

#include <stdlib.h>
#include <stdint.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include "occa/defines.hpp"
#include "occa/parser/types.hpp"

#if   (OCCA_OS & LINUX_OS)
#  include <sys/time.h>
#  include <unistd.h>
#  include <signal.h>
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
#  include <signal.h>
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

    inline void endDirWithSlash(std::string &dir){
      if((0 < dir.size()) &&
         (dir[dir.size() - 1] != '/')){

        dir += '/';
      }
    }

    class envInitializer_t {
    public: envInitializer_t();
    };
    extern envInitializer_t envInitializer;
  }

  namespace sys {
    namespace dirType {
      static const int none    = 0;
      static const int dir     = (1 << 0);
      static const int file    = (1 << 1);
      static const int symlink = (1 << 2);
      static const int hidden  = (1 << 3);
    }

    class dirTree_t {
    public:
      int info;
      std::string name;

      int dirCount, nestedDirCount;
      dirTree_t *dirs;

      // Store file names in [root]
      std::string *nestedDirNames;

      dirTree_t();

      dirTree_t(const dirTree_t &dt);
      dirTree_t& operator = (const dirTree_t &dt);

      dirTree_t(const std::string &dir);

      void load(const std::string &dir_);

      bool load(const std::string &base,
                stringVector_t &path,
                const int pathPos);

      void free();

      inline int fileCount(){
        return nestedDirCount;
      }

      inline std::string& filename(int i){
        return nestedDirNames[i];
      }

      void setNestedDirNames(std::string *fdn,
                             int fdnPos);

      void printOnString(std::string &str,
                         const char delimiter);

      inline std::string toString(const char delimiter = '\n'){
        std::string ret;
        printOnString(ret, delimiter);
        return ret;
      }

      bool hasWildcard(const char *c);
      bool matches(const char *search,
                   const char *c);
    };

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

    inline stringVector_t absolutePathVec(const std::string &path){
      stringVector_t pathVec;
      absolutePathVec(path, pathVec);
      return pathVec;
    }
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

  inline uintptr_t atoi(const std::string &str){
    return occa::atoi((const char*) str.c_str());
  }

  inline double atof(const char *c){
    return ::atof(c);
  }

  inline double atof(const std::string &str){
    return ::atof(str.c_str());
  }

  inline double atod(const char *c){
    double ret;
    sscanf(c, "%lf", &ret);
    return ret;
  }

  inline double atod(const std::string &str){
    return occa::atod(str.c_str());
  }

  std::string stringifyBytes(uintptr_t bytes);
  //==============================================


  //---[ Misc Functions ]-------------------------
  inline int maxBase2Bit(const int value){
    if(value <= 0)
      return 0;

    const int maxBits = 8 * sizeof(value);

    for(int i = 0; i < maxBits; ++i){
      if(value <= (1 << i))
        return i;
    }

    return 0;
  }

  inline int maxBase2(const int value){
    return (1 << maxBase2Bit(value));
  }

  inline uintptr_t ptrDiff(void *start, void *end){
    if(start < end)
      return (uintptr_t) (((char*) end) - ((char*) start));

    return (uintptr_t) (((char*) start) - ((char*) end));
  }

  inline void* ptrOff(void *ptr, uintptr_t offset){
    return (void*) (((char*) ptr) + offset);
  }
  //==============================================

  template <class TM>
  void ignoreResult(const TM &t){
    (void) t;
  }
}

#endif
