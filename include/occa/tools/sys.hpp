#ifndef OCCA_TOOLS_SYS_HEADER
#define OCCA_TOOLS_SYS_HEADER

#include <iostream>
#include <sstream>

#include <occa/defines.hpp>
#include <occa/types.hpp>
#include <occa/io/lock.hpp>
#include <occa/io/output.hpp>
#include <occa/tools/hash.hpp>

namespace occa {
  typedef void (*functionPtr_t)(...);

  namespace sys {
    namespace vendor {
      static const int notFound = 0;

      static const int b_GNU          = 0;
      static const int b_LLVM         = 1;
      static const int b_Intel        = 2;
      static const int b_Pathscale    = 3;
      static const int b_IBM          = 4;
      static const int b_PGI          = 5;
      static const int b_HP           = 6;
      static const int b_VisualStudio = 7;
      static const int b_Cray         = 8;
      static const int b_max          = 9;

      static const int GNU          = (1 << b_GNU);          // gcc    , g++
      static const int LLVM         = (1 << b_LLVM);         // clang  , clang++
      static const int Intel        = (1 << b_Intel);        // icc    , icpc
      static const int Pathscale    = (1 << b_Pathscale);    // pathCC
      static const int IBM          = (1 << b_IBM);          // xlc    , xlc++
      static const int PGI          = (1 << b_PGI);          // pgcc   , pgc++
      static const int HP           = (1 << b_HP);           // aCC
      static const int VisualStudio = (1 << b_VisualStudio); // cl.exe
      static const int Cray         = (1 << b_Cray);         // cc     , CC
    }

    //---[ System Info ]----------------
    double currentTime();
    std::string date();
    std::string humanDate();
    //==================================

    //---[ System Calls ]---------------
    int call(const std::string &cmdline);
    int call(const std::string &cmdline, std::string &output);

    std::string expandEnvVariables(const std::string &str);

    void rmdir(const std::string &dir,
               const bool recursive = false);
    void rmrf(const std::string &filename);
    bool isSafeToRmrf(const std::string &filename);

    int mkdir(const std::string &dir);
    void mkpath(const std::string &dir);

    bool pidExists(const int pid);

    int getPID();
    int getTID();
    void pinToCore(const int core);
    //==================================

    //---[ Processor Info ]-------------
    std::string getFieldFrom(const std::string &command,
                             const std::string &field);

    std::string getProcessorName();
    int getCoreCount();
    int getProcessorFrequency();
    std::string getProcessorCacheSize(int level);
    udim_t installedRAM();
    udim_t availableRAM();

    int compilerVendor(const std::string &compiler);

    std::string compilerSharedBinaryFlags(const std::string &compiler);
    std::string compilerSharedBinaryFlags(const int vendor_);

    void addSharedBinaryFlagsTo(const std::string &compiler, std::string &flags);
    void addSharedBinaryFlagsTo(const int vendor_, std::string &flags);
    //==================================

    //---[ Dynamic Methods ]------------
    void* malloc(udim_t bytes);
    void free(void *ptr);

    void* dlopen(const std::string &filename,
                 const io::lock_t &lock = io::lock_t());

    functionPtr_t dlsym(void *dlHandle,
                        const std::string &functionName,
                        const io::lock_t &lock = io::lock_t());

    void dlclose(void *dlHandle);

    void runFunction(functionPtr_t f, const int argc, void **args);

    std::string stacktrace(const int frameStart = 0,
                           const std::string indent = "");

    std::string prettyStackSymbol(void *frame, const char *symbol);
    //==================================
  }

  void _message(const std::string &header,
                const bool exitInFailure,
                const std::string &filename,
                const std::string &function,
                const int line,
                const std::string &message);

  void warn(const std::string &filename,
            const std::string &function,
            const int line,
            const std::string &message);

  void error(const std::string &filename,
             const std::string &function,
             const int line,
             const std::string &message);

  void printNote(io::output &out,
                 const std::string &message);

  inline void printNote(const std::string &message) {
    printNote(io::stderr, message);
  }

  void printWarning(io::output &out,
                    const std::string &message);

  inline void printWarning(const std::string &message) {
    printWarning(io::stderr, message);
  }

  void printError(io::output &out,
                  const std::string &message);

  inline void printError(const std::string &message) {
    printError(io::stderr, message);
  }

  class mutex {
  public:
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    pthread_mutex_t mutexHandle;
#else
    void *mutexHandle;
#endif

    mutex();
    void free();

    void lock();
    void unlock();
  };
}

#endif
