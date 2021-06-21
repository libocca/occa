#ifndef OCCA_INTERNAL_UTILS_SYS_HEADER
#define OCCA_INTERNAL_UTILS_SYS_HEADER

#include <iostream>
#include <sstream>

#include <occa/defines.hpp>
#include <occa/types.hpp>
#include <occa/internal/io/output.hpp>
#include <occa/internal/utils/enums.hpp>
#include <occa/utils/hash.hpp>
#include <occa/types/json.hpp>

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
      static const int b_PPC          = 9;
      static const int b_max          = 10;

      static const int GNU          = (1 << b_GNU);          // gcc    , g++
      static const int LLVM         = (1 << b_LLVM);         // clang  , clang++
      static const int Intel        = (1 << b_Intel);        // icc    , icpc
      static const int Pathscale    = (1 << b_Pathscale);    // pathcc , pathCC
      static const int IBM          = (1 << b_IBM);          // xlc    , xlc++
      static const int PGI          = (1 << b_PGI);          // pgcc   , pgc++
      static const int HP           = (1 << b_HP);           // aCC
      static const int VisualStudio = (1 << b_VisualStudio); // cl.exe
      static const int Cray         = (1 << b_Cray);         // cc     , CC
      static const int PPC          = (1 << b_PPC);          // cc     , CC
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
    class CacheInfo {
     public:
      udim_t l1d;
      udim_t l1i;
      udim_t l2;
      udim_t l3;

      CacheInfo();
    };

    class ProcessorInfo {
     public:
      std::string name;
      udim_t frequency;
      int coreCount;
      CacheInfo cache;

      ProcessorInfo();
    };

    class MemoryInfo {
     public:
      udim_t total;
      udim_t available;

      MemoryInfo();
    };

    class SystemInfo {
     public:
      ProcessorInfo processor;
      MemoryInfo memory;

      SystemInfo();

      static json getSystemInfo();
      static SystemInfo load();

     private:
      static json parseSystemInfoContent(const std::string &content);
      static json getSystemInfoField(const json &systemInfo,
                                     const std::string &field);

      // Processor
      void setProcessorInfo(const json &systemInfo);

      static std::string getProcessorName(const json &systemInfo);
      static udim_t getProcessorFrequency(const json &systemInfo);
      static udim_t getProcessorCacheSize(const json &systemInfo,
                                          CacheLevel level);
      static int getCoreCount(const json &systemInfo);

      // Memory
      void setMemoryInfo(const json &systemInfo);

      static udim_t installedMemory(const json &systemInfo);
      static udim_t availableMemory();
    };
    //==================================

    //---[ Compiler Info ]--------------
    int compilerVendor(const std::string &compiler);

    std::string compilerCpp11Flags(const std::string &compiler);
    std::string compilerCpp11Flags(const int vendor_);

    std::string compilerC99Flags(const std::string &compiler);
    std::string compilerC99Flags(const int vendor_);

    std::string compilerSharedBinaryFlags(const std::string &compiler);
    std::string compilerSharedBinaryFlags(const int vendor_);

    void addCompilerIncludeFlags(std::string &compilerFlags);
    void addCompilerLibraryFlags(std::string &compilerFlags);

    void addCompilerFlags(std::string &compilerFlags, const std::string &flags);
    void addCompilerFlags(std::string &compilerFlags, const strVector &flags);
    //==================================

    //---[ Dynamic Methods ]------------
    void* malloc(udim_t bytes);
    void free(void *ptr);

    void* dlopen(const std::string &filename);

    functionPtr_t dlsym(void *dlHandle,
                        const std::string &functionName);

    void dlclose(void *dlHandle);

    void runFunction(functionPtr_t f, const int argc, void **args);

    std::string stacktrace(const int frameStart = 0,
                           const std::string indent = "");

    std::string prettyStackSymbol(void *frame, const char *symbol);
    //==================================
  }
}

#endif
