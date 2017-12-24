/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#ifndef OCCA_TOOLS_SYS_HEADER
#define OCCA_TOOLS_SYS_HEADER

#include <iostream>
#include <sstream>

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/hash.hpp"

namespace occa {
  typedef void (*handleFunction_t)(...);

  namespace flags {
    extern const int checkCacheDir;
  }

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

    void rmdir(const std::string &dir);
    int mkdir(const std::string &dir);
    void mkpath(const std::string &dir);

    bool dirExists(const std::string &dir_);
    bool fileExists(const std::string &filename_,
                    const int flags = 0);

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
                 const hash_t &hash = hash_t(),
                 const std::string &hashTag = "");

    handleFunction_t dlsym(void *dlHandle,
                           const std::string &functionName,
                           const hash_t &hash = hash_t(),
                           const std::string &hashTag = "");

    void dlclose(void *dlHandle);

    void runFunction(handleFunction_t f, const int argc, void **args);

    void printStacktrace(const int frameStart = 0, const std::string indent = "");
    std::string prettyStackSymbol(void *frame, const char *symbol);
    //==================================
  }

  void _message(const std::string &title,
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

  class mutex {
  public:
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    pthread_mutex_t mutexHandle;
#else
    HANDLE mutexHandle;
#endif

    mutex();
    void free();

    void lock();
    void unlock();
  };
}

#endif
