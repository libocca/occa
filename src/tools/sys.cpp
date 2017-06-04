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

#include "occa/defines.hpp"

#if   (OCCA_OS & OCCA_LINUX_OS)
#  include <ctime>
#  include <cxxabi.h>
#  include <dlfcn.h>
#  include <errno.h>
#  include <execinfo.h>
#  include <sys/time.h>
#  include <sys/syscall.h>
#  include <sys/sysctl.h>
#  include <sys/sysinfo.h>
#  include <pthread.h>
#  include <unistd.h>
#elif (OCCA_OS & OCCA_OSX_OS)
#  include <ctime>
#  include <cxxabi.h>
#  include <dlfcn.h>
#  include <execinfo.h>
#  include <mach/mach_host.h>
#  include <sys/syscall.h>
#  include <sys/sysctl.h>
#  include <unistd.h>
#  ifdef __clang__
#    include <CoreServices/CoreServices.h>
#    include <mach/mach_time.h>
#  else
#    include <mach/clock.h>
#    include <mach/mach.h>
#  endif
#else
#  ifndef NOMINMAX
#    define NOMINMAX // Clear min/max macros
#  endif
#  include <windows.h>
#endif

#include <iomanip>
#include <sstream>

#include <sys/types.h>
#include <fcntl.h>

#include "occa/base.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/hash.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/lex.hpp"
#include "occa/tools/misc.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/sys.hpp"
#include "occa/parser/tools.hpp"

namespace occa {
  namespace flags {
    const int checkCacheDir = (1 << 0);
  }

  namespace sys {
    //---[ System Info ]----------------
    double currentTime() {
#if (OCCA_OS & OCCA_LINUX_OS)
      timespec ct;
      clock_gettime(CLOCK_MONOTONIC, &ct);

      return (double) (ct.tv_sec + (1.0e-9 * ct.tv_nsec));
#elif (OCCA_OS == OCCA_OSX_OS)
#  ifdef __clang__
      uint64_t ct;
      ct = mach_absolute_time();

      const Nanoseconds ct2 = AbsoluteToNanoseconds(*(AbsoluteTime *) &ct);

      return ((double) 1.0e-9) * ((double) ( *((uint64_t*) &ct2) ));
#  else
      clock_serv_t cclock;
      host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);

      mach_timespec_t ct;
      clock_get_time(cclock, &ct);

      mach_port_deallocate(mach_task_self(), cclock);

      return (double) (ct.tv_sec + (1.0e-9 * ct.tv_nsec));
#  endif
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      static LARGE_INTEGER freq;
      static bool haveFreq = false;

      if (!haveFreq) {
        QueryPerformanceFrequency(&freq);
        haveFreq=true;
      }

      LARGE_INTEGER ct;

      QueryPerformanceCounter(&ct);

      return ((double) (ct.QuadPart)) / ((double) (freq.QuadPart));
#endif
    }

    std::string date() {
      ::time_t time_ = ::time(0);
      struct ::tm &timeInfo = *(::localtime(&time_));
      const int year  = timeInfo.tm_year + 1900;
      const int month = timeInfo.tm_mon + 1;
      const int day   = timeInfo.tm_mday;
      const int hour  = timeInfo.tm_hour;
      const int min   = timeInfo.tm_min;
      const int sec   = timeInfo.tm_sec;

      std::stringstream ss;
      ss << year << '/';
      if (month < 10) ss << '0';
      ss << month << '/';
      if (day   < 10) ss << '0';
      ss << day << ' ';
      if (hour  < 10) ss << '0';
      ss << hour << ':';
      if (min   < 10) ss << '0';
      ss << min << ':';
      if (sec   < 10) ss << '0';
      ss << sec;
      return ss.str();
    }

    std::string humanDate() {
      ::time_t time_ = ::time(0);
      struct ::tm &timeInfo = *(::localtime(&time_));
      const int year  = timeInfo.tm_year + 1900;
      const int month = timeInfo.tm_mon + 1;
      const int day   = timeInfo.tm_mday;
      const int hour  = timeInfo.tm_hour;
      const int min   = timeInfo.tm_min;

      std::stringstream ss;

      switch (month) {
      case 0 : ss << "Jan"; break;
      case 1 : ss << "Feb"; break;
      case 2 : ss << "Mar"; break;
      case 3 : ss << "Apr"; break;
      case 4 : ss << "May"; break;
      case 5 : ss << "Jun"; break;
      case 6 : ss << "Jul"; break;
      case 7 : ss << "Aug"; break;
      case 8 : ss << "Sep"; break;
      case 9 : ss << "Oct"; break;
      case 10: ss << "Nov"; break;
      case 11: ss << "Dec"; break;
      }

      ss << ' ' << day << ' ' << year << ' ';
      if (hour < 10) ss << '0';
      ss << hour << ':';
      if (min  < 10) ss << '0';
      ss << min;

      return ss.str();
    }
    //==================================

    //---[ System Calls ]---------------
    int call(const std::string &cmdline) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      FILE *fp = popen(cmdline.c_str(), "r");
      return pclose(fp);
#else
      FILE *fp = _popen(cmdline.c_str(), "r");
      return _pclose(fp);
#endif
    }

    int call(const std::string &cmdline, std::string &output) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      FILE *fp = popen(cmdline.c_str(), "r");
#else
      FILE *fp = _popen(cmdline.c_str(), "r");
#endif

      size_t lineBytes = 512;
      char lineBuffer[512];

      while (fgets(lineBuffer, lineBytes, fp)) {
        output += lineBuffer;
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      return pclose(fp);
#else
      return _pclose(fp);
#endif
    }

    std::string expandEnvVariables(const std::string &str) {
      const char *c = str.c_str();
      const char *c0 = c;
      std::string expstr;

      while (*c != '\0') {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
        if ((*c == '$') && ((c0 < c) || (*(c - 1) != '\\'))) {
          if (*(c + 1) == '{') {
            const char *cStart = c + 2;
            skipTo(c, '}');

            if (*c == '\0')
              return expstr;

            expstr += env::var(std::string(cStart, c - cStart));
          } else {
            const char *cStart = c + 1;
            skipTo(c, '/');
            expstr += env::var(std::string(cStart, c - cStart));
          }
        }
#else
        if (*c == '%') {
          const char *cStart = (++c);
          skipTo(c, '%');
          expstr += env::var(std::string(cStart, c - cStart));
        }
#endif
        else {
          expstr += *c;
        }
        ++c;
      }

      return expstr;
    }

    void rmdir(const std::string &dir) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      ::rmdir(dir.c_str());
#else
      ::_rmdir(dir.c_str());
#endif
    }

    int mkdir(const std::string &dir) {
      errno = 0;

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      return ::mkdir(dir.c_str(), 0755);
#else
      return ::_mkdir(dir.c_str());
#endif
    }

    void mkpath(const std::string &dir) {
      strVector_t path = split(io::filename(dir), '/');

      const int dirCount = (int) path.size();
      std::string sPath;
      int makeFrom = -1;

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      const int firstDir = 0;
      if (dirCount == 0)
        return;
      sPath += '/';
#else
      const int firstDir = 1;
      if (dirCount <= 1)
        return;
      sPath += path[0];
      sPath += '/';
#endif

      for (int d = firstDir; d < dirCount; ++d) {
        sPath += path[d];
        if (!dirExists(sPath)) {
          makeFrom = d;
          break;
        }
        sPath += '/';
      }

      if (0 < makeFrom) {
        sys::mkdir(sPath);

        for (int d = (makeFrom + 1); d < dirCount; ++d) {
          sPath += '/';
          sPath += path[d];

          sys::mkdir(sPath);
        }
      }
    }

    bool dirExists(const std::string &dir_) {
      std::string dir = expandEnvVariables(dir_);
      strip(dir);

      struct stat statInfo;
      return ((stat(dir.c_str(), &statInfo) == 0) &&
              S_ISDIR(statInfo.st_mode));
    }

    bool fileExists(const std::string &filename_,
                    const int flags) {

      std::string filename = expandEnvVariables(filename_);
      strip(filename);

      if (flags & flags::checkCacheDir)
        return fileExists(io::filename(filename));

      struct stat statInfo;

      return (stat(filename.c_str(), &statInfo) == 0);
    }

    int getPID() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      return getpid();
#else
      return GetCurrentProcessId();
#endif
    }

    int getTID() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      return syscall(SYS_gettid);
#else
      return GetCurrentThreadId();
#endif
    }

    void pinToCore(const int core) {
      const int coreCount = getCoreCount();
      OCCA_ERROR("Core to pin (" << core << ") is not in range: [0, "
                 << coreCount << "]",
                 (0 <= core) && (core < coreCount));
#if (OCCA_OS == OCCA_LINUX_OS)
      cpu_set_t cpuSet;
      CPU_ZERO(&cpuSet);
      CPU_SET(core, &cpuSet);
      syscall(__NR_sched_setaffinity, getTID(), sizeof(cpu_set_t), &cpuSet);
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      SetThreadAffinityMask(GetCurrentThread(), 1 << core);
#endif
    }
    //==================================

    //---[ Processor Info ]-------------
    std::string getFieldFrom(const std::string &command,
                             const std::string &field) {
#if (OCCA_OS & LINUX)
      std::string shellToolsFile = io::filename("occa://occa/scripts/shellTools.sh");

      if (!sys::fileExists(shellToolsFile)) {
        sys::mkpath(dirname(shellToolsFile));

        std::ofstream fs2;
        fs2.open(shellToolsFile.c_str());

        fs2 << getCachedScript("shellTools.sh");

        fs2.close();
      }

      std::stringstream ss;

      ss << "echo \"(. " << shellToolsFile << "; " << command << " '" << field << "')\" | bash";

      std::string sCommand = ss.str();

      FILE *fp;
      fp = popen(sCommand.c_str(), "r");

      const int bufferSize = 4096;
      char *buffer = new char[bufferSize];

      ignoreResult( fread(buffer, sizeof(char), bufferSize, fp) );

      pclose(fp);

      int end;

      for (end = 0; end < bufferSize; ++end) {
        if (buffer[end] == '\n')
          break;
      }

      std::string ret(buffer, end);

      delete [] buffer;

      return ret;
#else
      return "";
#endif
    }

    std::string getProcessorName() {
#if   (OCCA_OS & OCCA_LINUX_OS)
      return getFieldFrom("getCPUINFOField", "model name");
#elif (OCCA_OS == OCCA_OSX_OS)
      size_t bufferSize = 100;
      char buffer[100];

      sysctlbyname("machdep.cpu.brand_string",
                   &buffer, &bufferSize,
                   NULL, 0);

      return std::string(buffer);
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      char buffer[MAX_COMPUTERNAME_LENGTH + 1];
      int bytes;

      GetComputerName((LPSTR) buffer, (LPDWORD) &bytes);

      return std::string(buffer, bytes);
#endif
    }

    int getCoreCount() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      return sysconf(_SC_NPROCESSORS_ONLN);
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      SYSTEM_INFO sysinfo;
      GetSystemInfo(&sysinfo);
      return sysinfo.dwNumberOfProcessors;
#endif
    }

    int getProcessorFrequency() {
#if   (OCCA_OS & OCCA_LINUX_OS)
      std::stringstream ss;
      int freq;

      ss << getFieldFrom("getCPUINFOField", "cpu MHz");

      ss >> freq;

      return freq;
#elif (OCCA_OS == OCCA_OSX_OS)
      uint64_t frequency = 0;
      size_t size = sizeof(frequency);

      int error = sysctlbyname("hw.cpufrequency", &frequency, &size, NULL, 0);

      OCCA_ERROR("Error getting CPU Frequency.\n",
                 error != ENOMEM);

      return frequency/1.0e6;
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      LARGE_INTEGER performanceFrequency;
      QueryPerformanceFrequency(&performanceFrequency);

      return (int) (((double) performanceFrequency.QuadPart) / 1000.0);
#endif
    }

    std::string getProcessorCacheSize(int level) {
#if   (OCCA_OS & OCCA_LINUX_OS)
      std::stringstream field;

      field << 'L' << level;

      if (level == 1)
        field << 'd';

      field << " cache";

      return getFieldFrom("getLSCPUField", field.str());
#elif (OCCA_OS == OCCA_OSX_OS)
      std::stringstream ss;
      ss << "hw.l" << level;

      if (level == 1)
        ss << 'd';

      ss << "cachesize";

      std::string field = ss.str();

      uint64_t cache = 0;
      size_t size = sizeof(cache);

      int error = sysctlbyname(field.c_str(), &cache, &size, NULL, 0);

      OCCA_ERROR("Error getting L" << level << " Cache Size.\n",
                 error != ENOMEM);

      return stringifyBytes(cache);
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      std::stringstream ss;
      DWORD cache = 0;
      int bytes = 0;

      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;

      GetLogicalProcessorInformation(buffer, (LPDWORD) &bytes);

      OCCA_ERROR("[GetLogicalProcessorInformation] Failed",
                 (GetLastError() == ERROR_INSUFFICIENT_BUFFER));

      buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION) sys::malloc(bytes);

      bool passed = GetLogicalProcessorInformation(buffer, (LPDWORD) &bytes);

      OCCA_ERROR("[GetLogicalProcessorInformation] Failed",
                 passed);

      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION pos = buffer;
      int off = 0;
      int sk = sizeof(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION);

      while ((off + sk) <= bytes) {
        switch(pos->Relationship) {
        case RelationCache:{
          CACHE_DESCRIPTOR info = pos->Cache;

          if (info.Level == level) {
            cache = info.Size;
            break;
          }
        }
        }
        ++pos;
        off += sk;
      }

      sys::free(buffer);

      return stringifyBytes(cache);
#endif
    }

    udim_t installedRAM() {
#if   (OCCA_OS & OCCA_LINUX_OS)
      struct sysinfo info;

      const int error = sysinfo(&info);

      if (error != 0)
        return 0;

      return info.totalram;
#elif (OCCA_OS == OCCA_OSX_OS)
      int64_t ram;

      int mib[2]   = {CTL_HW, HW_MEMSIZE};
      size_t bytes = sizeof(ram);

      sysctl(mib, 2, &ram, &bytes, NULL, 0);

      return ram;
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      return 0;
#endif
    }

    udim_t availableRAM() {
#if   (OCCA_OS & OCCA_LINUX_OS)
      struct sysinfo info;

      const int error = sysinfo(&info);

      if (error != 0)
        return 0;

      return info.freeram;
#elif (OCCA_OS == OCCA_OSX_OS)
      mach_msg_type_number_t infoCount = HOST_VM_INFO_COUNT;
      mach_port_t hostPort = mach_host_self();

      vm_statistics_data_t hostInfo;
      kern_return_t status;
      vm_size_t pageSize;

      status = host_page_size(hostPort, &pageSize);

      if (status != KERN_SUCCESS)
        return 0;

      status = host_statistics(hostPort,
                               HOST_VM_INFO,
                               (host_info_t) &hostInfo,
                               &infoCount);

      if (status != KERN_SUCCESS)
        return 0;

      return (hostInfo.free_count * pageSize);
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      return 0;
#endif
    }

    int compilerVendor(const std::string &compiler) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      const std::string safeCompiler = io::removeSlashes(compiler);
      int vendor_ = sys::vendor::notFound;
      std::stringstream ss;

      const std::string compilerVendorTest = env::OCCA_DIR + "/scripts/compilerVendorTest.cpp";
      hash_t hash = occa::hashFile(compilerVendorTest);
      hash ^= occa::hash(vendor_);
      hash ^= occa::hash(compiler);

      const std::string srcFilename = io::cacheFile(compilerVendorTest, "compilerVendorTest.cpp", hash);
      const std::string hashDir = io::dirname(srcFilename);
      const std::string binaryFilename   = hashDir + "binary";
      const std::string outFilename      = hashDir + "output";
      const std::string buildLogFilename = hashDir + "build.log";

      const std::string hashTag = "compiler";
      if (!io::haveHash(hash, hashTag)) {
        io::waitForHash(hash, hashTag);
      } else {
        if (!sys::fileExists(outFilename)) {
          ss << compiler
             << ' '    << srcFilename
             << " -o " << binaryFilename
             << " > " << buildLogFilename << " 2>&1";
          const std::string compileLine = ss.str();

          if (settings().get("verboseCompilation", true)) {
            std::cout << "Finding compiler vendor: " << compileLine << '\n';
          }

          system(compileLine.c_str());

          OCCA_ERROR("Could not compile compilerVendorTest.cpp with following command:\n" << compileLine,
                     sys::fileExists(binaryFilename));

          int exitStatus = system(binaryFilename.c_str());
          int vendorBit  = WEXITSTATUS(exitStatus);

          if (vendorBit < sys::vendor::b_max) {
            vendor_ = (1 << vendorBit);
          }

          ss.str("");
          ss << vendor_;

          io::write(outFilename, ss.str());
          io::releaseHash(hash, hashTag);

          return vendor_;
        }
        io::releaseHash(hash, hashTag);
      }

      ss << io::read(outFilename);
      ss >> vendor_;

      return vendor_;

#elif (OCCA_OS == OCCA_WINDOWS_OS)
#  if OCCA_USING_VS
      return sys::vendor::VisualStudio;
#  endif

      if (compiler.find("cl.exe") != std::string::npos) {
        return sys::vendor::VisualStudio;
      }
#endif
    }

    std::string compilerSharedBinaryFlags(const std::string &compiler) {
      return compilerSharedBinaryFlags( sys::compilerVendor(compiler) );
    }

    std::string compilerSharedBinaryFlags(const int vendor_) {
      if (vendor_ & (sys::vendor::GNU   |
                     sys::vendor::LLVM  |
                     sys::vendor::Intel |
                     sys::vendor::IBM   |
                     sys::vendor::PGI   |
                     sys::vendor::Cray  |
                     sys::vendor::Pathscale)) {

        return "-x c++ -fPIC -shared";
      } else if (vendor_ & sys::vendor::HP) {
        return "+z -b";
      } else if (vendor_ & sys::vendor::VisualStudio) {
#if OCCA_DEBUG_ENABLED
        return "/TP /LD /MDd";
#else
        return "/TP /LD /MD";
#endif
      }

      return "";
    }

    void addSharedBinaryFlagsTo(const std::string &compiler, std::string &flags) {
      addSharedBinaryFlagsTo(sys::compilerVendor(compiler), flags);
    }

    void addSharedBinaryFlagsTo(const int vendor_, std::string &flags) {
      std::string sFlags = sys::compilerSharedBinaryFlags(vendor_);

      if (flags.size() == 0) {
        flags = sFlags;
      }
      if (flags.find(sFlags) == std::string::npos) {
        flags = (sFlags + " " + flags);
      }
    }

    //---[ Dynamic Methods ]------------
    void* malloc(udim_t bytes) {
      void* ptr;

#if   (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      ignoreResult( posix_memalign(&ptr, env::OCCA_MEM_BYTE_ALIGN, bytes) );
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      ptr = ::malloc(bytes);
#endif

      return ptr;
    }

    void free(void *ptr) {
      ::free(ptr);
    }

    void* dlopen(const std::string &filename,
                 const hash_t &hash,
                 const std::string &hashTag) {

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      void *dlHandle = ::dlopen(filename.c_str(), RTLD_NOW);

      if ((dlHandle == NULL) && hash.initialized) {
        io::releaseHash(hash, hashTag);

        OCCA_ERROR("Error loading binary [" << io::shortname(filename) << "] with dlopen",
                   false);
      }
#else
      void *dlHandle = LoadLibraryA(filename.c_str());

      if ((dlHandle == NULL) && hash.initialized) {
        io::releaseHash(hash, hashTag);

        OCCA_ERROR("Error loading dll [" << io::shortname(filename) << "] (WIN32 error: " << GetLastError() << ")",
                   dlHandle != NULL);
      }
#endif

      return dlHandle;
    }

    handleFunction_t dlsym(void *dlHandle,
                           const std::string &functionName,
                           const hash_t &hash,
                           const std::string &hashTag) {

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      void *sym = ::dlsym(dlHandle, functionName.c_str());

      char *dlError;

      if (((dlError = dlerror()) != NULL) && hash.initialized) {
        io::releaseHash(hash, hashTag);

        OCCA_ERROR("Error loading symbol from binary with dlsym (DL Error: " << dlError << ")",
                   false);
      }
#else
      void *sym = GetProcAddress((HMODULE) dlHandle, functionName.c_str());

      if ((sym == NULL) && hash.initialized) {

        OCCA_ERROR("Error loading symbol from binary with GetProcAddress",
                   false);
      }
#endif

      handleFunction_t sym2;

      ::memcpy(&sym2, &sym, sizeof(sym));

      return sym2;
    }

    void dlclose(void *dlHandle) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      ::dlclose(dlHandle);
#else
      FreeLibrary((HMODULE) (dlHandle));
#endif
    }

    void runFunction(handleFunction_t f, const int argc, void **args) {
#include "operators/runFunctionFromArguments.cpp"
    }

    void printStacktrace(const int frameStart, const std::string indent) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      static const int maxFrames = 1024;
      static void *frames[maxFrames];

      const int frameCount = ::backtrace(frames, maxFrames);
      char **symbols = ::backtrace_symbols(frames, frameCount);

      const int digits = toString(frameCount - frameStart).size();

      for (int i = frameStart; i < frameCount; ++i) {
        const std::string localFrame = toString(frameCount - i);
        std::cout << indent
                  << localFrame << std::string(digits - localFrame.size() + 1, ' ')
                  << prettyStackSymbol(frames[i], symbols[i]) << '\n';
      }
      ::free(symbols);
#endif
    }

    std::string prettyStackSymbol(void *frame, const char *symbol) {
      static size_t maxChars = 1024;
      static char prettyBuffer[1024];
      std::stringstream ss;

#if (OCCA_OS == OCCA_OSX_OS)
      const char *c = symbol;
      // Skip stack depth
      lex::skipBetweenWhitespaces(c);
      // Get origin
      const char *originStart = c;
      lex::skipToWhitespace(c);
      std::string origin(originStart, (c - originStart));
      // Skip address
      lex::skipBetweenWhitespaces(c);
      // Get function name
      const char *functionStart = c;
      lex::skipToWhitespace(c);
      std::string function(functionStart, (c - functionStart));
      // Skip the +
      lex::skipBetweenWhitespaces(c);
      // Get address offset
      const char *offsetStart = c;
      lex::skipToWhitespace(c);
      std::string offset(offsetStart, (c - offsetStart));

      int status;
      const char *prettyFunction = abi::__cxa_demangle(function.c_str(),
                                                       prettyBuffer,
                                                       &maxChars,
                                                       &status);

      ss << std::left << std::setw(20) << origin
         << std::left << std::setw(50) << (status ? function : prettyFunction);
      return ss.str();
#elif (OCCA_OS == OCCA_LINUX_OS)
      std::string function;

      Dl_info frameInfo;
      int status = dladdr(frame, &frameInfo);
      const char *dl_name = frameInfo.dli_sname;

      if (status && dl_name) {
        const char *prettyFunction = abi::__cxa_demangle(dl_name,
                                                         prettyBuffer,
                                                         &maxChars,
                                                         &status);
        if (!status) {
          function = std::string(prettyFunction);
        }
      }
      if (function.size() == 0) {
        const char *c = symbol;
        // Get function name
        lex::skipWhitespace(c);
        const char *functionStart = c;
        lex::skipToWhitespace(c);
        function = std::string(functionStart, (c - functionStart));
      }
      return function;
#else
      return std::string(c);
#endif
    }
  }

  void _message(const std::string &title,
                const bool exitInFailure,
                const std::string &filename,
                const std::string &function,
                const int line,
                const std::string &message) {

    std::string header = "---[ " + title + " ]";
    header += std::string(60 - header.size(), '-');

    std::cerr << '\n'
              << header << '\n'
              << "    File     : " << filename << '\n'
              << "    Function : " << function << '\n'
              << "    Line     : " << line     << '\n';
    if (message.size()) {
      std::cerr << "    Message  : " << message << '\n';
    }
    std::cerr << "    Stack    :\n";
    sys::printStacktrace(3, "      ");
    std::cerr << std::string(60, '=') << '\n';

    if (exitInFailure) {
      throw 1;
    }
  }

  void warn(const std::string &filename,
            const std::string &function,
            const int line,
            const std::string &message) {
    _message("Warning", false,
             filename, function, line, message);
  }

  void error(const std::string &filename,
             const std::string &function,
             const int line,
             const std::string &message) {
    _message("Error", true,
             filename, function, line, message);
  }

  mutex_t::mutex_t() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    int error = pthread_mutex_init(&mutexHandle, NULL);

    OCCA_ERROR("Error initializing mutex",
               error == 0);
#else
    mutexHandle = CreateMutex(NULL, FALSE, NULL);
#endif
  }

  void mutex_t::free() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    int error = pthread_mutex_destroy(&mutexHandle);

    OCCA_ERROR("Error freeing mutex",
               error == 0);
#else
    CloseHandle(mutexHandle);
#endif
  }

  void mutex_t::lock() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    pthread_mutex_lock(&mutexHandle);
#else
    WaitForSingleObject(mutexHandle, INFINITE);
#endif
  }

  void mutex_t::unlock() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    pthread_mutex_unlock(&mutexHandle);
#else
    ReleaseMutex(mutexHandle);
#endif
  }
}
