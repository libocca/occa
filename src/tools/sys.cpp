#include <occa/defines.hpp>

#include <fstream>

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
#  include <ctime>
#  include <cxxabi.h>
#  include <dlfcn.h>
#  include <execinfo.h>
#  include <pthread.h>
#  include <signal.h>
#  include <stdio.h>
#  include <sys/types.h>
#  include <sys/stat.h>
#  include <sys/syscall.h>
#  include <sys/sysctl.h>
#  include <sys/time.h>
#  include <unistd.h>
#  if (OCCA_OS & OCCA_LINUX_OS)
#    include <errno.h>
#    include <sys/sysinfo.h>
#  else // OCCA_MACOS_OS
#    include <mach/mach_host.h>
#    ifdef __clang__
#      include <CoreServices/CoreServices.h>
#      include <mach/mach_time.h>
#    else
#      include <mach/clock.h>
#      include <mach/mach.h>
#    endif
#  endif
#else // OCCA_WINDOWS_OS
#  include <windows.h>
#endif

#include <iomanip>
#include <sstream>

#include <sys/types.h>
#include <fcntl.h>

#include <occa/core/base.hpp>
#include <occa/io.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/hash.hpp>
#include <occa/tools/exception.hpp>
#include <occa/tools/lex.hpp>
#include <occa/tools/misc.hpp>
#include <occa/tools/string.hpp>
#include <occa/tools/sys.hpp>
#include <occa/tools/vector.hpp>

namespace occa {
  namespace sys {
    //---[ System Info ]----------------
    double currentTime() {
      // Returns the current time in seconds
#if (OCCA_OS & OCCA_LINUX_OS)
      timespec ct;
      clock_gettime(CLOCK_MONOTONIC, &ct);

      return (double) (ct.tv_sec + (1.0e-9 * ct.tv_nsec));
#elif (OCCA_OS == OCCA_MACOS_OS)
#  ifdef __clang__
      uint64_t nanoseconds = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);

      return 1.0e-9 * nanoseconds;
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
      case 1 : ss << "Jan"; break;
      case 2 : ss << "Feb"; break;
      case 3 : ss << "Mar"; break;
      case 4 : ss << "Apr"; break;
      case 5 : ss << "May"; break;
      case 6 : ss << "Jun"; break;
      case 7 : ss << "Jul"; break;
      case 8 : ss << "Aug"; break;
      case 9 : ss << "Sep"; break;
      case 10: ss << "Oct"; break;
      case 11: ss << "Nov"; break;
      case 12: ss << "Dec"; break;
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
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      FILE *fp = popen(cmdline.c_str(), "r");
      return pclose(fp);
#else
      FILE *fp = _popen(cmdline.c_str(), "r");
      return _pclose(fp);
#endif
    }

    int call(const std::string &cmdline, std::string &output) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      FILE *fp = popen(cmdline.c_str(), "r");
#else
      FILE *fp = _popen(cmdline.c_str(), "r");
#endif

      size_t lineBytes = 512;
      char lineBuffer[512];

      while (fgets(lineBuffer, lineBytes, fp)) {
        output += lineBuffer;
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
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
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        if ((*c == '$') && ((c0 < c) || (*(c - 1) != '\\'))) {
          if (*(c + 1) == '{') {
            const char *cStart = c + 2;
            lex::skipTo(c, '}');

            if (*c == '\0') {
              return expstr;
            }
            expstr += env::var(std::string(cStart, c - cStart));
          } else {
            const char *cStart = c + 1;
            lex::skipTo(c, '/');
            expstr += env::var(std::string(cStart, c - cStart));
            if (*c) {
              expstr += '/';
            }
          }
        }
#else
        if (*c == '%') {
          const char *cStart = (++c);
          lex::skipTo(c, '%');
          expstr += env::var(std::string(cStart, c - cStart));
        }
#endif
        else {
          expstr += *c;
        }
        if (*c) {
          ++c;
        }
      }

      return expstr;
    }

    void rmdir(const std::string &dir,
               const bool recursive) {
      if (recursive) {
        // Remove files
        strVector files = io::files(dir);
        const int fileCount = (int) files.size();
        for (int i = 0; i < fileCount; ++i) {
          ::remove(files[i].c_str());
        }
        // Remove directories
        strVector directories = io::directories(dir);
        const int dirCount = (int) directories.size();
        for (int i = 0; i < dirCount; ++i) {
          rmdir(directories[i], true);
        }
      }
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      ::rmdir(dir.c_str());
#else
      ::_rmdir(dir.c_str());
#endif
    }

    void rmrf(const std::string &filename) {
      if (io::isFile(filename)) {
        ::remove(filename.c_str());
        return;
      }

      // Make sure we're not deleting /
      if (settings().get("sys/safe_rmrf", true)) {
        OCCA_ERROR("For safety, not deleting [" << filename << "]."
                   " To disable this error, set 'sys/safe_rmrf' settings to false",
                   isSafeToRmrf(filename));
      }
      rmdir(filename, true);
    }

    bool isSafeToRmrf(const std::string &filename) {
      const std::string expFilename = io::filename(filename);
      int depth = 0;

      strVector path = split(expFilename, '/', '\0');
      bool foundOcca = false;
      const int pathSize = (int) path.size();
      for (int i = 0; i < pathSize; ++i) {
        const std::string &dir = path[i];

        foundOcca |= (
          dir == "occa"
          || dir == ".occa"
          || startsWith(dir, "occa_")
          || startsWith(dir, ".occa_")
        );

        if (!dir.size() ||
            (dir == ".")) {
          continue;
        }
        if (dir == "..") {
          depth -= (depth > 0);
        } else {
          ++depth;
        }
      }

      return depth > 1 && foundOcca;
    }

    int mkdir(const std::string &dir) {
      errno = 0;
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      return ::mkdir(dir.c_str(), 0755);
#else
      return ::_mkdir(dir.c_str());
#endif
    }

    void mkpath(const std::string &dir) {
      strVector path = split(io::filename(dir), '/', '\0');

      const int dirCount = (int) path.size();
      std::string sPath;
      int makeFrom = -1;

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
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
        if (!io::isDir(sPath)) {
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

    bool pidExists(const int pid) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      return !::kill(pid, 0);
#else
      HANDLE hProc = OpenProcess(SYNCHRONIZE, FALSE, pid);
      if (!hProc) {
        return false; // Process has closed
      }
      DWORD ret = WaitForSingleObject(hProc, 0);
      CloseHandle(hProc);
      return (ret != WAIT_TIMEOUT);
#endif
    }

    int getPID() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      return getpid();
#else
      return GetCurrentProcessId();
#endif
    }

    int getTID() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      #if OCCA_OS == OCCA_MACOS_OS & (MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_12)
      uint64_t tid64;
      pthread_threadid_np(NULL, &tid64);
      pid_t tid = (pid_t)tid64;
      #else
      pid_t tid = syscall(SYS_gettid);
      #endif
      return tid;
#else
      return GetCurrentThreadId();
#endif
    }

    void pinToCore(const int core) {
      const int coreCount = getCoreCount();
#if OCCA_UNSAFE
    ignoreResult(coreCount);
#endif
      OCCA_ERROR("Core to pin (" << core << ") is not in range: [0, "
                 << coreCount << "]",
                 (0 <= core) && (core < coreCount));
#if (OCCA_OS == OCCA_LINUX_OS)
      cpu_set_t cpuSet;
      CPU_ZERO(&cpuSet);
      CPU_SET(core, &cpuSet);
      syscall(__NR_sched_setaffinity, getTID(), sizeof(cpu_set_t), &cpuSet);
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      SetThreadAffinityMask(GetCurrentThread(), ((uint64_t) 1) << core);
#endif
    }
    //==================================

    //---[ Processor Info ]-------------
    std::string getFieldFrom(const std::string &command,
                             const std::string &field) {
#if (OCCA_OS & OCCA_LINUX_OS)
      std::string shellToolsFile = env::OCCA_DIR + "include/occa/scripts/shellTools.sh";

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
#elif (OCCA_OS == OCCA_MACOS_OS)
      size_t bufferSize = 100;
      char buffer[100];

      sysctlbyname("machdep.cpu.brand_string",
                   &buffer, &bufferSize,
                   NULL, 0);

      return std::string(buffer);
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      char buffer[MAX_COMPUTERNAME_LENGTH + 1];
      int bytes;

      GetComputerName((LPTSTR) buffer, (LPDWORD) &bytes);

      return std::string(buffer, bytes);
#endif
    }

    int getCoreCount() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
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
      float freq = 0;

      ss << getFieldFrom("getLSCPUField", "cpu.*mhz");
      ss >> freq;

      return (int) freq;
#elif (OCCA_OS == OCCA_MACOS_OS)
      uint64_t frequency = 0;
      size_t size = sizeof(frequency);

      int error = sysctlbyname("hw.cpufrequency", &frequency, &size, NULL, 0);
#if OCCA_UNSAFE
    ignoreResult(error);
#endif

      OCCA_ERROR("Error getting CPU Frequency.\n",
                 error != ENOMEM);

      return frequency / 1.0e6;
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
#elif (OCCA_OS == OCCA_MACOS_OS)
      std::stringstream ss;
      ss << "hw.l" << level;

      if (level == 1)
        ss << 'd';

      ss << "cachesize";

      std::string field = ss.str();

      uint64_t cache = 0;
      size_t size = sizeof(cache);

      int error = sysctlbyname(field.c_str(), &cache, &size, NULL, 0);
#if OCCA_UNSAFE
    ignoreResult(error);
#endif

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
        if (pos->Relationship == RelationCache) {
          CACHE_DESCRIPTOR info = pos->Cache;
          if (info.Level == level) {
            cache = info.Size;
            break;
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

      if (error != 0) {
        return 0;
      }

      return info.totalram;
#elif (OCCA_OS == OCCA_MACOS_OS)
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

      if (error != 0) {
        return 0;
      }

      return info.freeram;
#elif (OCCA_OS == OCCA_MACOS_OS)
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
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      const std::string safeCompiler = io::slashToSnake(compiler);
      int vendor_ = sys::vendor::notFound;
      std::stringstream ss;

      const std::string compilerVendorTest = env::OCCA_DIR + "include/occa/scripts/tests/compiler.cpp";
      hash_t hash = occa::hashFile(compilerVendorTest);
      hash ^= occa::hash(compiler);

      const std::string srcFilename = io::cacheFile(compilerVendorTest,
                                                    "compilerVendorTest.cpp",
                                                    hash);
      const std::string &hashDir = io::dirname(srcFilename);
      const std::string binaryFilename   = hashDir + "binary";
      const std::string outFilename      = hashDir + "output";
      const std::string buildLogFilename = hashDir + "build.log";

      bool foundOutput = (
        io::cachedFileIsComplete(hashDir, "output")
        && io::isFile(outFilename)
      );

      // Avoid creating lockfile if possible
      if (!foundOutput) {
        io::lock_t lock(hash, "compiler");
        if (lock.isMine()) {
          ss << compiler
             << ' '    << srcFilename
             << " -o " << binaryFilename
             << " > " << buildLogFilename << " 2>&1";
          const std::string compileLine = ss.str();

          ignoreResult( system(compileLine.c_str()) );

          OCCA_ERROR("Could not compile compilerVendorTest.cpp with following command:\n" << compileLine,
                     io::isFile(binaryFilename));

          int exitStatus = system(binaryFilename.c_str());
          int vendorBit  = WEXITSTATUS(exitStatus);

          if (vendorBit < sys::vendor::b_max) {
            vendor_ = (1 << vendorBit);
          }

          io::write(outFilename, std::to_string(vendor_));
          io::markCachedFileComplete(hashDir, "output");

          return vendor_;
        }
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

    std::string compilerCpp11Flags(const std::string &compiler) {
      return compilerCpp11Flags( sys::compilerVendor(compiler) );
    }

    std::string compilerCpp11Flags(const int vendor_) {
      if (vendor_ & (sys::vendor::GNU   |
                     sys::vendor::LLVM  |
                     sys::vendor::Intel |
                     sys::vendor::HP    |
                     sys::vendor::PGI   |
                     sys::vendor::PPC   |
                     sys::vendor::Pathscale)) {
        return "-std=c++11";
      } else if (vendor_ & sys::vendor::Cray) {
        return "-hstd=c++11";
      } else if (vendor_ & sys::vendor::IBM) {
        return "-qlanglvl=extended0x";
      } else if (vendor_ & sys::vendor::VisualStudio) {
        return ""; // Defaults to C++14
      }
      OCCA_FORCE_ERROR("Could not find C++11 compiler flags");
      return "";
    }

    std::string compilerC99Flags(const std::string &compiler) {
      return compilerC99Flags( sys::compilerVendor(compiler) );
    }

    std::string compilerC99Flags(const int vendor_) {
      if (vendor_ & (sys::vendor::GNU   |
                     sys::vendor::LLVM  |
                     sys::vendor::Intel |
                     sys::vendor::HP    |
                     sys::vendor::PGI   |
                     sys::vendor::PPC   |
                     sys::vendor::Pathscale)) {
        return "-std=c99";
      } else if (vendor_ & sys::vendor::Cray) {
        return "-hstd=c99";
      } else if (vendor_ & sys::vendor::IBM) {
        return "-qlanglvl=stdc99";
      } else if (vendor_ & sys::vendor::VisualStudio) {
        return ""; // Defaults to C++14
      }
      OCCA_FORCE_ERROR("Could not find C99 compiler flags");
      return "";
    }

    std::string compilerSharedBinaryFlags(const std::string &compiler) {
      return compilerSharedBinaryFlags( sys::compilerVendor(compiler) );
    }

    std::string compilerSharedBinaryFlags(const int vendor_) {
      if (vendor_ & (sys::vendor::GNU   |
                     sys::vendor::LLVM  |
                     sys::vendor::Intel |
                     sys::vendor::PGI   |
                     sys::vendor::Cray  |
                     sys::vendor::PPC   |
                     sys::vendor::Pathscale)) {
        return "-fPIC -shared";
      } else if (vendor_ & sys::vendor::IBM) {
        return "-qpic -shared";
      } else if (vendor_ & sys::vendor::HP) {
        return "+z -b";
      } else if (vendor_ & sys::vendor::VisualStudio) {
        return "/TP /LD /MD"; // Note: Use /MDd for debug mode
      }
      OCCA_FORCE_ERROR("Could not find compiler flags for creating a shared object");
      return "";
    }

    void addCompilerIncludeFlags(std::string &compilerFlags) {
      strVector includeDirs = env::OCCA_INCLUDE_PATH;

      const int count = (int) includeDirs.size();
      for (int i = 0; i < count; ++i) {
        includeDirs[i] = "-I" + includeDirs[i];
      }

      addCompilerFlags(compilerFlags, includeDirs);
    }

    void addCompilerLibraryFlags(std::string &compilerFlags) {
      strVector libraryDirs = env::OCCA_LIBRARY_PATH;

      const int count = (int) libraryDirs.size();
      for (int i = 0; i < count; ++i) {
        libraryDirs[i] = "-L" + libraryDirs[i];
      }

      addCompilerFlags(compilerFlags, libraryDirs);
    }

    void addCompilerFlags(std::string &compilerFlags, const std::string &flags) {
      const strVector flagsVec = split(flags, ' ');
      addCompilerFlags(compilerFlags, flagsVec);
    }

    void addCompilerFlags(std::string &compilerFlags, const strVector &flags) {
      strVector compilerFlagsVec = split(compilerFlags, ' ');

      const int flagCount = (int) flags.size();
      for (int i = 0; i < flagCount; ++i) {
        const std::string &flag = flags[i];
        if (indexOf(compilerFlagsVec, flag) < 0) {
          compilerFlagsVec.push_back(flag);
        }
      }

      compilerFlags = join(compilerFlagsVec, " ");
    }

    //---[ Dynamic Methods ]------------
    void* malloc(udim_t bytes) {
      void* ptr;

#if   (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
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
                 const io::lock_t &lock) {

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      void *dlHandle = ::dlopen(filename.c_str(),
                                RTLD_NOW | RTLD_LOCAL);
      if (dlHandle == NULL) {
        lock.release();
        char *error = dlerror();
        if (error) {
          OCCA_FORCE_ERROR("Error loading binary [" << io::shortname(filename) << "] with dlopen: " << error);
        } else {
          OCCA_FORCE_ERROR("Error loading binary [" << io::shortname(filename) << "] with dlopen");
        }
      }
#else
      void *dlHandle = LoadLibraryA(filename.c_str());

      if (dlHandle == NULL) {
        lock.release();
        OCCA_ERROR("Error loading .dll [" << io::shortname(filename) << "]: " << GetLastError(),
                   dlHandle != NULL);
      }
#endif

      return dlHandle;
    }

    functionPtr_t dlsym(void *dlHandle,
                        const std::string &functionName,
                        const io::lock_t &lock) {
      OCCA_ERROR("dl handle is NULL",
                 dlHandle);

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      void *sym = ::dlsym(dlHandle, functionName.c_str());

      if (!sym) {
        lock.release();
        char *error = dlerror();
        if (error) {
          OCCA_FORCE_ERROR("Error loading symbol [" << functionName << "] from binary with dlsym: " << error << "");
        } else {
          OCCA_FORCE_ERROR("Error loading symbol [" << functionName << "] from binary with dlsym");
        }
      }
#else
      void *sym = GetProcAddress((HMODULE) dlHandle, functionName.c_str());

      if (sym == NULL) {
        lock.release();
        OCCA_FORCE_ERROR("Error loading symbol [" << functionName << "] from binary with GetProcAddress");
      }
#endif

      functionPtr_t sym2;
      ::memcpy(&sym2, &sym, sizeof(sym));
      return sym2;
    }

    void dlclose(void *dlHandle) {
      if (!dlHandle) {
        return;
      }
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      ::dlclose(dlHandle);
#else
      FreeLibrary((HMODULE) (dlHandle));
#endif
    }

    void runFunction(functionPtr_t f, const int argc, void **args) {
#include "runFunction.cpp"
    }

    std::string stacktrace(const int frameStart,
                           const std::string indent) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      static const int maxFrames = 1024;
      static void *frames[maxFrames];

      const int frameCount = ::backtrace(frames, maxFrames);
      char **symbols = ::backtrace_symbols(frames, frameCount);

      const int digits = toString(frameCount - frameStart).size();

      std::stringstream ss;
      for (int i = frameStart; i < frameCount; ++i) {
        const std::string localFrame = toString(frameCount - i);
        ss << indent
           << std::string(digits - localFrame.size(), ' ')
           << localFrame
           << ' '
           << prettyStackSymbol(frames[i], symbols[i]) << '\n';
      }
      ::free(symbols);

      return ss.str();
#endif
    }

    std::string prettyStackSymbol(void *frame, const char *symbol) {
#if (OCCA_OS == OCCA_MACOS_OS)
      std::stringstream ss;
      const char *c = symbol;

      // Skip stack depth
      lex::skipWhitespace(c);
      lex::skipToWhitespace(c);
      lex::skipWhitespace(c);

      // Get origin
      const char *originStart = c;
      lex::skipToWhitespace(c);
      std::string origin(originStart, (c - originStart));

      // Skip address
      lex::skipWhitespace(c);
      lex::skipToWhitespace(c);
      lex::skipWhitespace(c);

      // Get function name
      const char *functionStart = c;
      lex::skipToWhitespace(c);
      std::string function(functionStart, (c - functionStart));

      // Skip the +
      lex::skipWhitespace(c);
      lex::skipToWhitespace(c);
      lex::skipWhitespace(c);

      // Get address offset
      const char *offsetStart = c;
      lex::skipToWhitespace(c);
      std::string offset(offsetStart, (c - offsetStart));

      int status;
      const char *prettyFunction = abi::__cxa_demangle(function.c_str(),
                                                       NULL,
                                                       NULL,
                                                       &status);

      ss << std::left << std::setw(20) << origin
         << std::left << std::setw(50) << (status ? function : prettyFunction);

      ::free((void*) prettyFunction);

      return ss.str();
#elif (OCCA_OS == OCCA_LINUX_OS)
      std::stringstream ss;
      std::string function;

      Dl_info frameInfo;
      int status = dladdr(frame, &frameInfo);
      const char *dl_name = frameInfo.dli_sname;

      if (status && dl_name) {
        const char *prettyFunction = abi::__cxa_demangle(dl_name,
                                                         NULL,
                                                         NULL,
                                                         &status);

        if (!status) {
          function = std::string(prettyFunction);
        }
        ::free((void*) prettyFunction);
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
      return std::string(symbol);
#endif
    }
  }

  void _message(const std::string &header,
                const bool exitInFailure,
                const std::string &filename,
                const std::string &function,
                const int line,
                const std::string &message) {

    exception exp(header,
                  filename,
                  function,
                  line,
                  message);

    if (exitInFailure) {
      throw exp;
    }
    io::stderr << exp;
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

  void printWarning(io::output &out,
                    const std::string &message,
                    const std::string &code) {
    if (env::OCCA_VERBOSE) {
      if (code.size()) {
        out << yellow("Warning " + code);
      } else {
        out << yellow("Warning");
      }
      out << ": " << message << '\n';
    }
  }

  void printError(io::output &out,
                  const std::string &message,
                  const std::string &code) {
    if (code.size()) {
      out << red("Error " + code);
    } else {
      out << red("Error");
    }
    out << ": " << message << '\n';
  }

  mutex::mutex() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    int error = pthread_mutex_init(&mutexHandle, NULL);
#if OCCA_UNSAFE
    ignoreResult(error);
#endif

    OCCA_ERROR("Error initializing mutex",
               error == 0);
#else
    mutexHandle = CreateMutex(NULL, FALSE, NULL);
#endif
  }

  void mutex::free() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    int error = pthread_mutex_destroy(&mutexHandle);
#if OCCA_UNSAFE
    ignoreResult(error);
#endif

    OCCA_ERROR("Error freeing mutex",
               error == 0);
#else
    CloseHandle(mutexHandle);
#endif
  }

  void mutex::lock() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    pthread_mutex_lock(&mutexHandle);
#else
    WaitForSingleObject(mutexHandle, INFINITE);
#endif
  }

  void mutex::unlock() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    pthread_mutex_unlock(&mutexHandle);
#else
    ReleaseMutex(mutexHandle);
#endif
  }
}
