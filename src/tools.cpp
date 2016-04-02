#include <fstream>
#include <cstddef>

#include "occa/tools.hpp"
#include "occa/base.hpp"

#include "occa/parser/parser.hpp"

namespace occa {
  strToBoolMap_t fileLocks;

  //---[ Helper Info ]----------------
  namespace env {
    bool isInitialized = false;

    std::string HOME, PWD;
    std::string PATH, LD_LIBRARY_PATH;

    std::string OCCA_DIR, OCCA_CACHE_DIR;
    size_t OCCA_MEM_BYTE_ALIGN;
    stringVector_t OCCA_INCLUDE_PATH;

    void initialize() {
      if (isInitialized)
        return;

      ::signal(SIGTERM, env::signalExit);
      ::signal(SIGKILL, env::signalExit);
      ::signal(SIGINT , env::signalExit);
      ::signal(SIGQUIT, env::signalExit);

      // Standard environment variables
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      HOME            = env::var("HOME");
      PWD             = env::var("PWD");
      PATH            = env::var("PATH");
      LD_LIBRARY_PATH = env::var("LD_LIBRARY_PATH");

      endDirWithSlash(HOME);
      endDirWithSlash(PWD);
      endDirWithSlash(PATH);
#endif

      // OCCA environment variables
      OCCA_DIR = env::var("OCCA_DIR");
#ifdef OCCA_COMPILED_DIR
      if (OCCA_DIR.size() == 0) {
        OCCA_DIR = OCCA_STRINGIFY(OCCA_COMPILED_DIR);
      }
#endif

      OCCA_CHECK(0 < OCCA_DIR.size(),
                 "Environment variable [OCCA_DIR] is not set");

      initCachePath();
      initIncludePath();

      endDirWithSlash(OCCA_DIR);
      endDirWithSlash(OCCA_CACHE_DIR);

      OCCA_MEM_BYTE_ALIGN = OCCA_DEFAULT_MEM_BYTE_ALIGN;
      if(env::var("OCCA_MEM_BYTE_ALIGN").size() > 0){
        const size_t align = (size_t) std::atoi(env::var("OCCA_MEM_BYTE_ALIGN").c_str());

        if((align != 0) && ((align & (~align + 1)) == align)) {
          OCCA_MEM_BYTE_ALIGN = align;
        }
        else {
          std::cout << "Environment variable [OCCA_MEM_BYTE_ALIGN ("
                    << align << ")] is not a power of two, defaulting to "
                    << OCCA_DEFAULT_MEM_BYTE_ALIGN << '\n';
        }
      }

      isInitialized = true;
    }

    void initCachePath() {
      env::OCCA_CACHE_DIR = env::var("OCCA_CACHE_DIR");

      if (env::OCCA_CACHE_DIR.size() == 0) {
        std::stringstream ss;

#if (OCCA_OS & (LINUX_OS | OSX_OS))
        ss << env::var("HOME") << "/._occa";
#else
        ss << env::var("USERPROFILE") << "\\AppData\\Local\\OCCA";

#  if OCCA_64_BIT
        ss << "_amd64";  // use different dir's fro 32 and 64 bit
#  else
        ss << "_x86";    // use different dir's fro 32 and 64 bit
#  endif
#endif
        env::OCCA_CACHE_DIR = ss.str();
      }

      const int chars = env::OCCA_CACHE_DIR.size();

      OCCA_CHECK(0 < chars,
                 "Path to the OCCA caching directory is not set properly, "
                 "unset OCCA_CACHE_DIR to use default directory [~/._occa]");

      env::OCCA_CACHE_DIR = sys::getFilename(env::OCCA_CACHE_DIR);

      if (!sys::dirExists(env::OCCA_CACHE_DIR))
        sys::mkpath(env::OCCA_CACHE_DIR);
    }

    void initIncludePath() {
      env::OCCA_INCLUDE_PATH.clear();
      std::string oip = env::var("OCCA_INCLUDE_PATH");

      const char *cStart = oip.c_str();
      const char *cEnd;

      stringVector_t tmpOIP;

      while(cStart[0] != '\0') {
        cEnd = cStart;
        skipTo(cEnd, ':');

        if (0 < (cEnd - cStart)) {
          std::string newPath(cStart, cEnd - cStart);
          newPath = sys::getFilename(newPath);
          endDirWithSlash(newPath);

          tmpOIP.push_back(newPath);
        }

        cStart = (cEnd + (cEnd[0] != '\0'));
      }

      tmpOIP.swap(env::OCCA_INCLUDE_PATH);
    }

    void signalExit(int sig) {
      if ((sig == SIGTERM) ||
          (sig == SIGKILL) ||
          (sig == SIGINT ) ||
          (sig == SIGQUIT)) {

        clearLocks();
        ::exit(sig);
      }
    }

    std::string var(const std::string &varName) {
      char *c_varName = getenv(varName.c_str());

      if (c_varName != NULL)
        return std::string(c_varName);

      return "";
    }

    envInitializer_t::envInitializer_t() {
      env::initialize();
    }
    envInitializer_t envInitializer;
  }

  namespace sys {
    dirTree_t::dirTree_t() :
      info(dirType::none),
      name(),

      dirCount(0),
      nestedDirCount(0),
      dirs(NULL),

      nestedDirNames(NULL) {}

    dirTree_t::dirTree_t(const dirTree_t &dt) :
      info(dt.info),
      name(dt.name),

      dirCount(dt.dirCount),
      nestedDirCount(dt.nestedDirCount),
      dirs(dt.dirs),

      nestedDirNames(dt.nestedDirNames) {}

    dirTree_t& dirTree_t::operator = (const dirTree_t &dt) {
      info = dt.info;
      name = dt.name;

      dirCount       = dt.dirCount;
      nestedDirCount = dt.nestedDirCount;
      dirs           = dt.dirs;

      nestedDirNames = dt.nestedDirNames;

      return *this;
    }

    dirTree_t::dirTree_t(const std::string &dir) :
      info(dirType::none),
      name(),

      dirCount(0),
      nestedDirCount(0),
      dirs(NULL),

      nestedDirNames(NULL) {

      load(dir);
    }

    void dirTree_t::load(const std::string &dir_) {
      std::string dir = expandEnvVariables(dir_);
      strip(dir);
      dir = getFilename(dir);

      stringVector_t path;
      sys::absolutePathVec(dir, path);

      // Root directory doesn't have a name
      info = dirType::dir;
      name = "";

      load("/", path, 0);

      if (dirCount == 0)
        return;

      nestedDirNames = new std::string[nestedDirCount];

      setNestedDirNames(nestedDirNames, 0);
    }

    bool dirTree_t::load(const std::string &base,
                         stringVector_t &path,
                         const int pathPos) {

      // Return values (for readability)
      const bool somethingFound = true;
      const bool nothingFound   = false;

      // We reached the end
      if (((int) path.size()) <= pathPos) {
        nestedDirCount = 1;
        return somethingFound;
      }

      const std::string nextDir = path[pathPos];
      const char *c_nextDir     = nextDir.c_str();

      // Simple file traversal
      if (!hasWildcard(c_nextDir)) {
        std::string nextBase = base;
        nextBase += path[pathPos];
        nextBase += '/';

        if (!fileExists(nextBase))
          return nothingFound;

        dirCount = 1;
        dirs     = new dirTree_t[1];

        // Temporary info
        dirs[0].info = dirType::dir;
        dirs[0].name = path[pathPos];

        const bool dirsCheck = dirs[0].load(nextBase,
                                            path,
                                            pathPos + 1);

        if (dirsCheck == nothingFound) {
          free();
          return nothingFound;
        }

        nestedDirCount = dirs[0].nestedDirCount;

        return somethingFound;
      }

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      DIR *DIR_ = opendir(base.c_str());

      if (DIR_ == NULL)
        return nothingFound;

      std::vector<dirTree_t> vDirs;

      dirent *dirent_;

      while(true) {
        dirent_ = readdir(DIR_);

        if (dirent_ == NULL)
          break;

        const char *c_dirName = dirent_->d_name;

        if (matches(c_nextDir, c_dirName)) {
          std::string nextBase = base;
          nextBase += c_dirName;
          nextBase += '/';

          vDirs.push_back(dirTree_t());
          dirTree_t &vDir = vDirs.back();

          const bool dirsCheck = vDir.load(nextBase,
                                           path,
                                           pathPos + 1);

          if (dirsCheck == nothingFound) {
            vDir.free();
            vDirs.pop_back();
          }
          else {
            // Temporary info
            vDir.info = dirType::dir;
            vDir.name = c_dirName;
          }
        }
      }

      closedir(DIR_);

      if (vDirs.size() == 0)
        return nothingFound;

      dirCount = (int) vDirs.size();
      dirs     = new dirTree_t[dirCount];

      for(int i = 0; i < dirCount; ++i) {
        dirs[i] = vDirs[i];
        nestedDirCount += dirs[i].nestedDirCount;
      }
#else
      OCCA_CHECK(false,
                 "dirTree_t wildcard traversal is not supported yet in Windows");
#endif

      return somethingFound;
    }

    void dirTree_t::free() {
      info = dirType::none;

      if (0 < dirCount) {
        for(int i = 0; i < dirCount; ++i)
          dirs[i].free();

        dirCount = 0;

        delete [] dirs;
        dirs = NULL;
      }
    }

    void dirTree_t::setNestedDirNames(std::string *fdn,
                                      int fdnPos) {
      // Empty
      if (info == dirType::none)
        return;

      // Last dir
      if (dirCount == 0) {
        fdn[fdnPos] += name;

        if (info & dirType::dir)
          fdn[fdnPos] += '/';

        return;
      }

      for(int i = 0; i < nestedDirCount; ++i) {
        fdn[fdnPos + i] += name;
        fdn[fdnPos + i] += '/';
      }

      for(int i = 0; i < dirCount; ++i) {
        dirs[i].setNestedDirNames(fdn, fdnPos);
        fdnPos += dirs[i].nestedDirCount;
      }
    }

    void dirTree_t::printOnString(std::string &str,
                                  const char delimiter) {
      // Empty
      if (info == dirType::none)
        return;

      for(int i = 0; i < nestedDirCount; ++i) {
        if (0 < i)
          str += delimiter;

        str += nestedDirNames[i];
      }
    }

    bool dirTree_t::hasWildcard(const char *c) {
      const char *c0 = c;

      while(*c != '\0') {
        if (((c[0] == '*') || (c[0] == '?')) &&
           ((c0 == c) || (c[-1] != '\\'))) {

          return true;
        }

        ++c;
      }

      return false;
    }

    bool dirTree_t::matches(const char *search,
                            const char *c) {

      const int sSize = (int) strlen(search);
      const int cSize = (int) strlen(c);

      if ((cSize == 0) || (sSize == 0))
        return false;

      // Hidden files only show up when [.] is explicitly used
      if ((c[0]      == '.') &&
         (search[0] != '.')) {

        return false;
      }

      const int entries = ((sSize + 1) * (cSize + 1));

      // Create lookup table + initialize to false
      bool *found_ = new bool[entries];
      ::memset(found_, 0, entries * sizeof(bool));

      // \0 matches \0
      found_[entries - 1] = true;

      for(int sp = (sSize - 1); 0 <= sp; --sp) {
        bool *found0 = found_ + ((sp + 0) * (cSize + 1));
        bool *found1 = found_ + ((sp + 1) * (cSize + 1));

        // Wildcard with 1 or X characters
        const bool wc1 = ((search[sp] == '?') &&
                          ((sp == 0) || search[sp - 1] != '\\'));
        const bool wcX = ((search[sp] == '*') &&
                          ((sp == 0) || search[sp - 1] != '\\'));

        for(int cp = (cSize - 1); 0 <= cp; --cp) {

          if ((search[sp] == c[cp]) || wc1) {
            found0[cp] = found1[cp + 1];
          }
          else if (wcX) {
            // [  ,+1]: Wildcard matched next char, wildcard still applies
            // [+1,  ]: Alive because of wildcard , wildcard still applies
            // [+1,+1]: Alive because of next char, wildcard starts
            found0[cp] = (found0[cp + 1] ||
                          found1[cp + 0] ||
                          found1[cp + 1]);
          }

        }
      }

      // One match found it's way to the first char
      const bool foundMatch = found_[0];

      delete [] found_;

      return foundMatch;
    }

    int call(const std::string &cmdline) {
      FILE *fp = popen(cmdline.c_str(), "r");
      return pclose(fp);
    }

    int call(const std::string &cmdline, std::string &output) {
      FILE *fp = popen(cmdline.c_str(), "r");

      size_t lineBytes = 512;
      char lineBuffer[512];

      while(fgets(lineBuffer, lineBytes, fp))
        output += lineBuffer;

      return pclose(fp);
    }

    std::string expandEnvVariables(const std::string &str) {
      std::string ret;

      const char *cRoot = str.c_str();
      const char *c     = cRoot;

      while(*c != '\0') {
        const char C = c[0];

        if ((C == '$')     &&
           (c[1] != '\0') &&                   // Last $ doesn't expand
           ((cRoot == c) || (c[-1] != '\\'))) { // Escape the '$'

          ++c; // Skip $

          const bool hasBrace = (*c == '{');
          const char *c0 = (c + hasBrace);

          if (hasBrace)
            skipTo(c, '}');
          else
            skipToWhitespace(c);

          std::string envVar = env::var(std::string(c0, c - c0));

          ret += envVar;

          if (hasBrace)
            ++c;
        }
        else {
          ret += C;
          ++c;
        }
      }

      return ret;
    }

    void rmdir(const std::string &dir) {
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      ::rmdir(dir.c_str());
#else
      ::_rmdir(dir.c_str());
#endif
    }

    int mkdir(const std::string &dir) {
      errno = 0;

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      return ::mkdir(dir.c_str(), 0755);
#else
      return ::_mkdir(dir.c_str());
#endif
    }

    void mkpath(const std::string &dir) {
      stringVector_t path;
      sys::absolutePathVec(dir, path);

      const int dirCount = (int) path.size();
      int makeFrom = -1;

      if (dirCount == 0)
        return;

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      const char slash = '/';
#else
      const char slash = '\\';
#endif

      std::string sPath;

      for(int d = 0; d < dirCount; ++d) {
        sPath += slash;
        sPath += path[d];

        if (!dirExists(sPath)) {
          makeFrom = d;
          break;
        }
      }

      if (0 < makeFrom) {
        sys::mkdir(sPath);

        for(int d = (makeFrom + 1); d < dirCount; ++d) {
          sPath += slash;
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
              (statInfo.st_mode &S_IFDIR));
    }

    bool fileExists(const std::string &filename_,
                    const int flags) {

      std::string filename = expandEnvVariables(filename_);
      strip(filename);

      if (flags & flags::checkCacheDir)
        return fileExists(getFilename(filename));

      struct stat statInfo;

      return (stat(filename.c_str(), &statInfo) == 0);
    }

    std::string getFilename(const std::string &filename) {
      std::string ret;

      stringVector_t path;
      absolutePathVec(filename, path);

      const int dirCount = (int) path.size();

      if (dirCount == 0)
        return "";

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      const char slash = '/';
#else
      const char slash = '\\';
#endif

      for(int dir = 0; dir < dirCount; ++dir) {
        ret += slash;
        ret += path[dir];
      }

      return ret;
    }

    void absolutePathVec(const std::string &path_,
                         stringVector_t &pathVec) {

      std::string path = expandEnvVariables(path_);
      strip(path);

      const int chars = (int) path.size();
      const char *c   = path.c_str();

      bool foundDir = false;

      if (chars == 0)
        return;

      // Starts at home
      if ((c[0] == '~') &&
         ((c[1] == '/') || (c[1] == '\0'))) {

        absolutePathVec(env::HOME, pathVec);

        if (c[1] == '\0')
          return;

        foundDir = true;
        c += 2;
      }
      // OCCA path
      else if (c[0] == '[') {
        const char *c0 = (c + 1);
        skipTo(c, ']');

        if (c[0] == ']') {
          absolutePathVec(env::OCCA_CACHE_DIR, pathVec);

          pathVec.push_back("libraries");
          pathVec.push_back(std::string(c0, c - c0));

          foundDir = true;
          ++c;
        }
      }

      // Relative path
      if ((!foundDir) &&
         (c[0] != '/')) {

        stringVector_t::iterator it = env::OCCA_INCLUDE_PATH.begin();

        while(it != env::OCCA_INCLUDE_PATH.end()) {
          if (it->size() && sys::fileExists(*it + path)) {
            absolutePathVec(*it, pathVec);
            foundDir = true;
            break;
          }
          ++it;
        }

        if (!foundDir)
          absolutePathVec(env::PWD, pathVec);
      }

      while(c[0] != '\0') {
        if (c[0] == '/') {
          ++c;
          continue;
        }

        const char *c0 = c;
        skipTo(c, '/');

        pathVec.push_back(std::string(c0, c - c0));

        if (c[0] != '\0')
          ++c;
      }
    }
  }

  // Kernel Caching
  namespace kc {
    std::string sourceFile = "source.occa";
    std::string binaryFile = "binary";
  }
  //==================================

  mutex_t::mutex_t() {
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    int error = pthread_mutex_init(&mutexHandle, NULL);

    OCCA_CHECK(error == 0,
               "Error initializing mutex");
#else
    mutexHandle = CreateMutex(NULL, FALSE, NULL);
#endif
  }

  void mutex_t::free() {
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    int error = pthread_mutex_destroy(&mutexHandle);

    OCCA_CHECK(error == 0,
               "Error freeing mutex");
#else
    CloseHandle(mutexHandle);
#endif
  }

  void mutex_t::lock() {
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    pthread_mutex_lock(&mutexHandle);
#else
    WaitForSingleObject(mutexHandle, INFINITE);
#endif
  }

  void mutex_t::unlock() {
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    pthread_mutex_unlock(&mutexHandle);
#else
    ReleaseMutex(mutexHandle);
#endif
  }

  fnvOutput_t::fnvOutput_t() {
    h[0] = 101527; h[1] = 101531;
    h[2] = 101533; h[3] = 101537;
    h[4] = 101561; h[5] = 101573;
    h[6] = 101581; h[7] = 101599;
  }

  bool fnvOutput_t::operator == (const fnvOutput_t &fo) {
    for(int i = 0; i < 8; ++i) {
      if (h[i] != fo.h[i])
        return false;
    }

    return true;
  }

  bool fnvOutput_t::operator != (const fnvOutput_t &fo) {
    for(int i = 0; i < 8; ++i) {
      if (h[i] != fo.h[i])
        return true;
    }

    return false;
  }

  void fnvOutput_t::mergeWith(const fnvOutput_t &fo) {
    for(int i = 0; i < 8; ++i)
      h[i] ^= fo.h[i];
  }

  fnvOutput_t::operator std::string () {
    std::stringstream ss;

    for(int i = 0; i < 8; ++i)
      ss << std::hex << h[i];

    return ss.str();
  }

  double currentTime() {
#if (OCCA_OS & LINUX_OS)

    timespec ct;
    clock_gettime(CLOCK_MONOTONIC, &ct);

    return (double) (ct.tv_sec + (1.0e-9 * ct.tv_nsec));

#elif (OCCA_OS == OSX_OS)
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
#elif (OCCA_OS == WINDOWS_OS)
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

  //---[ File Functions ]-------------------------
  std::string getOnlyFilename(const std::string &filename) {
    std::string dir = getFileDirectory(filename);

    if (dir.size() < filename.size())
      return filename.substr(dir.size());

    return "";
  }

  std::string getPlainFilename(const std::string &filename) {
    std::string ext = getFileExtension(filename);
    std::string dir = getFileDirectory(filename);

    int start = (int) dir.size();
    int end = (int) filename.size();

    // For the [/] character
    if (0 < start)
      ++start;

    // For the [.ext] extension
    if (0 < ext.size())
      end -= (ext.size() - 1);

    return filename.substr(start, end - start);
  }

  std::string getFileDirectory(const std::string &filename) {
    const int chars = (int) filename.size();
    const char *c   = filename.c_str();

    int lastSlash = 0;

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    for(int i = 0; i < chars; ++i)
      if (c[i] == '/')
        lastSlash = i;
#else
    for(int i = 0; i < chars; ++i)
      if ((c[i] == '/') ||
         (c[i] == '\\'))
        lastSlash = i;
#endif

    if (lastSlash || (c[0] == '/'))
      ++lastSlash;

    return filename.substr(0, lastSlash);
  }

  std::string getFileExtension(const std::string &filename) {
    const char *c = filename.c_str();
    const char *i = NULL;

    while(*c != '\0') {
      if (*c == '.')
        i = c;

      ++c;
    }

    if (i != NULL)
      return filename.substr(i - filename.c_str() + 1);

    return "";
  }

  std::string compressFilename(const std::string &filename) {
    if (filename.find(env::OCCA_CACHE_DIR) != 0)
      return filename;

    const std::string libPath = env::OCCA_CACHE_DIR + "libraries/";
    const std::string kerPath = env::OCCA_CACHE_DIR + "kernels/";

    if (filename.find(libPath) == 0) {
      std::string libName = getLibraryName(filename);
      std::string theRest = filename.substr(libPath.size() + libName.size() + 1);

      return ("[" + libName + "]/" + theRest);
    }
    else if (filename.find(kerPath) == 0) {
      return filename.substr(kerPath.size());
    }

    return filename;
  }

  // NBN: handle binary mode and EOL chars on Windows
  std::string readFile(const std::string &filename, const bool readingBinary) {
    FILE *fp = NULL;

    if (!readingBinary) {
      fp = fopen(filename.c_str(), "r");
    }
    else{
      fp = fopen(filename.c_str(), "rb");
    }

    OCCA_CHECK(fp != 0,
               "Failed to open [" << compressFilename(filename) << "]");

    struct stat statbuf;
    stat(filename.c_str(), &statbuf);

    const size_t nchars = statbuf.st_size;

    char *buffer = (char*) calloc(nchars + 1, sizeof(char));
    size_t nread = fread(buffer, sizeof(char), nchars, fp);

    fclose(fp);
    buffer[nread] = '\0';

    std::string contents(buffer, nread);

    free(buffer);

    return contents;
  }

  void writeToFile(const std::string &filename,
                   const std::string &content) {

    sys::mkpath(getFileDirectory(filename));

    FILE *fp = fopen(filename.c_str(), "w");

    OCCA_CHECK(fp != 0,
               "Failed to open [" << compressFilename(filename) << "]");

    fputs(content.c_str(), fp);

    fclose(fp);
  }

  std::string getFileLock(const std::string &hash, const int depth) {
    std::string ret = (env::OCCA_CACHE_DIR + "locks/" + hash);

    ret += '_';
    ret += (char) ('0' + depth);

    return ret;
  }

  void clearLocks() {
    strToBoolMapIterator it = fileLocks.begin();
    while (it != fileLocks.end()) {
      releaseHash(it->first);
      ++it;
    }
    fileLocks.clear();
  }

  bool haveHash(const std::string &hash, const int depth) {
    std::string lockDir = getFileLock(hash, depth);

    sys::mkpath(env::OCCA_CACHE_DIR + "locks/");

    int mkdirStatus = sys::mkdir(lockDir);

    if (mkdirStatus && (errno == EEXIST))
      return false;

    fileLocks[lockDir] = true;

    return true;
  }

  void waitForHash(const std::string &hash, const int depth) {
    struct stat buffer;

    std::string lockDir   = getFileLock(hash, depth);
    const char *c_lockDir = lockDir.c_str();

    while(stat(c_lockDir, &buffer) == 0)
      ; // Do Nothing
  }

  void releaseHash(const std::string &hash, const int depth) {
    releaseHashLock(getFileLock(hash, depth));
  }

  void releaseHashLock(const std::string &lockDir) {
    sys::rmdir(lockDir);
    fileLocks.erase(lockDir);
  }

  bool fileNeedsParser(const std::string &filename) {
    std::string ext = getFileExtension(filename);

    return ((ext == "okl") ||
            (ext == "ofl") ||
            (ext == "cl") ||
            (ext == "cu"));
  }

  parsedKernelInfo parseFileForFunction(const std::string &deviceMode,
                                        const std::string &filename,
                                        const std::string &parsedFile,
                                        const std::string &functionName,
                                        const kernelInfo &info) {

    parser fileParser;

    const std::string extension = getFileExtension(filename);

    flags_t parserFlags = info.getParserFlags();

    parserFlags["mode"]     = deviceMode;
    parserFlags["language"] = ((extension != "ofl") ? "C" : "Fortran");

    if ((extension == "oak") ||
       (extension == "oaf")) {

      parserFlags["magic"] = "enabled";
    }

    std::string parsedContent = fileParser.parseFile(info.header,
                                                     filename,
                                                     parserFlags);

    if (!sys::fileExists(parsedFile)) {
      sys::mkpath(getFileDirectory(parsedFile));

      std::ofstream fs;
      fs.open(parsedFile.c_str());

      fs << parsedContent;

      fs.close();
    }

    kernelInfoIterator kIt = fileParser.kernelInfoMap.find(functionName);

    if (kIt != fileParser.kernelInfoMap.end())
      return (kIt->second)->makeParsedKernelInfo();

    OCCA_CHECK(false,
               "Could not find function ["
               << functionName << "] in file ["
               << compressFilename(filename    ) << "]");

    return parsedKernelInfo();
  }

  std::string removeSlashes(const std::string &str) {
    std::string ret = str;
    const size_t chars = str.size();

    for(size_t i = 0; i < chars; ++i) {
      if (ret[i] == '/')
        ret[i] = '_';
    }

    return ret;
  }

  std::string getOccaScriptFile(const std::string &filename) {
    return readFile(env::OCCA_DIR + "/scripts/" + filename);
  }

  void setupOccaHeaders(const kernelInfo &info) {
    cacheFile(sys::getFilename("[occa]/primitives.hpp"),
              readFile(env::OCCA_DIR + "/include/occa/defines/vector.hpp"),
              "vectorDefines");

    std::string mode = modeToStr(info.mode);
    cacheFile(info.getModeHeaderFilename(),
              readFile(env::OCCA_DIR + "/include/occa/defines/" + mode + ".hpp"),
              mode + "Defines");
  }

  void cacheFile(const std::string &filename,
                 std::string source,
                 const std::string &hash) {

    cacheFile(filename, source.c_str(), hash, false);
  }

  void cacheFile(const std::string &filename,
                 const char *source,
                 const std::string &hash,
                 const bool deleteSource) {
    if(!haveHash(hash)){
      waitForHash(hash);
    } else {
      if (!sys::fileExists(filename)) {
        sys::mkpath(getFileDirectory(filename));

        std::ofstream fs2;
        fs2.open(filename.c_str());
        fs2 << source;
        fs2.close();
      }
      releaseHash(hash);
    }
    if (deleteSource)
      delete [] source;
  }

  void createSourceFileFrom(const std::string &filename,
                            const std::string &hashDir,
                            const kernelInfo &info) {

    const std::string sourceFile = hashDir + kc::sourceFile;

    if (sys::fileExists(sourceFile))
      return;

    sys::mkpath(hashDir);

    setupOccaHeaders(info);

    std::ofstream fs;
    fs.open(sourceFile.c_str());

    fs << "#include \"" << info.getModeHeaderFilename() << "\"\n"
       << "#include \"" << sys::getFilename("[occa]/primitives.hpp") << "\"\n";

    if (info.mode & (Serial | OpenMP | Pthreads | CUDA)) {
      fs << "#if defined(OCCA_IN_KERNEL) && !OCCA_IN_KERNEL\n"
         << "using namespace occa;\n"
         << "#endif\n";
    }


    fs << info.header
       << readFile(filename);

    fs.close();
  }
  //==============================================


  //---[ Hash Functions ]-------------------------
  fnvOutput_t fnv(const void *ptr, uintptr_t bytes) {
    std::stringstream ss;

    const char *c = (char*) ptr;

    fnvOutput_t fo;
    int *h = fo.h;

    const int p[8] = {102679, 102701,
                      102761, 102763,
                      102769, 102793,
                      102797, 102811};

    for(uintptr_t i = 0; i < bytes; ++i) {
      for(int j = 0; j < 8; ++j)
        h[j] = (h[j] * p[j]) ^ c[i];
    }

    // int h2[8];

    // for(int i = 0; i < 8; ++i)
    //   h2[i] = ((h[0] & (0xFF << (8*i))) << (8*i + 0))
    //     |     ((h[1] & (0xFF << (8*i))) << (8*i + 1))
    //     |     ((h[2] & (0xFF << (8*i))) << (8*i + 2))
    //     |     ((h[3] & (0xFF << (8*i))) << (8*i + 3))
    //     |     ((h[4] & (0xFF << (8*i))) << (8*i + 4))
    //     |     ((h[5] & (0xFF << (8*i))) << (8*i + 5))
    //     |     ((h[6] & (0xFF << (8*i))) << (8*i + 6))
    //     |     ((h[7] & (0xFF << (8*i))) << (8*i + 7));

    return fo;
  }

  template <>
  fnvOutput_t fnv(const std::string &saltedString) {
    return fnv(saltedString.c_str(), saltedString.size());
  }

  std::string getContentHash(const std::string &content,
                             const std::string &salt) {

    std::string fo = fnv(content + salt);

    // Only taking the first 16 characters
    return fo.substr(0, 16);
  }

  std::string getFileContentHash(const std::string &filename,
                                 const std::string &salt) {

    return getContentHash(readFile(filename), salt);
  }

  std::string getLibraryName(const std::string &filename) {
    const std::string cacheLibraryPath = (env::OCCA_CACHE_DIR + "libraries/");

    if (filename.find(cacheLibraryPath) != 0)
      return "";

    const int chars = (int) filename.size();
    const char *c   = filename.c_str();

    int start = (int) cacheLibraryPath.size();
    int end;

    for(end = start; end < chars; ++end) {
      if (c[end] == '/')
        break;
    }

    return filename.substr(start, end - start);
  }

  std::string hashFrom(const std::string &filename) {
    std::string hashDir = hashDirFor(filename, "");

    const int chars = (int) filename.size();
    const char *c   = filename.c_str();

    int start = (int) hashDir.size();
    int end;

    for(end = (start + 1); end < chars; ++end) {
      if (c[end] == '/')
        break;
    }

    return filename.substr(start, end - start);
  }

  std::string hashDirFor(const std::string &filename,
                         const std::string &hash) {

    if (filename.size() == 0) {
      if (hash.size() != 0)
        return (env::OCCA_CACHE_DIR + "kernels/" + hash + "/");
      else
        return (env::OCCA_CACHE_DIR + "kernels/");
    }

    std::string occaLibName = getLibraryName(sys::getFilename(filename));

    if (occaLibName.size() == 0) {
      if (hash.size() != 0)
        return (env::OCCA_CACHE_DIR + "kernels/" + hash + "/");
      else
        return (env::OCCA_CACHE_DIR + "kernels/");
    }

    return (env::OCCA_CACHE_DIR + "libraries/" + occaLibName + "/" + hash + "/");
  }
  //==============================================


  //---[ String Functions ]-----------------------
  uintptr_t atoi(const char *c) {
    uintptr_t ret = 0;

    const char *c0 = c;

    bool negative  = false;
    bool unsigned_ = false;
    int longs      = 0;

    skipWhitespace(c);

    if ((*c == '+') || (*c == '-')) {
      negative = (*c == '-');
      ++c;
    }

    if (c[0] == '0')
      return atoiBase2(c0);

    while(('0' <= *c) && (*c <= '9')) {
      ret *= 10;
      ret += *(c++) - '0';
    }

    while(*c != '\0') {
      const char C = upChar(*c);

      if (C == 'L')
        ++longs;
      else if (C == 'U')
        unsigned_ = true;
      else
        break;

      ++c;
    }

    if (negative)
      ret = ((~ret) + 1);

    if (longs == 0) {
      if (!unsigned_)
        ret = ((uintptr_t) ((int) ret));
      else
        ret = ((uintptr_t) ((unsigned int) ret));
    }
    else if (longs == 1) {
      if (!unsigned_)
        ret = ((uintptr_t) ((long) ret));
      else
        ret = ((uintptr_t) ((unsigned long) ret));
    }
    // else it's already in uintptr_t form

    return ret;
  }

  uintptr_t atoiBase2(const char*c) {
    uintptr_t ret = 0;

    const char *c0 = c;

    bool negative     = false;
    int bits          = 3;
    int maxDigitValue = 10;
    char maxDigitChar = '9';

    skipWhitespace(c);

    if ((*c == '+') || (*c == '-')) {
      negative = (*c == '-');
      ++c;
    }

    if (*c == '0') {
      ++c;

      const char C = upChar(*c);

      if (C == 'X') {
        bits = 4;
        ++c;

        maxDigitValue = 16;
        maxDigitChar  = 'F';
      }
      else if (C == 'B') {
        bits = 1;
        ++c;

        maxDigitValue = 2;
        maxDigitChar  = '1';
      }
    }

    while(true) {
      if (('0' <= *c) && (*c <= '9')) {
        const char digitValue = *(c++) - '0';

        OCCA_CHECK(digitValue < maxDigitValue,
                   "Number [" << std::string(c0, c - c0)
                   << "...] must contain digits in the [0,"
                   << maxDigitChar << "] range");

        ret <<= bits;
        ret += digitValue;
      }
      else {
        const char C = upChar(*c);

        if (('A' <= C) && (C <= 'F')) {
          const char digitValue = 10 + (C - 'A');
          ++c;

          OCCA_CHECK(digitValue < maxDigitValue,
                     "Number [" << std::string(c0, c - c0)
                     << "...] must contain digits in the [0,"
                     << maxDigitChar << "] range");

          ret <<= bits;
          ret += digitValue;
        }
        else
          break;
      }
    }

    if (negative)
      ret = ((~ret) + 1);

    return ret;
  }

  std::string stringifyBytes(uintptr_t bytes) {
    if (0 < bytes) {
      std::stringstream ss;
      uintptr_t big1 = 1;

      if (bytes < (big1 << 10))
        ss << bytes << " bytes";
      else if (bytes < (big1 << 20))
        ss << (bytes >> 10) << " KB";
      else if (bytes < (big1 << 30))
        ss << (bytes >> 20) << " MB";
      else if (bytes < (big1 << 40))
        ss << (bytes >> 30) << " GB";
      else if (bytes < (big1 << 50))
        ss << (bytes >> 40) << " TB";
      else
        ss << bytes << " bytes";

      return ss.str();
    }

    return "";
  }
  //==============================================
}
