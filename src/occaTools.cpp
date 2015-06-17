#include "occaTools.hpp"
#include "occaParser.hpp"
#include "occa.hpp"      // For kernelInfo

namespace occa {
  //---[ Helper Info ]----------------
  namespace env {
    bool isInitialized = false;

    std::string HOME, PWD;
    std::string PATH, LD_LIBRARY_PATH;

    std::string OCCA_DIR, OCCA_CACHE_DIR;

    void initialize(){
      if(isInitialized)
        return;

      // Standard environment variables
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      HOME            = sys::echo("HOME");
      PWD             = sys::echo("PWD");
      PATH            = sys::echo("PATH");
      LD_LIBRARY_PATH = sys::echo("LD_LIBRARY_PATH");
#endif

      // OCCA environment variables
      OCCA_DIR       = sys::echo("OCCA_DIR");
      OCCA_CACHE_DIR = getAndInitCachePath();

      OCCA_CHECK(0 < OCCA_DIR.size(),
                 "Environment variable [OCCA_DIR] is not set");

      endDirWithSlash(HOME);
      endDirWithSlash(PWD);
      endDirWithSlash(PATH);

      endDirWithSlash(OCCA_DIR);
      endDirWithSlash(OCCA_CACHE_DIR);

      isInitialized = true;
    }

    std::string getAndInitCachePath(){
      std::string occaCachePath = sys::echo("OCCA_CACHE_DIR");

      if(occaCachePath.size() == 0){
        std::stringstream ss;

#if (OCCA_OS & (LINUX_OS | OSX_OS))
        ss << sys::echo("HOME") << "/._occa";
#else
        ss << sys::echo("USERPROFILE") << "\\AppData\\Local\\OCCA";

#  if OCCA_64_BIT
        ss << "_amd64";  // use different dir's fro 32 and 64 bit
#  else
        ss << "_x86";    // use different dir's fro 32 and 64 bit
#  endif
#endif
        occaCachePath = ss.str();
      }

      const int chars = occaCachePath.size();

      OCCA_CHECK(0 < chars,
                 "Path to the OCCA caching directory is not set properly, "
                 "unset OCCA_CACHE_DIR to use default directory [~/._occa]");

      occaCachePath = sys::absolutePath(occaCachePath);

      if(!dirExists(occaCachePath))
        sys::mkpath(occaCachePath);

      return occaCachePath;
    }
  };

  namespace sys {
    std::string echo(const std::string &var){
      char *c_var = getenv(var.c_str());

      if(c_var != NULL)
        return std::string(c_var);

      return "";
    }

    void rmdir(const std::strin &dir){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      rmdir(dir.c_str());
#else
      _rmdir(dir.c_str());
#endif
    }

    int mkdir(const std::string &dir){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      return mkdir(dir.c_str(), 0755);
#else
      return _mkdir(dir.c_str());
#endif
    }

    int mkpath(const std::string &dir){
      stringVector_t path;
      absolutePathVec(filename, path);

      const int dirCount = (int) path.size();
      int makeFrom = -1;

      if(dirCount == 0)
        return "";

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      const char slash = '/';
#else
      const char slash = '\\';
#endif

      std::string sPath;

      for(int dir = 0; dir < dirCount; ++dir){
        sPath += slash;
        sPath += path[dir];

        if(!dirExists(sPath)){
          makeFrom = dir;
          break;
        }
      }

      if(0 < makeFrom){
        sys::mkdir(sPath);

        for(int dir = (makeFrom + 1); dir < dirCount; ++dir){
          sPath += slash;
          sPath += path[dir];

          sys::mkdir(sPath);
        }
      }
    }

    bool dirExists(const std::string &dir){
      struct stat statInfo;

      return ((stat(filename.c_str(), &buffer) == 0) &&
              (statInfo.st_mode &S_IFDIR));
    }

    bool fileExists(const std::string &filename,
                    const int flags){

      if(flags & flags::checkCacheDir)
        return fileExists(getFilename(filename));

      struct stat statInfo;

      return (stat(filename.c_str(), &statInfo) == 0);
    }

    std::string getFilename(const std::string &filename){
      stringVector_t path;
      absolutePathVec(filename, path);

      const int dirCount = (int) path.size();

      if(dirCount == 0)
        return "";

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      const char slash = '/';
#else
      const char slash = '\\';
#endif

      for(int dir = 0; dir < dirCount; ++dir){
        ret += slash;
        ret += dir;
      }
    }

    void absolutePathVec(const std::string &dir_,
                         stringVector_t &pathVec){

      const std::string dir = strip(dir_);

      const int chars = (int) dir.size();
      const char *c   = dir.c_str();

      if(chars == 0)
        return;

      // Relative path
      if((c[0] == '~') &&
         ((c[1] == '/') || (c[1] == '\0'))){

        absolutePathVec(env::HOME, pathVec);

        if(c[1] == '\0')
          return pathVec;

        c += 2;
      }
      // OCCA path
      else if(c[0] == '['){
        const char *c0 = (c + 1);
        skipTo(c, ']');

        if(c[0] == ']'){
          absolutePathVec(env::OCCA_DIR, pathVec);

          pathVec.push_back("libraries");
          pathVec.push_back(std::string(c0, c - c0));

          ++c;
        }
        else {
          absolutePathVec(env::PWD, pathVec);
        }
      }
      else {
        absolutePathVec(env::PWD, pathVec);
      }

      while(c[0] != '\0'){
        if(c[0] == '/'){
          ++c;
          continue;
        }

        const char *c0 = c;

        skipTo(c, '/');

        pathVec.push_back(std::string(c0, c - c0));

        if(c[0] != '/0')
          ++c;
      }
    }
  };
  //==================================

  mutex_t::mutex_t(){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    int error = pthread_mutex_init(&mutexHandle, NULL);

    OCCA_CHECK(error == 0,
               "Error initializing mutex");
#else
    mutexHandle = CreateMutex(NULL, FALSE, NULL);
#endif
  }

  void mutex_t::free(){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    int error = pthread_mutex_destroy(&mutexHandle);

    OCCA_CHECK(error == 0,
               "Error freeing mutex");
#else
    CloseHandle(mutexHandle);
#endif
  }

  void mutex_t::lock(){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    pthread_mutex_lock(&mutexHandle);
#else
    WaitForSingleObject(mutexHandle, INFINITE);
#endif
  }

  void mutex_t::unlock(){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    pthread_mutex_unlock(&mutexHandle);
#else
    ReleaseMutex(mutexHandle);
#endif
  }

  fnvOutput_t::fnvOutput_t(){
    h[0] = 101527; h[1] = 101531;
    h[2] = 101533; h[3] = 101537;
    h[4] = 101561; h[5] = 101573;
    h[6] = 101581; h[7] = 101599;
  }

  bool fnvOutput_t::operator == (const fnvOutput_t &fo){
    for(int i = 0; i < 8; ++i){
      if(h[i] != fo.h[i])
        return false;
    }

    return true;
  }

  bool fnvOutput_t::operator != (const fnvOutput_t &fo){
    for(int i = 0; i < 8; ++i){
      if(h[i] != fo.h[i])
        return true;
    }

    return false;
  }

  void fnvOutput_t::mergeWith(const fnvOutput_t &fo){
    for(int i = 0; i < 8; ++i)
      h[i] ^= fo.h[i];
  }

  fnvOutput_t::operator std::string () {
    std::stringstream ss;

    for(int i = 0; i < 8; ++i)
      ss << std::hex << h[i];

    return ss.str();
  }

  double currentTime(){
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

    if(!haveFreq){
      QueryPerformanceFrequency(&freq);
      haveFreq=true;
    }

    LARGE_INTEGER ct;

    QueryPerformanceCounter(&ct);

    return ((double) (ct.QuadPart)) / ((double) (freq.QuadPart));
#endif
  }

  //---[ File Functions ]-------------------------
  std::string getFileDirectory(const std::string &filename){
    int lastSlash = 0;
    const int chars = filename.size();

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    for(int i = 0; i < chars; ++i)
      if(filename[i] == '/')
        lastSlash = i;
#else
    for(int i = 0; i < chars; ++i)
      if((filename[i] == '/') ||
         (filename[i] == '\\'))
        lastSlash = i;
#endif

    if(lastSlash || (filename[0] == '/'))
      ++lastSlash;

    return filename.substr(0, lastSlash);
  }

  std::string getFileExtension(const std::string &filename){
    const char *c = filename.c_str();
    const char *i = NULL;

    while(*c != '\0'){
      if(*c == '.')
        i = c;

      ++c;
    }

    if(i != NULL)
      return filename.substr(i - filename.c_str() + 1);

    return "";
  }

  // NBN: handle binary mode and EOL chars on Windows
  std::string readFile(const std::string &filename, const bool readingBinary){
    FILE *fp = NULL;

    if(!readingBinary){
      fp = fopen(filename.c_str(), "r");
    }
    else{
      fp = fopen(filename.c_str(), "rb");
    }

    OCCA_CHECK(fp != 0,
               "Failed to open [" << filename << "]");

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
                   const std::string &content){

    FILE *fp = fopen(filename.c_str(), "w");

    OCCA_CHECK(fp != 0,
               "Failed to open [" << filename << "]");

    fputs(content.c_str(), fp);

    fclose(fp);
  }

  std::string getFileLock(const std::string &hash, const int depth){
    std::string ret = (env::OCCA_CACHE_DIR + "locks/" + hash);

    ret += '_';
    ret += (char) ('0' + depth);

    return ret;
  }

  bool haveHash(const std::string &hash, const int depth){
    std::string lockDir = getFileLock(hash, depth);

    int mkdirStatus = sys::mkdir(lockDir);

    // Someone else is making it
    if(mkdirStatus && (errno == EEXIST))
      return false;

    return true;
  }

  void waitForHash(const std::string &hash, const int depth){
    struct stat buffer;

    std::string lockDir   = getFileLock(filename, depth);
    const char *c_lockDir = lockDir.c_str();

    while(stat(c_lockDir, &buffer) == 0)
      ; // Do Nothing
  }

  void releaseHash(const std::string &hash, const int depth){
    sys::rmdir( getFileLock(filename, depth) );
  }

  bool fileNeedsParser(const std::string &filename){
    std::string ext = getFileExtension(filename);

    return ((ext == "okl") ||
            (ext == "ofl") ||
            (ext == "cl") ||
            (ext == "cu"));
  }

  parsedKernelInfo parseFileForFunction(const std::string &filename,
                                        const std::string &parsedFile,
                                        const std::string &functionName,
                                        const kernelInfo &info){

    parser fileParser;

    int parsingLanguage;

    if(getFileExtension(filename) != "ofl")
      parsingLanguage = parserInfo::parsingC;
    else
      parsingLanguage = parserInfo::parsingFortran;

    std::ofstream fs;
    fs.open(pCachedBinary.c_str());

    fs << info.header << readFile(filename);

    fs.close();
    fs.clear();

    fs.open(iCachedBinary.c_str());
    fs << info.occaKeywords << fileParser.parseFile(pCachedBinary,
                                                    parsingLanguage);

    fs.close();

    kernelInfoIterator kIt = fileParser.kernelInfoMap.find(functionName);

    if(kIt != fileParser.kernelInfoMap.end()){
      return (kIt->second)->makeParsedKernelInfo();
    }

    OCCA_CHECK(false,
               "Could not find function ["
               << functionName << "] in file ["
               << filename     << "]");

    return parsedKernelInfo();
  }

  std::string removeSlashes(const std::string &str){
    std::string ret = str;
    const size_t chars = str.size();

    for(size_t i = 0; i < chars; ++i){
      if(ret[i] == '/')
        ret[i] = '_';
    }

    return ret;
  }

#if 0
  std::string createIntermediateSource(const std::string &filename,
                                       const std::string &cachedBinary,
                                       const kernelInfo &info,
                                       const bool useParser){
    std::string prefix, name;
    getFilePrefixAndName(cachedBinary, prefix, name);

    const std::string iCachedBinary = prefix + "i_" + name;

    if(fileExists(iCachedBinary))
      return iCachedBinary;

    if(useParser && fileNeedsParser(filename)){
      const std::string pCachedBinary = prefix + "p_" + name;
      parser fileParser;

      std::ofstream fs;
      fs.open(pCachedBinary.c_str());

      fs << info.header << readFile(filename);

      fs.close();

      fs.open(iCachedBinary.c_str());
      fs << info.occaKeywords
         << occaVectorDefines
         << fileParser.parseFile(pCachedBinary);

      fs.close();
    }
    else{
      std::ofstream fs;
      fs.open(iCachedBinary.c_str());

      fs << info.occaKeywords
         << occaVectorDefines
         << info.header
         << readFile(filename);

      fs.close();
    }

    return iCachedBinary;
  }
  #endif
  //==============================================


  //---[ Hash Functions ]-------------------------
  fnvOutput_t fnv(const void *ptr, uintptr_t bytes){
    std::stringstream ss;

    const char *c = (char*) ptr;

    fnvOutput_t fo;
    int *h = fo.h;

    const int p[8] = {102679, 102701,
                      102761, 102763,
                      102769, 102793,
                      102797, 102811};

    for(uintptr_t i = 0; i < bytes; ++i){
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
  fnvOutput_t fnv(const std::string &saltedString){
    return fnv(saltedString.c_str(), saltedString.size());
  }

  std::string getContentHash(const std::string &content,
                             const std::string &salt){

    std::string fo = fnv(content + salt);

    // Only taking the first 16 characters
    return fo.substr(0, 16);
  }

  std::string getFileContentHash(const std::string &filename,
                                 const std::string &salt){

    return getContentHash(readFile(filename), salt);
  }

  std::string getOccaLibraryName(const std::string &filename){
    if(filename.find(env::OCCA_CACHE_DIR) != 0)
      return "";

    const int chars = (int) filename.size();
    const char *c   = filename.c_str();

    int start = (int) env::OCCA_CACHE_DIR.size();
    int end;

    for(end = start; end < chars; ++end){
      if(c[end] == '/')
        break;
    }

    return filename.substr(start, end - start);
  }

  std::string hashDirFor(const std::string &filename,
                         const std::string &hash){

    if(filename.size() == 0)
      return (env::OCCA_CACHE_DIR + "homeless/" + hash);

    std::string occaLibName = getOccaLibraryName(sys::getFilename(filename));

    if(occaLibName.size() == 0)
      return (env::OCCA_CACHE_DIR + "homeless/" + hash);

    return (env::OCCA_CACHE_DIR + "libraries/" + occaLibName + "/" + hash);
  }
  //==============================================
};
