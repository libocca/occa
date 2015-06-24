#include "occaKernelDefines.hpp"
#include "occaTools.hpp"
#include "occaBase.hpp"

#include "occaParser.hpp"

namespace occa {
  //---[ Helper Info ]----------------
  namespace env {
    bool isInitialized = false;

    std::string HOME, PWD;
    std::string PATH, LD_LIBRARY_PATH;

    std::string OCCA_DIR, OCCA_CACHE_DIR;
    stringVector_t OCCA_INCLUDE_PATH;

    void initialize(){
      if(isInitialized)
        return;

      // Standard environment variables
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      HOME            = sys::echo("HOME");
      PWD             = sys::echo("PWD");
      PATH            = sys::echo("PATH");
      LD_LIBRARY_PATH = sys::echo("LD_LIBRARY_PATH");

      endDirWithSlash(HOME);
      endDirWithSlash(PWD);
      endDirWithSlash(PATH);
#endif

      // OCCA environment variables
      OCCA_DIR = sys::echo("OCCA_DIR");
      initCachePath();
      initIncludePath();

      OCCA_CHECK(0 < OCCA_DIR.size(),
                 "Environment variable [OCCA_DIR] is not set");

      endDirWithSlash(OCCA_DIR);
      endDirWithSlash(OCCA_CACHE_DIR);

      isInitialized = true;
    }

    void initCachePath(){
      env::OCCA_CACHE_DIR = sys::echo("OCCA_CACHE_DIR");

      if(env::OCCA_CACHE_DIR.size() == 0){
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
        env::OCCA_CACHE_DIR = ss.str();
      }

      const int chars = env::OCCA_CACHE_DIR.size();

      OCCA_CHECK(0 < chars,
                 "Path to the OCCA caching directory is not set properly, "
                 "unset OCCA_CACHE_DIR to use default directory [~/._occa]");

      env::OCCA_CACHE_DIR = sys::getFilename(env::OCCA_CACHE_DIR);

      if(!sys::dirExists(env::OCCA_CACHE_DIR))
        sys::mkpath(env::OCCA_CACHE_DIR);
    }

    void initIncludePath(){
      std::string oip = sys::echo("OCCA_INCLUDE_PATH");

      const char *cStart = oip.c_str();
      const char *cEnd;

      stringVector_t tmpOIP;

      while(cStart[0] != '\0'){
        cEnd = cStart;
        skipTo(cEnd, ':');

        if(0 < (cEnd - cStart)){
          std::string newPath(cStart, cEnd - cStart);
          endDirWithSlash(newPath);

          tmpOIP.push_back(newPath);
        }

        cStart = (cEnd + (cEnd[0] != '\0'));
      }

      tmpOIP.swap(env::OCCA_INCLUDE_PATH);
    }

    envInitializer_t envInitializer;
  };

  namespace sys {
    std::string echo(const std::string &var){
      char *c_var = getenv(var.c_str());

      if(c_var != NULL)
        return std::string(c_var);

      return "";
    }

    void rmdir(const std::string &dir){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      ::rmdir(dir.c_str());
#else
      ::_rmdir(dir.c_str());
#endif
    }

    int mkdir(const std::string &dir){
      errno = 0;

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      return ::mkdir(dir.c_str(), 0755);
#else
      return ::_mkdir(dir.c_str());
#endif
    }

    void mkpath(const std::string &dir){
      stringVector_t path;
      sys::absolutePathVec(dir, path);

      const int dirCount = (int) path.size();
      int makeFrom = -1;

      if(dirCount == 0)
        return;

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

      return ((stat(dir.c_str(), &statInfo) == 0) &&
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
      std::string ret;

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
        ret += path[dir];
      }

      return ret;
    }

    void absolutePathVec(const std::string &dir_,
                         stringVector_t &pathVec){

      std::string dir = dir_;
      strip(dir);

      const int chars = (int) dir.size();
      const char *c   = dir.c_str();

      bool foundIt = false;

      if(chars == 0)
        return;

      // Starts at home
      if((c[0] == '~') &&
         ((c[1] == '/') || (c[1] == '\0'))){

        absolutePathVec(env::HOME, pathVec);

        if(c[1] == '\0')
          return;

        foundIt = true;
        c += 2;
      }
      // OCCA path
      else if(c[0] == '['){
        const char *c0 = (c + 1);
        skipTo(c, ']');

        if(c[0] == ']'){
          absolutePathVec(env::OCCA_CACHE_DIR, pathVec);

          pathVec.push_back("libraries");
          pathVec.push_back(std::string(c0, c - c0));

          foundIt = true;
          ++c;
        }
      }

      // Relative path
      if((!foundIt) &&
         (c[0] != '/')){

        stringVector_t::iterator it = env::OCCA_INCLUDE_PATH.begin();

        while(it != env::OCCA_INCLUDE_PATH.end()){
          if(sys::fileExists(*it + dir)){
            absolutePathVec(*it, pathVec);

            foundIt = true;
          }

          ++it;
        }

        if(!foundIt)
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

        if(c[0] != '\0')
          ++c;
      }
    }
  };

  // Kernel Caching
  namespace kc {
    std::string sourceFile = "source.occa";
    std::string binaryFile = "binary";
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
  std::string getOnlyFilename(const std::string &filename){
    std::string dir = getFileDirectory(filename);

    if(dir.size() < filename.size())
      return filename.substr(dir.size());

    return "";
  }

  std::string getPlainFilename(const std::string &filename){
    std::string ext = getFileExtension(filename);
    std::string dir = getFileDirectory(filename);

    int start = (int) dir.size();
    int end = (int) filename.size();

    // For the [/] character
    if(0 < start)
      ++start;

    // For the [.ext] extension
    if(0 < ext.size())
      end -= (ext.size() - 1);

    return filename.substr(start, end - start);
  }

  std::string getFileDirectory(const std::string &filename){
    const int chars = (int) filename.size();
    const char *c   = filename.c_str();

    int lastSlash = 0;

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    for(int i = 0; i < chars; ++i)
      if(c[i] == '/')
        lastSlash = i;
#else
    for(int i = 0; i < chars; ++i)
      if((c[i] == '/') ||
         (c[i] == '\\'))
        lastSlash = i;
#endif

    if(lastSlash || (c[0] == '/'))
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

  std::string compressFilename(const std::string &filename){
    if(filename.find(env::OCCA_CACHE_DIR) != 0)
      return filename;

    const std::string libPath = env::OCCA_CACHE_DIR + "libraries/";
    const std::string kerPath = env::OCCA_CACHE_DIR + "kernels/";

    if(filename.find(libPath) == 0){
      std::string libName = getLibraryName(filename);
      std::string theRest = filename.substr(libPath.size() + libName.size() + 1);

      return ("[" + libName + "]/" + theRest);
    }
    else if(filename.find(kerPath) == 0){
      return filename.substr(kerPath.size());
    }

    return filename;
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
                   const std::string &content){

    sys::mkpath(getFileDirectory(filename));

    FILE *fp = fopen(filename.c_str(), "w");

    OCCA_CHECK(fp != 0,
               "Failed to open [" << compressFilename(filename) << "]");

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

    sys::mkpath(env::OCCA_CACHE_DIR + "locks/");

    int mkdirStatus = sys::mkdir(lockDir);

    if(mkdirStatus && (errno == EEXIST))
      return false;

    return true;
  }

  void waitForHash(const std::string &hash, const int depth){
    struct stat buffer;

    std::string lockDir   = getFileLock(hash, depth);
    const char *c_lockDir = lockDir.c_str();

    while(stat(c_lockDir, &buffer) == 0)
      ; // Do Nothing
  }

  void releaseHash(const std::string &hash, const int depth){
    sys::rmdir( getFileLock(hash, depth) );
  }

  bool fileNeedsParser(const std::string &filename){
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
                                        const kernelInfo &info){

    parser fileParser;
    strToStrMap_t compilerFlags;

    const std::string extension = getFileExtension(filename);

    compilerFlags["mode"]     = deviceMode;
    compilerFlags["language"] = ((extension != "ofl") ? "C" : "Fortran");

    if((extension == "oak") ||
       (extension == "oaf")){

      compilerFlags["magic"] = "enabled";
    }

    std::string parsedContent = fileParser.parseFile(info.header,
                                                     filename,
                                                     compilerFlags);

    if(!sys::fileExists(parsedFile)){
      sys::mkpath(getFileDirectory(parsedFile));

      std::ofstream fs;
      fs.open(parsedFile.c_str());

      fs << parsedContent;

      fs.close();
    }

    kernelInfoIterator kIt = fileParser.kernelInfoMap.find(functionName);

    if(kIt != fileParser.kernelInfoMap.end())
      return (kIt->second)->makeParsedKernelInfo();

    OCCA_CHECK(false,
               "Could not find function ["
               << functionName << "] in file ["
               << compressFilename(filename    ) << "]");

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

  void setupOccaHeaders(const kernelInfo &info){
    std::string primitivesFile = sys::getFilename("[occa]/primitives.hpp");
    std::string headerFile     = info.getModeHeaderFilename();

    if(!sys::fileExists(primitivesFile)){
      sys::mkpath(getFileDirectory(primitivesFile));

      std::ofstream fs2;
      fs2.open(primitivesFile.c_str());

      fs2 << occaVectorDefines;

      fs2.close();
    }

    if(!sys::fileExists(headerFile)){
      sys::mkpath(getFileDirectory(headerFile));

      std::ofstream fs2;
      fs2.open(headerFile.c_str());

      if(info.mode & Serial)   fs2 << occaSerialDefines;
      if(info.mode & OpenMP)   fs2 << occaOpenMPDefines;
      if(info.mode & OpenCL)   fs2 << occaOpenCLDefines;
      if(info.mode & CUDA)     fs2 << occaCUDADefines;
      // if(info.mode & HSA)      fs2 << occaHSADefines;
      if(info.mode & Pthreads) fs2 << occaPthreadsDefines;
      if(info.mode & COI)      fs2 << occaCOIDefines;

      fs2.close();
    }
  }

  void createSourceFileFrom(const std::string &filename,
                            const std::string &hashDir,
                            const kernelInfo &info){

    const std::string sourceFile = hashDir + kc::sourceFile;

    if(sys::fileExists(sourceFile))
      return;

    sys::mkpath(hashDir);

    setupOccaHeaders(info);

    std::ofstream fs;
    fs.open(sourceFile.c_str());

    fs << "#include \"" << info.getModeHeaderFilename() << "\"\n"
       << "#include \"" << sys::getFilename("[occa]/primitives.hpp") << "\"\n"
       << info.header
       << readFile(filename);

    fs.close();
  }
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

  std::string getLibraryName(const std::string &filename){
    const std::string cacheLibraryPath = (env::OCCA_CACHE_DIR + "libraries/");

    if(filename.find(cacheLibraryPath) != 0)
      return "";

    const int chars = (int) filename.size();
    const char *c   = filename.c_str();

    int start = (int) cacheLibraryPath.size();
    int end;

    for(end = start; end < chars; ++end){
      if(c[end] == '/')
        break;
    }

    return filename.substr(start, end - start);
  }

  std::string hashFrom(const std::string &filename){
    std::string hashDir = hashDirFor(filename, "");

    const int chars = (int) filename.size();
    const char *c   = filename.c_str();

    int start = (int) hashDir.size();
    int end;

    for(end = (start + 1); end < chars; ++end){
      if(c[end] == '/')
        break;
    }

    return filename.substr(start, end - start);
  }

  std::string hashDirFor(const std::string &filename,
                         const std::string &hash){

    if(filename.size() == 0){
      if(hash.size() != 0)
        return (env::OCCA_CACHE_DIR + "kernels/" + hash + "/");
      else
        return (env::OCCA_CACHE_DIR + "kernels/");
    }

    std::string occaLibName = getLibraryName(sys::getFilename(filename));

    if(occaLibName.size() == 0){
      if(hash.size() != 0)
        return (env::OCCA_CACHE_DIR + "kernels/" + hash + "/");
      else
        return (env::OCCA_CACHE_DIR + "kernels/");
    }

    return (env::OCCA_CACHE_DIR + "libraries/" + occaLibName + "/" + hash + "/");
  }
  //==============================================
};
