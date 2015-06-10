#include "occaTools.hpp"
#include "occaParser.hpp"
#include "occa.hpp"      // For kernelInfo

namespace occa {
  //---[ Helper Info ]----------------
  namespace env {
    bool isInitialized = false;

    std::string HOME;
    std::string PATH, LD_LIBRARY_PATH;

    void initialize(){
      if(isInitialized)
        return;

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      HOME            = echo("HOME");
      PATH            = echo("PATH");
      LD_LIBRARY_PATH = echo("LD_LIBRARY_PATH");
#endif

      isInitialized = true;
    }

    std::string echo(const std::string &var){
      char *c_var = getenv(var.c_str());

      if(c_var != NULL)
        return std::string(c_var);

      return "";
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

  std::string getFilePrefix(const std::string &filename){
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

  void getFilePrefixAndName(const std::string &fullFilename,
                            std::string &prefix,
                            std::string &filename){
    int lastSlash = 0;
    const int chars = fullFilename.size();

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    for(int i = 0; i < chars; ++i)
      if(fullFilename[i] == '/')
        lastSlash = i;
#else
    for(int i = 0; i < chars; ++i)
      if((fullFilename[i] == '/') ||
         (fullFilename[i] == '\\'))
        lastSlash = i;
#endif

    if(lastSlash || (fullFilename[0] == '/'))
      ++lastSlash;

    prefix   = fullFilename.substr(0, lastSlash);
    filename = fullFilename.substr(lastSlash, chars - lastSlash);
  }

  std::string getMidCachedBinaryName(const std::string &cachedBinary,
                                     const std::string &namePrefix){
    std::string prefix, name;
    getFilePrefixAndName(cachedBinary, prefix, name);

    return (prefix + namePrefix + "_" + name);
  }

  std::string getFileLock(const std::string &filename){
    std::string prefix, name;
    getFilePrefixAndName(filename, prefix, name);

    return (prefix + "._occa_dir_" + name);
  }

  bool haveFile(const std::string &filename){
    std::string lockDir = getFileLock(filename);

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    int mkdirStatus = mkdir(lockDir.c_str(), 0755);
#else
    int mkdirStatus = _mkdir(lockDir.c_str());
#endif

    // Someone else is making it
    if(mkdirStatus && (errno == EEXIST))
      return false;

    return true;
  }

  void waitForFile(const std::string &filename){
    struct stat buffer;

    std::string lockDir   = getFileLock(filename);
    const char *c_lockDir = lockDir.c_str();

    while(stat(c_lockDir, &buffer) == 0)
      ; // Do Nothing
  }

  void releaseFile(const std::string &filename){
    std::string lockDir = getFileLock(filename);

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    rmdir(lockDir.c_str());
#else
    _rmdir(lockDir.c_str());
#endif
  }

  parsedKernelInfo parseFileForFunction(const std::string &filename,
                                        const std::string &cachedBinary,
                                        const std::string &functionName,
                                        const kernelInfo &info){

    const std::string iCachedBinary = getMidCachedBinaryName(cachedBinary, "i");
    const std::string pCachedBinary = getMidCachedBinaryName(cachedBinary, "p");

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

  bool fileExists(const std::string &filename){
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
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

  std::string getOCCADir(){
    static std::string occaDir = "";

    if(occaDir.size())
      return occaDir;

    char *c_occaPath = getenv("OCCA_DIR");

    if(c_occaPath != NULL){
      occaDir = c_occaPath;
      return occaDir;
    }

    OCCA_CHECK(false,
               "Environment variable [OCCA_DIR] is not set");

    return occaDir;
  }

  std::string getCachePath(){
    static std::string occaCachePath = "";

    if(occaCachePath.size())
      return occaCachePath;

    char *c_cachePath = getenv("OCCA_CACHE_DIR");

    if(c_cachePath != NULL)
      occaCachePath = c_cachePath;
    else{
      std::stringstream ss;
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      char *c_home = getenv("HOME");
      ss << c_home << "/._occa";

      occaCachePath = ss.str();
#else
      char *c_home = getenv("USERPROFILE");

      ss << c_home << "\\AppData\\Local\\OCCA";

#if OCCA_64_BIT
      ss << "_amd64";  // use different dir's fro 32 and 64 bit
#else
      ss << "_x86";    // use different dir's fro 32 and 64 bit
#endif

      occaCachePath = ss.str();
#endif
    }

    const int chars = occaCachePath.size();

    OCCA_CHECK(0 < chars,
               "OCCA Cache Path is not set");

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    const char slashChar = '/';
#else
    const char slashChar = '\\';
#endif

    // Take out the pesky //'s
    int pos = 0;

    for(int i = 0; i < chars; ++i){
      if(occaCachePath[i] == slashChar)
        while(i < (chars - 1) && occaCachePath[i + 1] == slashChar)
          ++i;

      occaCachePath[pos++] = occaCachePath[i];
    }

    if(occaCachePath[pos - 1] != slashChar){
      if(pos != chars)
        occaCachePath[pos] = slashChar;
      else
        occaCachePath += slashChar;
    }

    if(!fileExists(occaCachePath)){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      mkdir(occaCachePath.c_str(), 0755);
#else
      LPCSTR w_occaCachePath = occaCachePath.c_str();
      BOOL mkdirStatus = CreateDirectoryA(w_occaCachePath, NULL);

      if(mkdirStatus == FALSE)
        assert(GetLastError() == ERROR_ALREADY_EXISTS);
#endif
    }

    return occaCachePath;
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

  bool fileNeedsParser(const std::string &filename){
    std::string ext = getFileExtension(filename);

    return ((ext == "okl") ||
            (ext == "ofl") ||
            (ext == "cl") ||
            (ext == "cu"));
  }

  std::string getCacheHash(const std::string &content,
                           const std::string &salt){

    std::string fo = fnv(content + salt);

    // Only taking the first 16 characters
    return fo.substr(0, 16);
  }

  std::string getCachedName(const std::string &filename,
                            const std::string &salt){

    return getCachePath() + getCacheHash(readFile(filename), salt);
  }

  std::string getContentCachedName(const std::string &content,
                                   const std::string &salt){

    return getCachePath() + getCacheHash(content, salt);
  }

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
};
