#include "occaTools.hpp"
#include "occa.hpp"      // For kernelInfo

namespace occa {
  mutex_t::mutex_t(){
#if (OCL_OS == OCL_LINUX_OS) || (OCL_OS == OCL_OSX_OS)
    int error = pthread_mutex_init(&mutexHandle, NULL);

    if(error){
      std::cout << "Error initializing mutex\n";
      throw 1;
    }
#else
    mutexHandle = CreateMutex(NULL, FALSE, NULL);
#endif
  }

  void mutex_t::free(){
#if (OCL_OS == OCL_LINUX_OS) || (OCL_OS == OCL_OSX_OS)
    int error = pthread_mutex_destroy(&mutexHandle);

    if(error){
      std::cout << "Error freeing mutex\n";
      throw 1;
    }
#else
    CloseHandle(mutexHandle);
#endif
  }

  void mutex_t::lock(){
#if (OCL_OS == OCL_LINUX_OS) || (OCL_OS == OCL_OSX_OS)
    pthread_mutex_lock(&mutexHandle);
#else
    WaitForSingleObject(mutexHandle, INFINITE);
#endif
  }

  void mutex_t::unlock(){
#if (OCL_OS == OCL_LINUX_OS) || (OCL_OS == OCL_OSX_OS)
    pthread_mutex_unlock(&mutexHandle);
#else
    ReleaseMutex(mutexHandle);
#endif
  }

  double currentTime(){
#if OCCA_OS == LINUX_OS

    timespec ct;
    clock_gettime(CLOCK_MONOTONIC, &ct);

    return (double) (ct.tv_sec + (1.0e-9 * ct.tv_nsec));

#elif OCCA_OS == OSX_OS

    uint64_t ct;
    ct = mach_absolute_time();

    const Nanoseconds ct2 = AbsoluteToNanoseconds(*(AbsoluteTime *) &ct);

    return ((double) 1.0e-9) * ((double) ( *((uint64_t*) &ct2) ));

#elif OCCA_OS == WINDOWS_OS
    LARGE_INTEGER timestamp, timerfreq;

    QueryPerformanceFrequency(&timerfreq);
    QueryPerformanceCounter(&timestamp);

    return ((double) timestamp.QuadPart) / ((double) timerfreq.QuadPart);
#endif
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

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    for(int i = 0; i < chars; ++i)
      if(fullFilename[i] == '/')
        lastSlash = i;
#else
    for(int i = 0; i < chars; ++i)
      if((fullFilename[i] == '/') ||
         (fullFilename[i] == '\\'))
        lastSlash = i;
#endif

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
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    int mkdirStatus = mkdir(lockDir.c_str(), 0755);

    // Someone else is making it
    if(mkdirStatus && (errno == EEXIST))
      return false;

    return true;
#else
    LPCSTR lockDirStr = lockDir.c_str();
    BOOL mkdirStatus = CreateDirectoryA(lockDirStr, NULL);

    if( mkdirStatus == FALSE) {
      assert(GetLastError() == ERROR_ALREADY_EXISTS);
      return false;
    }
    return true;
#endif
  }

  void waitForFile(const std::string &filename){
    struct stat buffer;

    std::string lockDir   = getFileLock(filename);
    const char *c_lockDir = lockDir.c_str();

    while(stat(c_lockDir, &buffer) == 0)
      /* Do Nothing */;
  }

  void releaseFile(const std::string &filename){
    std::string lockDir = getFileLock(filename);
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    rmdir(lockDir.c_str());
#else
    BOOL retStatus = RemoveDirectoryA(lockDir.c_str());
    assert(retStatus == TRUE);
#endif
  }

  parsedKernelInfo parseFileForFunction(const std::string &filename,
                                        const std::string &cachedBinary,
                                        const std::string &functionName,
                                        const kernelInfo &info){
    const std::string iCachedBinary = getMidCachedBinaryName(cachedBinary, "i");
    const std::string pCachedBinary = getMidCachedBinaryName(cachedBinary, "p");

    parser fileParser;

    std::ofstream fs;
    fs.open(pCachedBinary.c_str());

    fs << info.header << readFile(filename);

    fs.close();
    fs.clear();

    fs.open(iCachedBinary.c_str());
    fs << info.occaKeywords << fileParser.parseFile(pCachedBinary);

    fs.close();

    {
      kernelInfoIterator kIt = fileParser.kernelInfoMap.begin();

      while(kIt != fileParser.kernelInfoMap.end()){
        std::cout
          << "kIt = " << (kIt->first) << '\n';

        ++kIt;
      }
    }

    kernelInfoIterator kIt = fileParser.kernelInfoMap.find(functionName);

    if(kIt != fileParser.kernelInfoMap.end()){
      return parsedKernelInfo( *((kIt++)->second) );
    }

    std::cout << "Could not find function ["
              << functionName << "] in file ["
              << filename     << "].\n";
    throw 1;

    return parsedKernelInfo();
  }

  std::string fnv(const std::string &saltedString){
    const int len = saltedString.size();
    std::stringstream ss;

    int h[8] = {101527, 101531,
                101533, 101537,
                101561, 101573,
                101581, 101599};

    const int p[8] = {102679, 102701,
                      102761, 102763,
                      102769, 102793,
                      102797, 102811};

    for(int c = 0; c < len; ++c)
      for(int i = 0; i < 8; ++i)
        h[i] = (h[i] * p[i]) ^ saltedString[c];

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

    for(int i = 0; i < 8; ++i)
      ss <<  std::hex << h[i];

    return ss.str();
  }

  bool fileExists(const std::string &filename){
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
  }

  std::string readFile(const std::string &filename){
    struct stat fileInfo;

    int fileHandle = ::open(filename.c_str(), O_RDWR);
    const int status = fstat(fileHandle, &fileInfo);

    const int chars = fileInfo.st_size;

    if(status != 0){
      printf("File [%s] does not exist or could not be properly opened\n", filename.c_str());
      throw 1;
    }

    char *buffer = (char*) malloc(chars);
    memset(buffer, '\0', chars);

    std::ifstream fs(filename.c_str());

    if(!fs) {
      std::cerr << "Unable to read file " << filename;
      throw 1;
    }

    fs.read(buffer, chars);

    std::string contents(buffer, chars);

    free(buffer);

    return contents;
  }

  std::string getOCCADir(){
    char *c_occaPath = getenv("OCCA_DIR");

    if(c_occaPath != NULL)
      return c_occaPath;

    std::cout << "Environment variable [OCCA_DIR] is not set.\n";
    throw 1;

    return "";
  }

  std::string getCachePath(){
    char *c_cachePath = getenv("OCCA_CACHE_DIR");

    bool hasDir = false;

    std::string occaCachePath;

    if(c_cachePath != NULL)
      occaCachePath = c_cachePath;
    else{
      std::stringstream ss;
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
      char *c_home = getenv("HOME");
      ss << c_home << "/._occa";

      std::string defaultCacheDir = ss.str();
#else
      char *c_home = getenv("USERPROFILE");

      ss << c_home << "\\AppData\\Local\\OCCA";

#  if OCCA_64_BIT
      ss << "_amd64";  // use different dir's fro 32 and 64 bit
#  else
      ss << "_x86";    // use different dir's fro 32 and 64 bit
#  endif

      occaCachePath = ss.str();
#endif
    }

    const int chars = occaCachePath.size();

    OCCA_CHECK(0 < chars);

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
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
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
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

  std::string getCachedName(const std::string &filename,
                            const std::string &salt){

    std::string occaCachePath = getCachePath();
    //================================

    const std::string fileContents = readFile(filename);
    const std::string contentsSHA  = fnv(fileContents + salt);

    // Only taking the first 16 characters
    return occaCachePath + contentsSHA.substr(0, 16);
  }

  std::string createIntermediateSource(const std::string &filename,
                                       const std::string &cachedBinary,
                                       const kernelInfo &info){
    std::string prefix, name;
    getFilePrefixAndName(cachedBinary, prefix, name);

    std::string extension = getFileExtension(filename);

    const std::string iCachedBinary = prefix + "i_" + name;

    if(extension == "okl"){
      const std::string pCachedBinary = prefix + "p_" + name;
      parser fileParser;

      std::ofstream fs;
      fs.open(pCachedBinary.c_str());

      fs << info.header << readFile(filename);

      fs.close();

      fs.open(iCachedBinary.c_str());
      fs << info.occaKeywords << fileParser.parseFile(pCachedBinary);

      fs.close();
    }
    else{
      std::ofstream fs;
      fs.open(iCachedBinary.c_str());

      fs << info.occaKeywords << info.header << readFile(filename);

      fs.close();
    }

    return iCachedBinary;
  }
};
