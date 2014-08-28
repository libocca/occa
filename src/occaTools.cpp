#include "occaTools.hpp"
#include "occa.hpp"      // For kernelInfo

namespace occa {


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

    return ((double)(timestamp.QuadPart))/((double)(timerfreq.QuadPart));
#endif
  }

  void getFilePrefixAndName(const std::string &fullFilename,
                            std::string &prefix,
                            std::string &filename){
    int lastSlash = 0;
    const int chars = fullFilename.size();

    for(int i = 0; i < chars; ++i)
      if(fullFilename[i] == '/')
        lastSlash = i;

    ++lastSlash;

    prefix   = fullFilename.substr(0, lastSlash);
    filename = fullFilename.substr(lastSlash, chars - lastSlash);
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

  std::string readFile(const std::string &filename){
    struct stat fileInfo;

    int fileHandle = ::open(filename.c_str(), O_RDWR);
    const int status = fstat(fileHandle, &fileInfo);

    const int chars = fileInfo.st_size;

    if(status != 0)
      printf("File [%s] gave a bad stat", filename.c_str());

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

  std::string getCachedName(const std::string &filename,
                            const std::string &salt){
    //---[ Place Somewhere Else ]-----
    char *c_cachePath = getenv("OCCA_CACHE_DIR");

    std::string occaCachePath;

    if(c_cachePath == NULL){
      std::stringstream ss;
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
      char *c_home = getenv("HOME");
      ss << c_home << "/._occa";

      std::string defaultCacheDir = ss.str();
      mkdir(defaultCacheDir.c_str(), 0755);
#else
      char *c_home = getenv("USERPROFILE");

      ss << c_home << "\\AppData\\Local\\OCCA";
#  if OCCA_64_BIT
      ss << "\\amd64";  // use different dir's fro 32 and 64 bit
#  else
      ss << "\\x86";    // use different dir's fro 32 and 64 bit
#  endif

      std::string defaultCacheDir = ss.str();

      LPCSTR C_defaultCacheDir = defaultCacheDir.c_str();
      CreateDirectoryA(C_defaultCacheDir, NULL);
#endif
      occaCachePath = defaultCacheDir;
    }
    else
      occaCachePath = c_cachePath;

    const int chars = occaCachePath.size();

    OCCA_CHECK(chars > 0);

    // Take out the pesky //'s
    int pos = 0;

    for(int i = 0; i < chars; ++i){
      if(occaCachePath[i] == '/')
        while(i < (chars - 1) && occaCachePath[i + 1] == '/')
          ++i;

      occaCachePath[pos++] = occaCachePath[i];
    }

    if(occaCachePath[pos - 1] != '/'){
      if(pos != chars)
        occaCachePath[pos] = '/';
      else
        occaCachePath += '/';
    }
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

    const std::string iCachedBinary = prefix + "i_" + name;

    std::ofstream fs;
    fs.open(iCachedBinary.c_str());

    fs << info.occaKeywords << info.header << readFile(filename);

    fs.close();

    return iCachedBinary;
  }
};
