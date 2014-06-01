#include "occaTools.hpp"
#include "occa.hpp"      // For kernelInfo

namespace occa {
  std::string fnv(const std::string &saltedString){
    const int len = saltedString.size();
    std::cout << "len = " << len << '\n';
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
    std::ifstream fs(filename.c_str());
    return std::string(std::istreambuf_iterator<char>(fs),
                       std::istreambuf_iterator<char>());
  }

  std::string binaryIsCached(const std::string &filename,
                             const std::string &salt){
    //---[ Place Somewhere Else ]-----
    char *c_cachePath = getenv("OCCA_CACHE_DIR");
    OCCA_CHECK(c_cachePath != NULL);

    std::string occaCachePath(c_cachePath);

    OCCA_CHECK(occaCachePath.size() > 0);


    if(occaCachePath[occaCachePath.size() - 1] != '/')
      occaCachePath += '/';
    //================================

    const std::string fileContents = readFile(filename);
    const std::string contentsSHA  = fnv(fileContents + salt);

    // Only taking the first 16 characters
    return occaCachePath + contentsSHA.substr(0, 16);
  }

  std::string createIntermediateSource(const std::string &filename,
                                       const std::string &cachedBinary,
                                       const kernelInfo &info){
    int lastSlash = 0;
    const int chars = cachedBinary.size();

    for(int i = 0; i < chars; ++i)
      if(cachedBinary[i] == '/')
        lastSlash = i;

    ++lastSlash;
    const std::string iCachedBinary =
      cachedBinary.substr(0, lastSlash) + "i_" + cachedBinary.substr(lastSlash, chars);

    std::ofstream fs;
    fs.open(iCachedBinary.c_str());

    fs << info.occaKeywords << info.header << readFile(filename);

    fs.close();

    return iCachedBinary;
  }
};
