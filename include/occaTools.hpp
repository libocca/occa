#ifndef OCCA_TOOLS_HEADER
#define OCCA_TOOLS_HEADER

#include <iostream>
// cry

namespace occa {
  class kernelInfo;

  std::string fnv(const std::string &filename);

  std::string readFile(const std::string &filename);

  std::string binaryIsCached(const std::string &filename,
                             const std::string &salt);

  std::string createIntermediateSource(const std::string &filename,
                                       const std::string &cachedBinary,
                                       const kernelInfo &info);
};

#endif
