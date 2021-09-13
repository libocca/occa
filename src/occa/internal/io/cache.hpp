#ifndef OCCA_INTERNAL_IO_CACHE_HEADER
#define OCCA_INTERNAL_IO_CACHE_HEADER

#include <iostream>

#include <occa/utils/hash.hpp>

namespace occa {
  class json;

  namespace io {
    bool isCached(const std::string &filename);

    std::string hashDir(const hash_t &hash);

    std::string hashDir(const std::string &filename,
                        const hash_t &hash = hash_t());

    std::string cacheFile(const std::string &filename,
                          const std::string &cachedName,
                          const std::string &header = "");

    std::string cacheFile(const std::string &filename,
                          const std::string &cachedName,
                          const hash_t &hash,
                          const std::string &header = "");

    bool cachedFileIsComplete(const std::string &hashDir,
                              const std::string &filename);

    void setBuildProps(occa::json &props);

    void writeBuildFile(const std::string &filename,
                        const occa::json &props);
  }
}

#endif
