/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include "occa/defines.hpp"
#include "occa/io/cache.hpp"
#include "occa/io/lock.hpp"
#include "occa/io/utils.hpp"
#include "occa/tools/hash.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/properties.hpp"

namespace occa {
  namespace io {
    void cache(const std::string &filename,
               std::string source,
               const hash_t &hash) {

      cache(filename, source.c_str(), hash, false);
    }

    void cache(const std::string &filename,
               const char *source,
               const hash_t &hash,
               const bool deleteSource) {

      const std::string expFilename = io::filename(filename);
      const std::string hashTag = "cache";
      if (io::haveHash(hash, hashTag)) {
        if (!sys::fileExists(expFilename)) {
          write(expFilename, source);
        }
        io::releaseHash(hash, hashTag);
      }
      if (deleteSource) {
        delete [] source;
      }
    }

    std::string cacheFile(const std::string &filename,
                          const std::string &cachedName,
                          const std::string &header,
                          const std::string &footer) {

      return cacheFile(filename,
                       cachedName,
                       occa::hashFile(filename),
                       header,
                       footer);
    }

    std::string cacheFile(const std::string &filename,
                          const std::string &cachedName,
                          const hash_t &hash,
                          const std::string &header,
                          const std::string &footer) {

      const std::string expFilename = io::filename(filename);
      const std::string hashDir     = io::hashDir(expFilename, hash);
      const std::string buildFile   = hashDir + kc::buildFile;
      const std::string sourceFile  = hashDir + cachedName;

      if (!sys::fileExists(sourceFile)) {
        std::stringstream ss;
        ss << header                << '\n'
           << io::read(expFilename) << '\n'
           << footer;
        write(sourceFile, ss.str());
      }

      return sourceFile;
    }

    void storeCacheInfo(const std::string &filename,
                        const hash_t &hash,
                        const occa::properties &props) {
      const std::string hashDir   = io::hashDir(filename, hash);
      const std::string buildFile = hashDir + kc::buildFile;

      const std::string hashTag = "kernel-info";
      if (io::haveHash(hash, hashTag)) {
        if (!sys::fileExists(buildFile)) {
          occa::properties info;
          info["date"]      = sys::date();
          info["humanDate"] = sys::humanDate();
          info["info"]      = props;

          write(buildFile, info.toString());
        }
        io::releaseHash(hash, hashTag);
      }
    }

    std::string getLibraryName(const std::string &filename) {
      std::string expFilename = io::filename(filename);
      const std::string cacheLibraryPath = (env::OCCA_CACHE_DIR + "libraries/");

      if (expFilename.find(cacheLibraryPath) != 0) {
        return "";
      }
      const int chars = (int) expFilename.size();
      const char *c   = expFilename.c_str();

      int start = (int) cacheLibraryPath.size();
      int end;

      for (end = start; end < chars; ++end) {
        if (c[end] == '/') {
          break;
        }
      }
      return expFilename.substr(start, end - start);
    }

    std::string hashFrom(const std::string &filename) {
      std::string expFilename = io::filename(filename);
      std::string dir = hashDir(expFilename);

      const int chars = (int) expFilename.size();
      const char *c   = expFilename.c_str();

      int start = (int) dir.size();
      int end;

      for (end = (start + 1); end < chars; ++end) {
        if (c[end] == '/') {
          break;
        }
      }

      return expFilename.substr(start, end - start);
    }

    std::string hashDir(const hash_t &hash) {
      return hashDir("", hash);
    }

    std::string hashDir(const std::string &filename, const hash_t &hash) {
      if (filename.size() == 0) {
        if (hash.initialized) {
          return (cachePath() + hash.toString() + "/");
        } else {
          return cachePath();
        }
      }

      std::string occaLibName = getLibraryName(filename);

      if (occaLibName.size() == 0) {
        if (hash.initialized) {
          return (cachePath() + hash.toString() + "/");
        } else {
          return (cachePath());
        }
      }

      return (libraryPath() + occaLibName + "/" + hash.toString() + "/");
    }
  }
}
