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

#include <occa/defines.hpp>
#include <occa/io/cache.hpp>
#include <occa/io/lock.hpp>
#include <occa/io/utils.hpp>
#include <occa/tools/hash.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/properties.hpp>

namespace occa {
  namespace io {
    bool isCached(const std::string &filename) {
      // Directory, not file
      if (filename.size() == 0) {
        return false;
      }

      const std::string &cpath = cachePath();

      // File is already cached
      if (startsWith(filename, cpath)) {
        return true;
      }

      std::string occaLibName = getLibraryName(filename);

      if (occaLibName.size() == 0) {
        return false;
      }

      // File is already cached in the library cache
      const std::string lpath = libraryPath() + occaLibName + "/cache/";
      return startsWith(filename, lpath);
    }

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
      io::lock_t lock(hash, "cache");
      if (lock.isMine()
          && !sys::fileExists(expFilename)) {
        write(expFilename, source);
      }
      if (deleteSource) {
        delete [] source;
      }
    }

    std::string cacheFile(const std::string &filename,
                          const std::string &cachedName,
                          const std::string &header) {
      return cacheFile(filename,
                       cachedName,
                       occa::hashFile(filename),
                       header);
    }

    std::string cacheFile(const std::string &filename,
                          const std::string &cachedName,
                          const hash_t &hash,
                          const std::string &header) {
      // File is already cached
      if (isCached(filename)) {
        return filename;
      }

      const std::string expFilename = io::filename(filename);
      const std::string hashDir     = io::hashDir(expFilename, hash);
      const std::string buildFile   = hashDir + kc::buildFile;
      const std::string sourceFile  = hashDir + cachedName;

      if (!sys::fileExists(sourceFile)) {
        std::stringstream ss;
        ss << header << '\n'
           << io::read(expFilename);
        write(sourceFile, ss.str());
      }

      return sourceFile;
    }

    void writeBuildFile(const std::string &filename,
                        const hash_t &hash,
                        const occa::properties &props) {

      io::lock_t lock(hash, "kernel-info");
      if (lock.isMine()
          && !sys::fileExists(filename)) {
        occa::properties info = props;
        json &build = info["build"];
        build["date"]      = sys::date();
        build["humanDate"] = sys::humanDate();

        info.write(filename);
      }
    }

    std::string getLibraryName(const std::string &filename) {
      std::string expFilename = io::filename(filename);
      const std::string cacheLibraryPath = (env::OCCA_CACHE_DIR + "libraries/");

      if (!startsWith(expFilename, cacheLibraryPath)) {
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

    std::string hashDir(const std::string &filename,
                        const hash_t &hash) {
      bool fileIsCached = isCached(filename);

      const std::string &cpath = cachePath();

      // Directory, not file
      if (filename.size() == 0) {
        if (hash.initialized) {
          return (cpath + hash.toString() + "/");
        } else {
          return cpath;
        }
      }

      // File is already cached
      if (fileIsCached &&
          startsWith(filename, cpath)) {
        const char *c = filename.c_str() + cpath.size();
        lex::skipTo(c, '/', '\\');
        if (!c) {
          return filename;
        }
        return filename.substr(0, c - filename.c_str() + 1);
      }

      std::string occaLibName = getLibraryName(filename);

      if (occaLibName.size() == 0) {
        if (hash.initialized) {
          return (cpath + hash.toString() + "/");
        } else {
          return cpath;
        }
      }

      const std::string lpath = libraryPath() + occaLibName + "/cache/";

      // File is already cached
      if (fileIsCached &&
          startsWith(filename, lpath)) {
        const char *c = filename.c_str() + lpath.size();
        lex::skipTo(c, '/', '\\');
        if (!c) {
          return filename;
        }
        return filename.substr(0, c - filename.c_str() + 1);
      }

      return (lpath + hash.toString() + "/");
    }
  }
}
