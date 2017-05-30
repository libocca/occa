/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#if (OCCA_OS & OCCA_LINUX_OS)
#  include <dirent.h>
#  include <unistd.h>
#  include <errno.h>
#  include <sys/types.h>
#  include <sys/dir.h>
#elif (OCCA_OS & OCCA_OSX_OS)
#  include <dirent.h>
#  include <sys/types.h>
#  include <sys/dir.h>
#else
#  ifndef NOMINMAX
#    define NOMINMAX  // Clear min/max macros
#  endif
#  include <windows.h>
#  include <string>
#  include <direct.h> // rmdir _rmdir
#endif

#include <fstream>
#include <stddef.h>

#include "occa/parser/parser.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/sys.hpp"
#include "occa/par/tls.hpp"

namespace occa {
  // Kernel Caching
  namespace kc {
    const std::string parsedSourceFile = "parsedSource.cpp";
    const std::string launchSourceFile = "launchSource.cpp";
    const std::string launchBinaryFile = "launch-binary";
    const std::string sourceFile       = "deviceSource.cpp";
    const std::string binaryFile       = "device-binary";
    const std::string infoFile         = "build-info.json";
  }

  namespace io {
    hashMap_t& fileLocks() {
      static tls<hashMap_t> locks;
      return locks.value();
    }

    //---[ File Openers ]---------------
    std::vector<fileOpener*>& fileOpener::getOpeners() {
      static std::vector<fileOpener*> openers;
      return openers;
    }

    fileOpener& fileOpener::defaultOpener() {
      static defaultFileOpener_t fo;
      return fo;
    }

    void fileOpener::add(fileOpener* opener) {
      getOpeners().push_back(opener);
    }

    fileOpener& fileOpener::get(const std::string &filename) {
      std::vector<fileOpener*> &openers = getOpeners();
      for (size_t i = 0; i < openers.size(); ++i) {
        if (openers[i]->handles(filename)) {
          return *(openers[i]);
        }
      }
      return defaultOpener();
    }

    //  ---[ Default File Opener ]------
    defaultFileOpener_t::defaultFileOpener_t() {}

    bool defaultFileOpener_t::handles(const std::string &filename) {
      return true;
    }

    std::string defaultFileOpener_t::expand(const std::string &filename) {
      return filename;
    }
    //  ================================

    //  ---[ OCCA File Opener ]---------
    occaFileOpener_t::occaFileOpener_t() {}

    bool occaFileOpener_t::handles(const std::string &filename) {
      return ((7 <= filename.size()) &&
              (filename.substr(0, 7) == "occa://"));
    }

    std::string occaFileOpener_t::expand(const std::string &filename) {
      if (filename.size() == 7) {
        return cachePath();
      }
      return (libraryPath() + filename.substr(7));
    }
    //  ================================

    //  ---[ Header File Opener ]-------
    headerFileOpener_t::headerFileOpener_t() {}

    bool headerFileOpener_t::handles(const std::string &filename) {
      return ((2 <= filename.size()) &&
              (filename[0] == '"')   &&
              (filename[filename.size() - 1] == '"'));
    }

    std::string headerFileOpener_t::expand(const std::string &filename) {
      return filename.substr(1, filename.size() - 2);
    }
    //  ================================

    //  ---[ System Header File Opener ]---
    systemHeaderFileOpener_t::systemHeaderFileOpener_t() {}

    bool systemHeaderFileOpener_t::handles(const std::string &filename) {
      return ((2 <= filename.size()) &&
              (filename[0] == '<')   &&
              (filename[filename.size() - 1] == '>'));
    }

    std::string systemHeaderFileOpener_t::expand(const std::string &filename) {
      return filename.substr(1, filename.size() - 2);
    }
    //  ================================
    //==================================

    const std::string& cachePath() {
      static std::string path;
      if (path.size() == 0) {
        path = env::OCCA_CACHE_DIR + "cache/";
      }
      return path;
    }

    const std::string& libraryPath() {
      static std::string path;
      if (path.size() == 0) {
        path = env::OCCA_CACHE_DIR + "libraries/";
      }
      return path;
    }

    void endWithSlash(std::string &dir) {
      const int chars = (int) dir.size();
      if ((0 < chars) &&
          (dir[chars - 1] != '/')) {
        dir += '/';
      }
    }

    std::string endWithSlash(const std::string &dir) {
      std::string ret = dir;
      endWithSlash(ret);
      return ret;
    }

    void removeEndSlash(std::string &dir) {
      const int chars = (int) dir.size();
      if ((0 < chars) &&
          (dir[chars - 1] == '/')) {
        dir.erase(chars - 1, 1);
      }
    }

    std::string removeEndSlash(const std::string &dir) {
      std::string ret = dir;
      removeEndSlash(ret);
      return ret;
    }

    bool isAbsolutePath(const std::string &filename) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      return ((0 < filename.size()) &&
              (filename[0] == '/'));
#else
      return ((3 <= filename.size())
              isAlpha(filename[0]) &&
              (filename[1] == ':') &&
              ((filename[2] == '\\') || (filename[2] == '/')));
#endif
    }

    std::string expandEnvVariables(const std::string &filename) {
      const char *c = filename.c_str();
      std::string expFilename = sys::expandEnvVariables(filename);

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      if (*c == '~') {
        expFilename += env::HOME;
        c += 1 + (*c != '\0');
      }
#endif

      return expFilename;
    }

    std::string convertSlashes(const std::string &filename) {
#if (OCCA_OS == OCCA_WINDOWS_OS)
      char slash = '\\';
      if (isAbsolutePath(filename)) {
          slash = filename[2];
      } else {
        const char *c = filename.c_str();
        const char *c0 = c;

        while (*c != '\0') {
          if ((*c == '\\') || (*c == '/')) {
            slash = *c;
            break;
          }
          ++c;
        }
      }
      if (slash == '\\')
        return std::replace(filename.begin(), filename.end(), '\\', '/');
#endif
      return filename;
    }

    std::string filename(const std::string &filename, bool makeAbsolute) {
      std::string expFilename = convertSlashes(expandEnvVariables(filename));

      fileOpener &fo = fileOpener::get(expFilename);
      expFilename = fo.expand(expFilename);

      if (makeAbsolute && !isAbsolutePath(expFilename)) {
        expFilename = env::PWD + expFilename;
      }
      return expFilename;
    }

    std::string binaryName(const std::string &filename) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      return filename;
#else
      return (filename + ".dll");
#endif
    }

    std::string basename(const std::string &filename,
                         const bool keepExtension) {

      const int chars = (int) filename.size();
      const char *c   = filename.c_str();

      int lastSlash = 0;

      for (int i = 0; i < chars; ++i) {
        lastSlash = (c[i] == '/') ? i : lastSlash;
      }
      if (lastSlash || (c[0] == '/')) {
        ++lastSlash;
      }

      if (keepExtension) {
        return filename.substr(lastSlash);
      }
      return filename.substr(lastSlash,
                             filename.size() - extension(filename).size() - 1);
    }

    std::string dirname(const std::string &filename) {
      std::string expFilename = io::filename(filename);
      std::string basename = io::basename(filename);
      return expFilename.substr(0, expFilename.size() - basename.size());
    }

    std::string extension(const std::string &filename) {
      const char *cStart = filename.c_str();
      const char *c = cStart + filename.size();

      while (cStart < c) {
        if (*c == '.') {
          break;
        }
        --c;
      }

      if (*c == '.') {
        return filename.substr(c - cStart + 1);
      }
      return "";
    }

    std::string shortname(const std::string &filename) {
      std::string expFilename = io::filename(filename);

      if (expFilename.find(env::OCCA_CACHE_DIR) != 0) {
        return filename;
      }

      const std::string &lPath = libraryPath();
      const std::string &cPath = cachePath();

      if (expFilename.find(lPath) == 0) {
        std::string libName = getLibraryName(expFilename);
        std::string theRest = expFilename.substr(lPath.size() + libName.size() + 1);

        return ("occa://" + libName + "/" + theRest);
      } else if (expFilename.find(cPath) == 0) {
        return expFilename.substr(cPath.size());
      }

      return expFilename;
    }

    strVector_t filesInDir(const std::string &dir, const unsigned char fileType) {
      strVector_t files;
      const std::string expDir = filename(dir);

      DIR *c_dir = ::opendir(expDir.c_str());
      struct dirent *file;
      while ((file = ::readdir(c_dir)) != NULL) {
        const std::string filename = file->d_name;
        if ((filename == ".") ||
            (filename == "..")) {
          continue;
        }
        if (file->d_type == fileType) {
          std::string fullname = expDir + filename;
          if (fileType == DT_DIR) {
            endWithSlash(fullname);
          }
          files.push_back(fullname);
        }
      }
      ::closedir(c_dir);
      return files;
    }

    strVector_t directories(const std::string &dir) {
      return filesInDir(dir, DT_DIR);
    }

    strVector_t files(const std::string &dir) {
      return filesInDir(dir, DT_REG);
    }

    char* c_read(const std::string &filename, size_t *chars, const bool readingBinary) {
      std::string expFilename = io::filename(filename);

      FILE *fp = fopen(filename.c_str(), readingBinary ? "rb" : "r");
      OCCA_ERROR("Failed to open [" << io::shortname(filename) << "]",
                 fp != 0);

      struct stat statbuf;
      stat(filename.c_str(), &statbuf);

      const size_t nchars = statbuf.st_size;

      char *buffer = (char*) calloc(nchars + 1, sizeof(char));
      size_t nread = fread(buffer, sizeof(char), nchars, fp);

      fclose(fp);
      buffer[nread] = '\0';

      if (chars != NULL) {
        *chars = nread;
      }

      return buffer;
    }

    std::string read(const std::string &filename, const bool readingBinary) {
      size_t chars;
      const char *c = c_read(filename, &chars, readingBinary);

      std::string contents(c, chars);

      ::free((void*) c);
      return contents;
    }

    void write(const std::string &filename, const std::string &content) {
      sys::mkpath(dirname(filename));

      FILE *fp = fopen(filename.c_str(), "w");
      OCCA_ERROR("Failed to open [" << io::shortname(filename) << "]",
                 fp != 0);

      fputs(content.c_str(), fp);

      fclose(fp);
    }

    std::string getFileLock(const hash_t &hash, const std::string &tag) {
      std::string ret = (env::OCCA_CACHE_DIR + "locks/" + hash.toString());
      ret += '_';
      ret += tag;
      return ret;
    }

    void clearLocks() {
      hashMap_t::iterator it = fileLocks().begin();
      while (it != fileLocks().end()) {
        hashAndTag &ht = it->second;
        releaseHash(ht.hash, ht.tag);
        ++it;
      }
      fileLocks().clear();
    }

    bool haveHash(const hash_t &hash, const std::string &tag) {
      std::string lockDir = getFileLock(hash, tag);

      sys::mkpath(env::OCCA_CACHE_DIR + "locks/");

      int mkdirStatus = sys::mkdir(lockDir);

      if (mkdirStatus && (errno == EEXIST)) {
        return false;
      }

      fileLocks()[lockDir] = hashAndTag(hash, tag);

      return true;
    }

    void waitForHash(const hash_t &hash, const std::string &tag) {
      struct stat buffer;

      std::string lockDir   = getFileLock(hash, tag);
      const char *c_lockDir = lockDir.c_str();

      while(stat(c_lockDir, &buffer) == 0)
        ; // Do Nothing
    }

    void releaseHash(const hash_t &hash, const std::string &tag) {
      releaseHashLock(getFileLock(hash, tag));
    }

    void releaseHashLock(const std::string &lockDir) {
      sys::rmdir(lockDir);
      fileLocks().erase(lockDir);
    }

    kernelMetadata parseFileForFunction(const std::string &filename,
                                        const std::string &outputFile,
                                        const std::string &functionName,
                                        const occa::properties &props) {

      const std::string ext = extension(filename);
      parser fileParser;

      std::string parsedContent = fileParser.parseFile(io::filename(filename),
                                                       props);

      if (!sys::fileExists(outputFile)) {
        hash_t hash = occa::hash(outputFile);
        const std::string hashTag = "parse-file";

        if (io::haveHash(hash, hashTag)) {
          write(outputFile, parsedContent);
          io::releaseHash(hash, hashTag);
        } else {
          io::waitForHash(hash, hashTag);
        }
      }

      kernelInfoIterator kIt = fileParser.kernelInfoMap.find(functionName);
      if (kIt != fileParser.kernelInfoMap.end()) {
        return (kIt->second)->metadata();
      }

      OCCA_ERROR("Could not find function ["
                 << functionName << "] in file ["
                 << io::shortname(filename) << "]",
                 false);

      return kernelMetadata();
    }

    std::string removeSlashes(const std::string &str) {
      std::string ret = str;
      const size_t chars = str.size();

      for (size_t i = 0; i < chars; ++i) {
        if (ret[i] == '/')
          ret[i] = '_';
      }

      return ret;
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
      const std::string hashTag = "cache";
      if (!io::haveHash(hash, hashTag)) {
        io::waitForHash(hash, hashTag);
      } else {
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

      return cacheFile(filename, cachedName, occa::hashFile(filename), header, footer);
    }

    std::string cacheFile(const std::string &filename,
                          const std::string &cachedName,
                          const hash_t &hash,
                          const std::string &header,
                          const std::string &footer) {

      const std::string expFilename = io::filename(filename);
      const std::string hashDir     = io::hashDir(expFilename, hash);
      const std::string infoFile    = hashDir + kc::infoFile;
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
      const std::string hashDir  = io::hashDir(filename, hash);
      const std::string infoFile = hashDir + kc::infoFile;

      const std::string hashTag = "kernel-info";
      if (!io::haveHash(hash, hashTag)) {
        return;
      } else if (sys::fileExists(infoFile)) {
        io::releaseHash(hash, hashTag);
        return;
      }

      occa::properties info;
      info["date"]      = sys::date();
      info["humanDate"] = sys::humanDate();
      info["info"]      = props;

      write(infoFile, info.toString());
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

      std::string occaLibName = getLibraryName(io::filename(filename));

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
