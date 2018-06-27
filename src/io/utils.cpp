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
#include <iostream>
#include <fstream>
#include <vector>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <occa/defines.hpp>

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
#    include <dirent.h>
#else
#  include <windows.h>
#  include <string>
#  include <direct.h> // rmdir _rmdir
#endif

#include <occa/io/cache.hpp>
#include <occa/io/fileOpener.hpp>
#include <occa/io/utils.hpp>
#include <occa/tools/env.hpp>

namespace occa {
  // Kernel Caching
  namespace kc {
    const std::string rawSourceFile  = "raw_source.cpp";
    const std::string sourceFile     = "source.cpp";
    const std::string binaryFile     = "binary";
    const std::string buildFile      = "build.json";
    const std::string hostSourceFile = "host_source.cpp";
    const std::string hostBinaryFile = "host_binary";
    const std::string hostBuildFile  = "host_build.json";
  }

  namespace io {
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
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
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

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      if ((c[0] == '~') && (c[1] == '/')) {
        std::string localPath = filename.substr(2, filename.size() - 1);
        return env::HOME + sys::expandEnvVariables(localPath);
      }
#endif
      return sys::expandEnvVariables(filename);
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
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
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

    strVector filesInDir(const std::string &dir, const unsigned char fileType) {
      strVector files;
      const std::string expDir = filename(dir);

      DIR *c_dir = ::opendir(expDir.c_str());
      if (!c_dir) {
        return files;
      }

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

    strVector directories(const std::string &dir) {
      return filesInDir(endWithSlash(dir), DT_DIR);
    }

    strVector files(const std::string &dir) {
      return filesInDir(dir, DT_REG);
    }

    bool exists(const std::string &filename) {
      std::string expFilename = io::filename(filename);
      FILE *fp = fopen(filename.c_str(), "rb");
      if (fp == NULL) {
        return false;
      }
      fclose(fp);
      return true;
    }

    char* c_read(const std::string &filename,
                 size_t *chars,
                 const bool readingBinary) {
      std::string expFilename = io::filename(filename);

      FILE *fp = fopen(filename.c_str(), readingBinary ? "rb" : "r");
      OCCA_ERROR("Failed to open [" << io::shortname(filename) << "]",
                 fp != NULL);

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
      std::string expFilename = io::filename(filename);
      sys::mkpath(dirname(expFilename));

      FILE *fp = fopen(expFilename.c_str(), "w");
      OCCA_ERROR("Failed to open [" << io::shortname(expFilename) << "]",
                 fp != 0);

      fputs(content.c_str(), fp);

      fclose(fp);
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
  }
}
