#include <iostream>
#include <fstream>
#include <vector>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <occa/defines.hpp>

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
#  include <dirent.h>
#  include <unistd.h>
#else
#  include <windows.h>
#  include <string>
#  include <algorithm> // std::replace
#endif

#include <occa/io/cache.hpp>
#include <occa/io/fileOpener.hpp>
#include <occa/io/utils.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/lex.hpp>
#include <occa/tools/misc.hpp>

namespace occa {
  // Kernel Caching
  namespace kc {
    const std::string cppRawSourceFile   = "raw_source.cpp";
    const std::string cRawSourceFile     = "raw_source.c";
    const std::string sourceFile         = "source.cpp";
    const std::string launcherSourceFile = "launcher_source.cpp";
    const std::string buildFile          = "build.json";
    const std::string launcherBuildFile  = "launcher_build.json";
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    const std::string binaryFile         = "binary";
    const std::string launcherBinaryFile = "launcher_binary";
#else
    const std::string binaryFile         = "binary.dll";
    const std::string launcherBinaryFile = "launcher_binary.dll";
#endif
  }

  namespace io {
    // Might not be defined in Windows
#ifndef DT_REG
    static const unsigned char DT_REG = 'r';
#endif
#ifndef DT_DIR
    static const unsigned char DT_DIR = 'd';
#endif

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

    std::string currentWorkingDirectory() {
      char cwdBuff[FILENAME_MAX];
#if (OCCA_OS == OCCA_WINDOWS_OS)
      ignoreResult(
        _getcwd(cwdBuff, sizeof(cwdBuff))
      );
#else
      ignoreResult(
        getcwd(cwdBuff, sizeof(cwdBuff))
      );
#endif
      return endWithSlash(std::string(cwdBuff));
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

    std::string slashToSnake(const std::string &str) {
      std::string ret = str;
      const size_t chars = str.size();

      for (size_t i = 0; i < chars; ++i) {
        if (ret[i] == '/')
          ret[i] = '_';
      }

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

    std::string getRelativePath(const std::string &filename) {
      if (startsWith(filename, "./")) {
        return filename.substr(2);
      }
      return filename;
    }

    std::string expandEnvVariables(const std::string &filename) {
      const int chars = (int) filename.size();
      if (!chars) {
        return filename;
      }

      const char *c = filename.c_str();
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      if ((c[0] == '~') &&
          ((c[1] == '/') || (c[1] == '\0'))) {
        if (chars == 1) {
          return env::HOME;
        }
        std::string localPath = filename.substr(2, filename.size() - 1);
        return env::HOME + sys::expandEnvVariables(localPath);
      }
#endif
      return sys::expandEnvVariables(filename);
    }

    std::string filename(const std::string &filename, bool makeAbsolute) {
      std::string expFilename = convertSlashes(expandEnvVariables(filename));

      fileOpener &fo = fileOpener::get(expFilename);
      expFilename = fo.expand(expFilename);

      if (makeAbsolute && !isAbsolutePath(expFilename)) {
        return env::CWD + getRelativePath(expFilename);
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
      const char *c = filename.c_str();

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
      int extLength = (int) extension(filename).size();
      // Include the .
      if (extLength) {
        extLength += 1;
      }
      return filename.substr(lastSlash,
                             filename.size() - lastSlash - extLength);
    }

    std::string dirname(const std::string &filename) {
      std::string expFilename = removeEndSlash(io::filename(filename));
      std::string basename = io::basename(expFilename);
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

      if (!startsWith(expFilename, env::OCCA_CACHE_DIR)) {
        return filename;
      }

      const std::string &cPath = cachePath();
      return expFilename.substr(cPath.size());
    }

    bool isFile(const std::string &filename) {
      const std::string expFilename = io::filename(filename);
      struct stat statInfo;
      return ((stat(expFilename.c_str(), &statInfo) == 0) &&
              ((statInfo.st_mode & S_IFMT) == S_IFREG));
    }

    bool isDir(const std::string &filename) {
      const std::string expFilename = io::filename(filename);
      struct stat statInfo;
      return ((stat(expFilename.c_str(), &statInfo) == 0) &&
              ((statInfo.st_mode & S_IFMT) == S_IFDIR));
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
      FILE *fp = fopen(expFilename.c_str(), "rb");
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

      FILE *fp = fopen(expFilename.c_str(), readingBinary ? "rb" : "r");
      OCCA_ERROR("Failed to open [" << io::shortname(expFilename) << "]",
                 fp != NULL);

      struct stat statbuf;
      stat(expFilename.c_str(), &statbuf);

      const size_t nchars = statbuf.st_size;

      char *buffer = new char[nchars + 1];
      ::memset(buffer, 0, nchars + 1);
      size_t nread = fread(buffer, sizeof(char), nchars, fp);

      fclose(fp);
      buffer[nread] = '\0';

      if (chars != NULL) {
        *chars = nread;
      }

      return buffer;
    }

    std::string read(const std::string &filename,
                     const bool readingBinary) {
      size_t chars;
      const char *c = c_read(filename, &chars, readingBinary);
      std::string contents(c, chars);
      delete [] c;
      return contents;
    }

    void write(const std::string &filename,
               const std::string &content) {
      std::string expFilename = io::filename(filename);
      sys::mkpath(dirname(expFilename));

      FILE *fp = fopen(expFilename.c_str(), "w");
      OCCA_ERROR("Failed to open [" << io::shortname(expFilename) << "]",
                 fp != 0);

      fputs(content.c_str(), fp);

      fsync(fileno(fp));
      fclose(fp);
    }
  }
}
