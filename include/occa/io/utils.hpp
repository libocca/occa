#ifndef OCCA_IO_UTILS_HEADER
#define OCCA_IO_UTILS_HEADER

#include <iostream>

#include <occa/types.hpp>

namespace occa {
  // Kernel Caching
  namespace kc {
    extern const std::string cppRawSourceFile;
    extern const std::string cRawSourceFile;
    extern const std::string sourceFile;
    extern const std::string binaryFile;
    extern const std::string buildFile;
    extern const std::string launcherSourceFile;
    extern const std::string launcherBinaryFile;
    extern const std::string launcherBuildFile;
  }

  namespace io {
    const std::string& cachePath();
    const std::string& libraryPath();

    std::string currentWorkingDirectory();

    void endWithSlash(std::string &dir);
    std::string endWithSlash(const std::string &dir);

    void removeEndSlash(std::string &dir);
    std::string removeEndSlash(const std::string &dir);

    std::string convertSlashes(const std::string &filename);

    std::string slashToSnake(const std::string &str);

    bool isAbsolutePath(const std::string &filename);
    std::string getRelativePath(const std::string &filename);

    std::string expandEnvVariables(const std::string &filename);

    std::string filename(const std::string &filename,
                         bool makeAbsolute = true);

    std::string binaryName(const std::string &filename);

    std::string basename(const std::string &filename,
                         const bool keepExtension = true);

    std::string dirname(const std::string &filename);

    std::string extension(const std::string &filename);

    std::string shortname(const std::string &filename);

    std::string findInPaths(const std::string &filename, const strVector &paths);

    bool isDir(const std::string &filename);
    bool isFile(const std::string &filename);

    strVector filesInDir(const std::string &dir,
                         const unsigned char fileType);

    strVector directories(const std::string &dir);

    strVector files(const std::string &dir);

    bool exists(const std::string &filename);

    char* c_read(const std::string &filename,
                 size_t *chars = NULL,
                 const bool readingBinary = false);

    std::string read(const std::string &filename,
                     const bool readingBinary = false);

    void write(const std::string &filename,
               const std::string &content);
  }
}

#endif
