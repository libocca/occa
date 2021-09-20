#ifndef OCCA_INTERNAL_IO_UTILS_HEADER
#define OCCA_INTERNAL_IO_UTILS_HEADER

#include <functional>
#include <iostream>

#include <occa/types.hpp>
#include <occa/internal/io/enums.hpp>

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
    typedef std::map<std::string, std::string> libraryPathMap_t;

    std::string cachePath();
    std::string libraryPath();

    std::string currentWorkingDirectory();
    libraryPathMap_t &getLibraryPathMap();

    void endWithSlash(std::string &dir);
    std::string endWithSlash(const std::string &dir);

    void removeEndSlash(std::string &dir);
    std::string removeEndSlash(const std::string &dir);

    std::string convertSlashes(const std::string &filename);

    std::string slashToSnake(const std::string &str);

    bool isAbsolutePath(const std::string &filename);
    std::string getRelativePath(const std::string &filename);

    std::string expandEnvVariables(const std::string &filename);

    std::string expandFilename(const std::string &filename,
                               bool makeAbsolute = true);

    std::string expandOccaFilename(const std::string &filename);

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
                 const enums::FileType fileType = enums::FILE_TYPE_TEXT);

    std::string read(const std::string &filename,
                     const enums::FileType fileType = enums::FILE_TYPE_TEXT);

    void sync(const std::string &filename);

    void write(const std::string &filename,
               const std::string &content);

    void stageFile(
      const std::string &filename,
      const bool skipExisting,
      std::function<bool(const std::string &tempFilename)> func
    );

    void stageFiles(
      const strVector &filenames,
      const bool skipExisting,
      std::function<bool(const strVector &tempFilenames)> func
    );

    std::string getStagedTempFilename(const std::string &expFilename);

    void moveStagedTempFile(const std::string &tempFilename,
                            const std::string &expFilename);
  }
}

#endif
