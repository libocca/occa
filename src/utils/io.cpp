#include <occa/utils/io.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  namespace io {
    bool exists(const std::string &filename) {
      std::string expFilename = io::expandFilename(filename);
      FILE *fp = fopen(expFilename.c_str(), "rb");
      if (fp == NULL) {
        return false;
      }
      fclose(fp);
      return true;
    }

    void addLibraryPath(const std::string &library,
                        const std::string &path) {
      std::string safeLibrary = library;
      if (endsWith(safeLibrary, "/")) {
        safeLibrary = safeLibrary.substr(0, safeLibrary.size() - 1);
      }
      OCCA_ERROR("Library name cannot be empty",
                 safeLibrary.size());
      OCCA_ERROR("Library name cannot have / characters",
                 safeLibrary.find('/') == std::string::npos);

      std::string safePath = path;
      // Remove the trailing /
      if (endsWith(safePath, "/")) {
        safePath = safePath.substr(0, safePath.size() - 1);
      }
      if (!safePath.size()) {
        return;
      }

      libraryPathMap_t &libraryPaths = getLibraryPathMap();
      libraryPaths[safeLibrary] = safePath;
    }
  }
}
