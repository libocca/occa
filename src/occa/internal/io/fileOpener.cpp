#include <occa/defines.hpp>
#include <occa/internal/io/fileOpener.hpp>
#include <occa/internal/io/utils.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace io {
    fileOpener::~fileOpener() {}

    std::vector<fileOpener*>& fileOpener::getOpeners() {
      static std::vector<fileOpener*> openers;
      return openers;
    }

    fileOpener& fileOpener::defaultOpener() {
      static defaultFileOpener fo;
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

    //---[ Default File Opener ]---------
    defaultFileOpener::defaultFileOpener() {}
    defaultFileOpener::~defaultFileOpener() {}

    bool defaultFileOpener::handles(const std::string &filename) {
      return true;
    }

    std::string defaultFileOpener::expand(const std::string &filename) {
      return filename;
    }
    //==================================

    //-----[ OCCA File Opener ]---------
    occaFileOpener::occaFileOpener() {}
    occaFileOpener::~occaFileOpener() {}

    bool occaFileOpener::handles(const std::string &filename) {
      return ((7 <= filename.size()) &&
              (filename.substr(0, 7) == "occa://"));
    }

    std::string occaFileOpener::expand(const std::string &filename) {
      const std::string path = filename.substr(7);
      const size_t firstSlash = path.find('/');

      if ((firstSlash == 0) ||
          (firstSlash == std::string::npos)) {
        return "";
      }

      const std::string library = path.substr(0, firstSlash);
      const std::string relativePath = path.substr(firstSlash);

      libraryPathMap_t libraryPaths = getLibraryPathMap();
      libraryPathMap_t::const_iterator it = libraryPaths.find(library);
      if (it == libraryPaths.end()) {
        return "";
      }

      return it->second + relativePath;
    }

    libraryPathMap_t &getLibraryPathMap() {
      static libraryPathMap_t libraryPaths;
      return libraryPaths;
    }
    //==================================
  }
}
