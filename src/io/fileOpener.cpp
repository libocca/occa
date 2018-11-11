#include <occa/defines.hpp>
#include <occa/io/fileOpener.hpp>
#include <occa/io/utils.hpp>

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
      if (filename.size() == 7) {
        return cachePath();
      }
      return (libraryPath() + filename.substr(7));
    }
    //==================================
  }
}
