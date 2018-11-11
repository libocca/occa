#ifndef OCCA_IO_FILEOPENER_HEADER
#define OCCA_IO_FILEOPENER_HEADER

#include <iostream>
#include <map>
#include <vector>

namespace occa {
  namespace env {
    class envInitializer_t;
  }

  namespace io {
    typedef std::map<std::string, std::string> libraryPathMap_t;
    class fileOpener {
      friend class occa::env::envInitializer_t;

    private:
      static std::vector<fileOpener*>& getOpeners();
      static fileOpener& defaultOpener();

    public:
      static fileOpener& get(const std::string &filename);
      static void add(fileOpener* opener);

      virtual ~fileOpener();

      virtual bool handles(const std::string &filename) = 0;
      virtual std::string expand(const std::string &filename) = 0;
    };

    //---[ Default File Opener ]---------
    class defaultFileOpener : public fileOpener {
    public:
      defaultFileOpener();
      virtual ~defaultFileOpener();

      bool handles(const std::string &filename);
      std::string expand(const std::string &filename);
    };
    //==================================

    //---[ OCCA File Opener ]------------
    class occaFileOpener : public fileOpener {
    public:
      occaFileOpener();
      virtual ~occaFileOpener();

      bool handles(const std::string &filename);
      std::string expand(const std::string &filename);
    };

    libraryPathMap_t &getLibraryPathMap();

    void addLibraryPath(const std::string &library,
                        const std::string &path);
    //==================================
  }
}

#endif
