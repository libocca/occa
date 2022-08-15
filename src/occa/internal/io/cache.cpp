#include <occa/defines.hpp>
#include <occa/internal/io/cache.hpp>
#include <occa/internal/io/utils.hpp>
#include <occa/utils/hash.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/lex.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace io {
    bool isCached(const std::string &filename) {
      // Directory, not file
      if (filename.size() == 0) {
        return false;
      }

      std::string expFilename = io::expandFilename(filename);

      // File is already cached
      const std::string &cPath = cachePath();
      return startsWith(expFilename, cPath);
    }

    std::string hashDir(const hash_t &hash) {
      return hashDir("", hash);
    }

    std::string hashDir(const std::string &filename,
                        const hash_t &hash) {
      bool fileIsCached = isCached(filename);

      const std::string &cPath = cachePath();
      std::string cacheDir = cPath;

      const bool useHash = !filename.size() || !fileIsCached;

      // Regular file, use hash
      if (useHash) {
        if (hash.initialized) {
          return (cacheDir + hash.getString() + "/");
        }
        return cacheDir;
      }

      // Extract hash out of filename
      const char *c = filename.c_str() + cacheDir.size();
      lex::skipTo(c, '/');
      if (!c) {
        return filename;
      }
      return filename.substr(0, c - filename.c_str() + 1);
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
      const std::string expFilename = io::expandFilename(filename);
      const std::string hashDir     = io::hashDir(expFilename, hash);
      const std::string buildFile   = hashDir + kc::buildFile;
      const std::string sourceFile  = hashDir + cachedName;

      // File is already cached
      if (filename == sourceFile) {
        return filename;
      }

      if (!io::isFile(sourceFile)) {
        std::stringstream ss;
        ss << header << '\n'
           << io::read(expFilename);
        io::stageFile(
          sourceFile,
          true,
          [&](const std::string &tempFilename) -> bool {
            io::write(tempFilename, ss.str());
            return true;
          }
        );
      }
      return sourceFile;
    }

    bool cachedFileIsComplete(const std::string &hashDir,
                              const std::string &filename) {
      std::string successFile = hashDir;
      successFile += filename;

      return io::exists(successFile);
    }

    void setBuildProps(occa::json &props) {
      props["date"]       = sys::date();
      props["human_date"] = sys::humanDate();
      props["version/occa"] = OCCA_VERSION_STR;
      props["version/okl"]  = OKL_VERSION_STR;
    }

    void writeBuildFile(const std::string &filename,
                        const occa::json &props) {
      io::stageFile(
        filename,
        true,
        [&](const std::string &tempFilename) -> bool {
          occa::json info = props;
          setBuildProps(info["build"]);
          info.write(tempFilename);

          return true;
        }
      );
    }
  }
}
