#ifndef OCCA_TOOLS_ENV_HEADER
#define OCCA_TOOLS_ENV_HEADER

#include <occa/defines.hpp>
#include <occa/types.hpp>
#include <occa/tools/string.hpp>
#include <occa/tools/properties.hpp>

namespace occa {
  properties& settings();

  namespace env {
    extern std::string HOME, PWD;
    extern std::string PATH, LD_LIBRARY_PATH;

    extern std::string OCCA_DIR, OCCA_CACHE_DIR;
    extern size_t      OCCA_MEM_BYTE_ALIGN;
    extern strVector   OCCA_PATH;

    properties& baseSettings();

    std::string var(const std::string &var);

    template <class TM>
    TM get(const std::string &var, const TM &defaultsTo = TM()) {
      const std::string v = env::var(var);
      if (v.size() == 0) {
        return defaultsTo;
      }
      return fromString<TM>(var);
    }

    class envInitializer_t {
    public:
      envInitializer_t();
      ~envInitializer_t();

    private:
      bool isInitialized;

      void initSettings();
      void initEnvironment();
      void initCachePath();
      void initIncludePath();
      void registerFileOpeners();

      void cleanFileOpeners();
    };

    extern envInitializer_t envInitializer;
  }
}

#endif
