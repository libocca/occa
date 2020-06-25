#ifndef OCCA_TOOLS_ENV_HEADER
#define OCCA_TOOLS_ENV_HEADER

#include <occa/defines.hpp>
#include <occa/types.hpp>
#include <occa/tools/string.hpp>
#include <occa/tools/properties.hpp>

namespace occa {
  properties& settings();

  namespace env {
    extern std::string HOME, CWD;
    extern std::string PATH, LD_LIBRARY_PATH;

    extern std::string OCCA_DIR, OCCA_INSTALL_DIR, OCCA_CACHE_DIR;
    extern size_t      OCCA_MEM_BYTE_ALIGN;
    extern strVector   OCCA_INCLUDE_PATH;
    extern strVector   OCCA_LIBRARY_PATH;
    extern strVector   OCCA_KERNEL_PATH;
    extern bool        OCCA_VERBOSE;
    extern bool        OCCA_COLOR_ENABLED;

    properties& baseSettings();

    std::string var(const std::string &var);

    template <class TM>
    TM get(const std::string &var, const TM &defaultsTo = TM()) {
      const std::string value = env::var(var);
      if (value.size() == 0) {
        return defaultsTo;
      }
      return fromString<TM>(value);
    }

    class envInitializer_t {
    public:
      envInitializer_t();
      ~envInitializer_t();

    private:
      bool isInitialized;

      void initSettings();
      void initEnvironment();
      void loadConfig();

      void setupCachePath();
      void setupIncludePath();
      void registerFileOpeners();

      void cleanFileOpeners();
    };

    extern envInitializer_t envInitializer;
  }
}

#endif
