#ifndef OCCA_INTERNAL_UTILS_ENV_HEADER
#define OCCA_INTERNAL_UTILS_ENV_HEADER

#include <occa/defines.hpp>
#include <occa/types.hpp>
#include <occa/utils/env.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/types/json.hpp>

namespace occa {
  namespace env {
    extern std::string HOME, CWD;
    extern std::string PATH, LD_LIBRARY_PATH;

    extern size_t      OCCA_MEM_BYTE_ALIGN;
    extern strVector   OCCA_INCLUDE_PATH;
    extern strVector   OCCA_LIBRARY_PATH;
    extern strVector   OCCA_KERNEL_PATH;
    extern bool        OCCA_VERBOSE;
    extern bool        OCCA_COLOR_ENABLED;

    json& baseSettings();

    std::string var(const std::string &var);

    template <class TM>
    TM get(const std::string &var, const TM &defaultsTo = TM()) {
      const std::string value = env::var(var);
      if (value.size() == 0) {
        return defaultsTo;
      }
      return fromString<TM>(value);
    }

    void setOccaCacheDir(const std::string &path);

    class envInitializer_t {
     public:
      envInitializer_t();
      ~envInitializer_t();

     private:
      bool isInitialized;

      static void initSettings();
      static void initEnvironment();
      static void loadConfig();

      static void setupCachePath();
      static void setupIncludePath();
      static void registerFileOpeners();

      static void cleanFileOpeners();

      friend void setOccaCacheDir(const std::string &path);
    };

    extern envInitializer_t envInitializer;
  }
}

#endif
