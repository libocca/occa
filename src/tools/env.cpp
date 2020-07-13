#include <cstdlib>

#include <occa/core/base.hpp>
#include <occa/io.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/tools/tls.hpp>

namespace occa {
  properties& settings() {
    static tls<properties> settings_;
    properties& props = settings_.value();
    if (!props.isInitialized()) {
      props = env::baseSettings();
    }
    return props;
  }

  namespace env {
    std::string HOME, CWD;
    std::string PATH, LD_LIBRARY_PATH;

    std::string OCCA_DIR, OCCA_INSTALL_DIR, OCCA_CACHE_DIR;
    size_t      OCCA_MEM_BYTE_ALIGN;
    strVector   OCCA_INCLUDE_PATH;
    strVector   OCCA_LIBRARY_PATH;
    strVector   OCCA_KERNEL_PATH;
    bool        OCCA_VERBOSE;
    bool        OCCA_COLOR_ENABLED;

    properties& baseSettings() {
      static properties settings_;
      return settings_;
    }

    std::string var(const std::string &varName) {
      char *c_varName = getenv(varName.c_str());
      if (c_varName != NULL) {
        return std::string(c_varName);
      }
      return "";
    }

    envInitializer_t::envInitializer_t() :
      isInitialized(false) {
      if (isInitialized) {
        return;
      }

      initSettings();
      initEnvironment();
      loadConfig();

      setupCachePath();
      registerFileOpeners();

      isInitialized = true;
    }

    void envInitializer_t::initSettings() {
      properties &settings_ = baseSettings();
      settings_["version"]     = OCCA_VERSION_STR;
      settings_["okl_version"] = OKL_VERSION_STR;

      OCCA_VERBOSE = env::get<bool>("OCCA_VERBOSE", false);
      if (OCCA_VERBOSE) {
        settings_["device/verbose"] = true;
        settings_["kernel/verbose"] = true;
        settings_["memory/verbose"] = true;
      }
    }

    void envInitializer_t::initEnvironment() {
      // Standard environment variables
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      HOME               = env::var("HOME");
      CWD                = occa::io::currentWorkingDirectory();
      PATH               = env::var("PATH");
      LD_LIBRARY_PATH    = env::var("LD_LIBRARY_PATH");

      OCCA_CACHE_DIR     = env::var("OCCA_CACHE_DIR");
      OCCA_COLOR_ENABLED = env::get<bool>("OCCA_COLOR_ENABLED", true);

      OCCA_INCLUDE_PATH = split(env::var("OCCA_INCLUDE_PATH"), ':', '\\');
      OCCA_LIBRARY_PATH = split(env::var("OCCA_LIBRARY_PATH"), ':', '\\');
      OCCA_KERNEL_PATH  = split(env::var("OCCA_KERNEL_PATH"), ':', '\\');

      io::endWithSlash(HOME);
      io::endWithSlash(CWD);
      io::endWithSlash(PATH);
#endif

      // OCCA environment variables
       OCCA_DIR = env::var("OCCA_DIR");
      if (OCCA_DIR.size() == 0) {
#ifdef OCCA_SOURCE_DIR
        OCCA_DIR = OCCA_SOURCE_DIR;
#else
        OCCA_DIR = OCCA_BUILD_DIR;
#endif
      }
      OCCA_INSTALL_DIR = env::var("OCCA_INSTALL_DIR");
      if (OCCA_INSTALL_DIR.size() == 0) {
        OCCA_INSTALL_DIR = OCCA_BUILD_DIR;
      }
      OCCA_COLOR_ENABLED = env::get<bool>("OCCA_COLOR_ENABLED", true);

      io::endWithSlash(OCCA_DIR);
      io::endWithSlash(OCCA_INSTALL_DIR);
      io::endWithSlash(OCCA_CACHE_DIR);

      OCCA_MEM_BYTE_ALIGN = OCCA_DEFAULT_MEM_BYTE_ALIGN;
      if (env::var("OCCA_MEM_BYTE_ALIGN").size() > 0) {
        const size_t align = (size_t) std::atoi(env::var("OCCA_MEM_BYTE_ALIGN").c_str());

        if ((align != 0) && ((align & (~align + 1)) == align)) {
          OCCA_MEM_BYTE_ALIGN = align;
        } else {
          io::stdout << "Environment variable [OCCA_MEM_BYTE_ALIGN ("
                     << align << ")] is not a power of two, defaulting to "
                     << OCCA_DEFAULT_MEM_BYTE_ALIGN << '\n';
        }
      }
    }

    void envInitializer_t::loadConfig() {
      const std::string configFile = (
        env::get("OCCA_CONFIG",
                 OCCA_CACHE_DIR + "config.json")
      );

      if (!io::exists(configFile)) {
        return;
      }

      properties &settings_ = baseSettings();

      settings_ += json::read(configFile);
    }

    void envInitializer_t::setupCachePath() {
      if (env::OCCA_CACHE_DIR.size() == 0) {
        std::stringstream ss;

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        ss << env::var("HOME") << "/.occa";
#else
        ss << env::var("USERPROFILE") << "/AppData/Local/OCCA";

#  if OCCA_64_BIT
        ss << "_amd64";  // use different dir's fro 32 and 64 bit
#  else
        ss << "_x86";    // use different dir's fro 32 and 64 bit
#  endif
#endif
        env::OCCA_CACHE_DIR = ss.str();
      }
      env::OCCA_CACHE_DIR = io::filename(env::OCCA_CACHE_DIR);
      io::endWithSlash(env::OCCA_CACHE_DIR);

      if (!io::isDir(env::OCCA_CACHE_DIR)) {
        sys::mkpath(env::OCCA_CACHE_DIR);
      }
    }

    void envInitializer_t::registerFileOpeners() {
      io::fileOpener::add(new io::occaFileOpener());
    }

    void envInitializer_t::cleanFileOpeners() {
      std::vector<io::fileOpener*> &openers = io::fileOpener::getOpeners();
      const int count = (int) openers.size();
      for (int i = 0; i < count; ++i) {
        delete openers[i];
      }
      openers.clear();
    }

    envInitializer_t::~envInitializer_t() {
      if (isInitialized) {
        cleanFileOpeners();
      }
    }

    envInitializer_t envInitializer;
  }
}
