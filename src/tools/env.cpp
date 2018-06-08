/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include <cstdlib>

#include <occa/base.hpp>
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
    std::string HOME, PWD;
    std::string PATH, LD_LIBRARY_PATH;

    std::string OCCA_DIR, OCCA_CACHE_DIR;
    size_t OCCA_MEM_BYTE_ALIGN;
    strVector OCCA_PATH;

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
      registerFileOpeners();

      isInitialized = true;
    }

    void envInitializer_t::initSettings() {
      properties &settings_ = baseSettings();
      settings_["version"]     = OCCA_VERSION_STR;
      settings_["okl-version"] = OKL_VERSION_STR;

      const bool isVerbose = env::get("OCCA_VERBOSE", false);
      if (isVerbose) {
        settings_["device/verbose"] = true;
        settings_["kernel/verbose"] = true;
        settings_["memory/verbose"] = true;
      }
    }

    void envInitializer_t::initEnvironment() {
      // Standard environment variables
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      HOME            = env::var("HOME");
      PWD             = env::var("PWD");
      PATH            = env::var("PATH");
      LD_LIBRARY_PATH = env::var("LD_LIBRARY_PATH");

      io::endWithSlash(HOME);
      io::endWithSlash(PWD);
      io::endWithSlash(PATH);
#endif

      // OCCA environment variables
      OCCA_DIR = env::var("OCCA_DIR");
      if (OCCA_DIR.size() == 0) {
        OCCA_DIR = io::filename(io::dirname(__FILE__) + "/../..");
      }

      initCachePath();
      initIncludePath();

      io::endWithSlash(OCCA_DIR);
      io::endWithSlash(OCCA_CACHE_DIR);

      OCCA_MEM_BYTE_ALIGN = OCCA_DEFAULT_MEM_BYTE_ALIGN;
      if (env::var("OCCA_MEM_BYTE_ALIGN").size() > 0) {
        const size_t align = (size_t) std::atoi(env::var("OCCA_MEM_BYTE_ALIGN").c_str());

        if ((align != 0) && ((align & (~align + 1)) == align)) {
          OCCA_MEM_BYTE_ALIGN = align;
        } else {
          std::cout << "Environment variable [OCCA_MEM_BYTE_ALIGN ("
                    << align << ")] is not a power of two, defaulting to "
                    << OCCA_DEFAULT_MEM_BYTE_ALIGN << '\n';
        }
      }
    }

    void envInitializer_t::initCachePath() {
      env::OCCA_CACHE_DIR = env::var("OCCA_CACHE_DIR");

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

      if (!sys::dirExists(env::OCCA_CACHE_DIR)) {
        sys::mkpath(env::OCCA_CACHE_DIR);
      }
    }

    void envInitializer_t::initIncludePath() {
      strVector &oipVec = env::OCCA_PATH;
      oipVec.clear();
      std::string oip = env::var("OCCA_PATH");

      const char *cStart = oip.c_str();
      const char *cEnd;

      while(cStart[0] != '\0') {
        cEnd = cStart;
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        lex::skipTo(cEnd, ':');
#else
        lex::skipTo(cEnd, ';');
#endif

        if (0 < (cEnd - cStart)) {
          std::string newPath(cStart, cEnd - cStart);
          newPath = io::filename(newPath);
          io::endWithSlash(newPath);

          oipVec.push_back(newPath);
        }

        cStart = (cEnd + (cEnd[0] != '\0'));
      }
    }

    void envInitializer_t::registerFileOpeners() {
      io::fileOpener::add(new io::occaFileOpener());
      io::fileOpener::add(new io::headerFileOpener());
      io::fileOpener::add(new io::systemHeaderFileOpener());
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
