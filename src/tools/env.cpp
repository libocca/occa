/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include <signal.h>

#include "occa/base.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/sys.hpp"
#include "occa/parser/tools.hpp"

namespace occa {
  namespace env {
    std::string HOME, PWD;
    std::string PATH, LD_LIBRARY_PATH;

    std::string OCCA_DIR, OCCA_CACHE_DIR;
    size_t OCCA_MEM_BYTE_ALIGN;
    strVector_t OCCA_INCLUDE_PATH;

    void initialize() {
      static bool isInitialized = false;
      if (isInitialized) {
        return;
      }

      initSettings();
      initSignalHandling();
      initEnvironment();
      registerFileOpeners();

      isInitialized = true;
    }

    void initSettings() {
      properties &settings_ = settings();
      settings_["version"] = "1.0";
      settings_["parser-version"] = "0.1";
      settings_["verboseCompilation"] = true;
    }

    void initSignalHandling() {
      // Signal handling
      ::signal(SIGTERM, env::signalExit);
      ::signal(SIGINT , env::signalExit);
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      ::signal(SIGKILL, env::signalExit);
      ::signal(SIGQUIT, env::signalExit);
#endif
    }

    void initEnvironment() {
      // Standard environment variables
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
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
#ifdef OCCA_COMPILED_DIR
      if (OCCA_DIR.size() == 0) {
        OCCA_DIR = OCCA_COMPILED_DIR;
      }
#endif
      OCCA_ERROR("Environment variable [OCCA_DIR] is not set",
                 0 < OCCA_DIR.size());

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

    void initCachePath() {
      env::OCCA_CACHE_DIR = env::var("OCCA_CACHE_DIR");

      if (env::OCCA_CACHE_DIR.size() == 0) {
        std::stringstream ss;

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
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

      const int chars = env::OCCA_CACHE_DIR.size();

      OCCA_ERROR("Path to the OCCA caching directory is not set properly, "
                 "unset OCCA_CACHE_DIR to use default directory [~/.occa]",
                 0 < chars);

      env::OCCA_CACHE_DIR = io::filename(env::OCCA_CACHE_DIR);

      if (!sys::dirExists(env::OCCA_CACHE_DIR)) {
        sys::mkpath(env::OCCA_CACHE_DIR);
      }
    }

    void initIncludePath() {
      env::OCCA_INCLUDE_PATH.clear();
      std::string oip = env::var("OCCA_INCLUDE_PATH");

      const char *cStart = oip.c_str();
      const char *cEnd;

      strVector_t tmpOIP;

      while(cStart[0] != '\0') {
        cEnd = cStart;
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
        skipTo(cEnd, ':');
#else
        skipTo(cEnd, ';');
#endif

        if (0 < (cEnd - cStart)) {
          std::string newPath(cStart, cEnd - cStart);
          newPath = io::filename(newPath);
          io::endWithSlash(newPath);

          tmpOIP.push_back(newPath);
        }

        cStart = (cEnd + (cEnd[0] != '\0'));
      }

      tmpOIP.swap(env::OCCA_INCLUDE_PATH);
    }

    void registerFileOpeners() {
      io::fileOpener::add(new io::occaFileOpener_t());
      io::fileOpener::add(new io::headerFileOpener_t());
      io::fileOpener::add(new io::systemHeaderFileOpener_t());
    }

    std::string var(const std::string &varName) {
      char *c_varName = getenv(varName.c_str());
      if (c_varName != NULL) {
        return std::string(c_varName);
      }
      return "";
    }

    void signalExit(int sig) {
      io::clearLocks();
      ::exit(sig);
    }

    envInitializer_t::envInitializer_t() {
      env::initialize();
    }
    envInitializer_t envInitializer;
  }
}
