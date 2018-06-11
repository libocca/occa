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
#include <cmath>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <occa/defines.hpp>
#include <occa/io/lock.hpp>
#include <occa/tools/env.hpp>

namespace occa {
  namespace io {
    lock_t::lock_t() :
      isMineCached(false),
      released(true) {}

    lock_t::lock_t(const hash_t &hash,
                   const std::string &tag,
                   const int staleAge_) :
      isMineCached(false),
      staleAge(staleAge_),
      released(false) {

      lockDir = env::OCCA_CACHE_DIR;
      lockDir += "locks/";
      lockDir += hash.toString();
      lockDir += '_';
      lockDir += tag;

      occa::json &lockSettings = settings()["locks"];
      staleWarning = lockSettings.get("stale-warning",
                                      (float) 10.0);
      if (staleAge <= 0) {
        staleAge = lockSettings.get("stale-age",
                                    (float) 20.0);
      }
    }

    lock_t::~lock_t() {
      release();
    }

    bool lock_t::isInitialized() const {
      return lockDir.size();
    }

    const std::string& lock_t::dir() const {
      return lockDir;
    }

    void lock_t::release() const {
      if (!released) {
        sys::rmdir(lockDir);
        released = true;
      }
    }

    void lock_t::release(const hash_t &hash,
                         const std::string &tag) {
      lock_t(hash, tag).release();
    }

    bool lock_t::isMine() {
      if (isMineCached) {
        return true;
      }
      sys::mkpath(env::OCCA_CACHE_DIR + "locks/");

      while (true) {
        int mkdirStatus = sys::mkdir(lockDir);

        if (!mkdirStatus
            || (errno != EEXIST)) {
          isMineCached = true;
          return true;
        }

        if (isReleased()) {
          break;
        }
      }
      return false;
    }

    bool lock_t::isReleased() {
      struct stat buffer;

      const char *c_lockDir = lockDir.c_str();

      bool isStale = false;

      double startTime = sys::currentTime();
      while(!stat(c_lockDir, &buffer)) {
        const double age = ::difftime(::time(NULL),
                                      buffer.st_ctime);
        if (std::abs(age) >= staleAge) {
          isStale = true;
          break;
        }
        // Print warning only once
        if ((sys::currentTime() - startTime) > staleWarning) {
          std::cerr << "Located possible stale hash: ["
                    << lockDir
                    << "]\n";
          staleWarning = staleAge + 10;
        }

        // Wait 0.5 seconds before trying again
        ::usleep(500000);
      }

      // Other process released the hash
      if (!isStale) {
        return true;
      }
      // Delete lock and recreate it
      release();
      released = false;

      return false;
    }
  }
}
