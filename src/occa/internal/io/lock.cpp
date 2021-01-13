#include <cmath>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <occa/defines.hpp>
#include <occa/internal/io/lock.hpp>
#include <occa/internal/io/output.hpp>
#include <occa/internal/io/utils.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/sys.hpp>

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
#  include <unistd.h>
#else
#  include <windows.h> // Sleep
#endif

namespace occa {
  namespace io {
    lock_t::lock_t() :
      isMineCached(false),
      released(true) {}

    lock_t::lock_t(const hash_t &hash,
                   const std::string &tag,
                   const float staleAge_) :
      isMineCached(false),
      staleAge(staleAge_),
      released(false) {

      lockDir = env::OCCA_CACHE_DIR;
      lockDir += "locks/";
      lockDir += hash.getString();
      lockDir += '_';
      lockDir += tag;

      occa::json &lockSettings = settings()["locks"];
      staleWarning = lockSettings.get("stale_warning",
                                      (float) 10.0);
      if (staleAge <= 0) {
        staleAge = lockSettings.get("stale_age",
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
      const char *c_lockDir = lockDir.c_str();
      double startTime = sys::currentTime();
      bool isStale = false;

      struct stat buffer;
      while(!stat(c_lockDir, &buffer)) {
        const double age = ::difftime(::time(NULL),
                                      buffer.st_ctime);
        if (std::abs(age) >= staleAge) {
          isStale = true;
          break;
        }
        // Print warning only once
        if ((sys::currentTime() - startTime) > staleWarning) {
          io::stderr << "Located possible stale hash: ["
                     << lockDir
                     << "]\n";
          staleWarning = staleAge + 10;
        }

        // Wait 0.5 seconds before trying again
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        ::usleep(500000);
#else
        Sleep(500);
#endif
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
