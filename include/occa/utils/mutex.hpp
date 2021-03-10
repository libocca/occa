#ifndef OCCA_UTILS_MUTEX_HEADER
#define OCCA_UTILS_MUTEX_HEADER

#include <occa/defines.hpp>

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
#  include <pthread.h>
#endif

namespace occa {
  class mutex_t {
  public:
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    pthread_mutex_t mutexHandle;
#else
    void *mutexHandle;
#endif

    mutex_t();
    void free();

    void lock();
    void unlock();
  };
}

#endif
