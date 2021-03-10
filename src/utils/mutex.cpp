#include <occa/utils/mutex.hpp>
#include <occa/utils/logging.hpp>

namespace occa {
  mutex_t::mutex_t() :
    mutexHandle(PTHREAD_MUTEX_INITIALIZER) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    int error = pthread_mutex_init(&mutexHandle, NULL);
#if OCCA_UNSAFE
    ignoreResult(error);
#endif

    OCCA_ERROR("Error initializing mutex",
               error == 0);
#else
    mutexHandle = CreateMutex(NULL, FALSE, NULL);
#endif
  }

  void mutex_t::free() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    int error = pthread_mutex_destroy(&mutexHandle);
#if OCCA_UNSAFE
    ignoreResult(error);
#endif

    OCCA_ERROR("Error freeing mutex",
               error == 0);
#else
    CloseHandle(mutexHandle);
#endif
  }

  void mutex_t::lock() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    pthread_mutex_lock(&mutexHandle);
#else
    WaitForSingleObject(mutexHandle, INFINITE);
#endif
  }

  void mutex_t::unlock() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    pthread_mutex_unlock(&mutexHandle);
#else
    ReleaseMutex(mutexHandle);
#endif
  }
}
