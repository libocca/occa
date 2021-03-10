#ifndef OCCA_INTERNAL_IO_LOCK_HEADER
#define OCCA_INTERNAL_IO_LOCK_HEADER

#include <iostream>

namespace occa {
  class hash_t;

  namespace io {
    class lock_t {
    private:
      mutable std::string lockDir;
      mutable bool isMineCached;
      float staleWarning;
      float staleAge;
      mutable bool released;

    public:
      lock_t();

      lock_t(const hash_t &hash,
             const std::string &tag,
             const float staleAge_ = -1);

      ~lock_t();

      bool isInitialized() const;

      const std::string& dir() const;

      void release() const;

      bool isMine();

      bool isReleased();
    };
  }
}

#endif
