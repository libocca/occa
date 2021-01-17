#ifndef OCCA_UTILS_HASH_HEADER
#define OCCA_UTILS_HASH_HEADER

#include <iostream>

#include <occa/defines.hpp>
#include <occa/types/typedefs.hpp>

namespace occa {
  namespace io {
    class output;
  }

  // Uses FNV hashing
  class hash_t {
  public:
    bool initialized;
    int h[8];

    mutable std::string h_string;
    mutable int sh[8];

    hash_t();
    hash_t(const int *h_);
    hash_t(const hash_t &hash);
    hash_t& operator = (const hash_t &hash);

    void clear();

    inline bool isInitialized() const { return initialized; }

    bool operator < (const hash_t &fo) const;
    bool operator == (const hash_t &fo) const;
    bool operator != (const hash_t &fo) const;

    hash_t& operator ^= (const hash_t hash);

    template <class TM>
    hash_t operator ^ (const TM &t) const;

    int getInt() const;

    std::string getFullString() const;
    std::string getString() const;
    operator std::string () const;

    static hash_t fromString(const std::string &s);

    friend std::ostream& operator << (std::ostream &out,
                                    const hash_t &hash);
  };
  std::ostream& operator << (std::ostream &out,
                           const hash_t &hash);

  hash_t hash(const void *ptr, udim_t bytes);

  template <class TM>
  inline hash_t hash(const std::vector<TM> &vec) {
    hash_t h;
    for (const TM &value : vec) {
      h ^= hash(value);
    }
    return h;
  }

  template <class TM>
  inline hash_t hash(const TM &t) {
    return hash(&t, sizeof(TM));
  }

  template <class TM>
  inline hash_t hash_t::operator ^ (const TM &t) const {
    return (*this ^ hash(t));
  }

  template <>
  hash_t hash_t::operator ^ (const hash_t &hash) const;

  hash_t hash(const char *c);
  hash_t hash(const std::string &str);
  hash_t hashFile(const std::string &filename);
}

#endif
