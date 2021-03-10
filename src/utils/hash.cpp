#include <sstream>
#include <stdint.h>

#include <occa/types.hpp>
#include <occa/utils/hash.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>

namespace occa {
  hash_t::hash_t() {
    initialized = false;
    h[0] = 101527; h[1] = 101531;
    h[2] = 101533; h[3] = 101537;
    h[4] = 101561; h[5] = 101573;
    h[6] = 101581; h[7] = 101599;
    for (int i = 0; i < 8; ++i) {
      sh[i] = 0;
    }
  }

  hash_t::hash_t(const int *h_) {
    initialized = true;
    for (int i = 0; i < 8; ++i) {
      h[i] = h_[i];
    }
    for (int i = 0; i < 8; ++i) {
      sh[i] = 0;
    }
  }

  hash_t::hash_t(const hash_t &hash) {
    *this = hash;
  }

  hash_t& hash_t::operator = (const hash_t &hash) {
    initialized = hash.initialized;
    for (int i = 0; i < 8; ++i) {
      h[i] = hash.h[i];
    }
    for (int i = 0; i < 8; ++i) {
      sh[i] = 0;
    }
    return *this;
  }

  void hash_t::clear() {
    *this = hash_t();
  }

  bool hash_t::operator < (const hash_t &fo) const {
    for (int i = 0; i < 8; ++i) {
      if (h[i] < fo.h[i]) {
        return true;
      } else if (h[i] > fo.h[i]) {
        return false;
      }
    }
    return false;
  }

  bool hash_t::operator == (const hash_t &fo) const {
    for (int i = 0; i < 8; ++i) {
      if (h[i] != fo.h[i]) {
        return false;
      }
    }
    return true;
  }

  bool hash_t::operator != (const hash_t &fo) const {
    for (int i = 0; i < 8; ++i) {
      if (h[i] != fo.h[i]) {
        return true;
      }
    }
    return false;
  }

  template <>
  hash_t hash_t::operator ^ (const hash_t &hash) const {
    hash_t mix;
    for (int i = 0; i < 8; ++i) {
      mix.h[i] = (h[i] ^ hash.h[i]);
    }
    mix.initialized = true;
    return mix;
  }

  hash_t& hash_t::operator ^= (const hash_t hash) {
    *this = (*this ^ hash);
    return *this;
  }

  int hash_t::getInt() const {
    return h[0];
  }

  std::string hash_t::getFullString() const {
    std::string ret;
    for (int i = 0; i < 8; ++i) {
      ret += toHex(h[i]);
    }
    return ret;
  }

  std::string hash_t::getString() const {
    if (*this != hash_t(sh)) {
      h_string = getFullString();
      h_string = (h_string.size() < 16) ? h_string : h_string.substr(0, 16);
      for (int i = 0; i < 8; ++i) {
        sh[i] = h[i];
      }
    }
    return h_string;
  }

  hash_t::operator std::string () const {
    return getString();
  }

  hash_t hash_t::fromString(const std::string &s) {
    hash_t hash;
    fromHex(s, hash.h, 8 * sizeof(int));
    hash.initialized = true;
    return hash;
  }

  std::ostream& operator << (std::ostream &out, const hash_t &hash) {
    out << hash.getString();
    return out;
  }

  hash_t hash(const void *ptr, udim_t bytes) {
    std::stringstream ss;
    const char *c = (char*) ptr;

    hash_t hash;
    int *h = hash.h;

    const int p[8] = {
      102679, 102701, 102761, 102763,
      102769, 102793, 102797, 102811
    };

    for (udim_t i = 0; i < bytes; ++i) {
      for (int j = 0; j < 8; ++j) {
        h[j] = (h[j] * p[j]) ^ c[i];
      }
    }
    hash.initialized = true;

    return hash;
  }

  hash_t hash(const char *c) {
    return hash(c, strlen(c));
  }

  hash_t hash(const std::string &str) {
    return hash(str.c_str(), str.size());
  }

  hash_t hashFile(const std::string &filename) {
    const char *c = io::c_read(io::expandFilename(filename));
    hash_t ret = hash(c);
    delete [] c;
    return ret;
  }
}
