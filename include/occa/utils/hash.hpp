#ifndef OCCA_UTILS_HASH_HEADER
#define OCCA_UTILS_HASH_HEADER

#include <iostream>

#include <occa/defines.hpp>
#include <occa/types/typedefs.hpp>

namespace occa {
  namespace io {
    class output;
  }

  /**
   * @startDoc{hash_t}
   *
   * Description:
   *   An object used to represent a hash value.
   *   It's intent isn't for security purposes, but rather to distinguish "things".
   *
   *   > It currently uses FNV hashing since it's quick, but it can be changed if something more useful shows up.
   *
   * @endDoc
   */
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

    /**
     * @startDoc{isInitialized}
     *
     * Description:
     *   Return whether the [[hash_t]] was initialized
     *
     * @endDoc
     */
    inline bool isInitialized() const { return initialized; }

    /**
     * @startDoc{operator_less_than}
     *
     * Description:
     *   Implemented for comparison purposes, such as sorting
     *
     * @endDoc
     */
    bool operator < (const hash_t &fo) const;

    /**
     * @startDoc{operator_equals[1]}
     *
     * Description:
     *   Returns `true` if two [[hash_t]] objects are the same
     *
     * @endDoc
     */
    bool operator == (const hash_t &fo) const;

    /**
     * @startDoc{operator_equals[1]}
     *
     * Description:
     *   Returns `true` if two [[hash_t]] objects are the different
     *
     * @endDoc
     */
    bool operator != (const hash_t &fo) const;

    /**
     * @startDoc{operator_xor[0]}
     *
     * Description:
     *   Apply a XOR (`^`) operation between two hashes, a common way to "combine" hashes
     *
     * Returns:
     *   A new hash
     *
     * @endDoc
     */
    template <class T>
    hash_t operator ^ (const T &t) const;

    /**
     * @startDoc{operator_xor[1]}
     *
     * Description:
     *   Same as above but applies it inplace
     *
     * Returns:
     *   The same [[hash_t]] as the caller
     *
     * @endDoc
     */
    hash_t& operator ^= (const hash_t hash);

    /**
     * @startDoc{getInt}
     *
     * Description:
     *   Return an integer representation of the hash
     *
     *   ?> Note that this does not fully represent the hash.
     *   ?> .
     *   ?> There isn't a way to recreate the hash from just this `int` value
     *
     * @endDoc
     */
    int getInt() const;

    /**
     * @startDoc{getString[0]}
     *
     * Description:
     *   Return a short string representation of the hash
     *
     *   ?> Note that this does not fully represent the hash.
     *   ?> .
     *   ?> There isn't a way to recreate the hash from just this `std::string` value
     *
     * @endDoc
     */
    std::string getString() const;

    /**
     * @doc{getString[1]}
     */
    operator std::string () const;

    /**
     * @startDoc{getFullString}
     *
     * Description:
     *   Return the full string representation of the hash.
     *
     *   Use the [[hash_t.fromString]] method to get the [[hash_t]] object back.
     *
     * @endDoc
     */
    std::string getFullString() const;

    /**
     * @startDoc{fromString}
     *
     * Description:
     *   Given the full string representation of the hash ([[hash_t.getFullString]]),
     *   get the original [[hash_t]] object back.
     *
     * @endDoc
     */
    static hash_t fromString(const std::string &s);

    static hash_t random();

    friend std::ostream& operator << (std::ostream &out,
                                    const hash_t &hash);
  };
  std::ostream& operator << (std::ostream &out,
                           const hash_t &hash);

  hash_t hash(const void *ptr, udim_t bytes);

  template <class T>
  inline hash_t hash(const std::vector<T> &vec) {
    hash_t h;
    for (const T &value : vec) {
      h ^= hash(value);
    }
    return h;
  }

  template <class T>
  inline hash_t hash(const T &t) {
    return hash(&t, sizeof(T));
  }

  template <class T>
  inline hash_t hash_t::operator ^ (const T &t) const {
    return (*this ^ hash(t));
  }

  template <>
  hash_t hash_t::operator ^ (const hash_t &hash) const;

  hash_t hash(const char *c);
  hash_t hash(const std::string &str);
  hash_t hashFile(const std::string &filename);
}

#endif
