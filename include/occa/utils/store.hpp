#ifndef OCCA_UTILS_STORE_HEADER
#define OCCA_UTILS_STORE_HEADER

#include <cmath>
#include <map>
#include <memory>

#include <occa/utils/hash.hpp>
#include <occa/utils/mutex.hpp>

#define OCCA_ATOMIC_STORE_HASH_SIZE 128

namespace occa {
  template <class HashableKeyType, class ValueType>
  class store_t {
  private:
    using ValueSharedPtr = std::shared_ptr<ValueType>;
    using storeMapType = std::map<HashableKeyType, ValueSharedPtr>;

    storeMapType store;
    mutex_t *mutexList;

    inline mutex_t& getMutexForKey(const HashableKeyType &key) {
      return mutexList[
        std::abs(hash(key).getInt()) % OCCA_ATOMIC_STORE_HASH_SIZE
      ];
    }

  public:
    store_t() :
      mutexList(new mutex_t[OCCA_ATOMIC_STORE_HASH_SIZE]) {}

    ~store_t() {
      delete [] mutexList;
    }

    void lock(const HashableKeyType &key) {
      getMutexForKey(key).lock();
    }

    void unlock(const HashableKeyType &key) {
      getMutexForKey(key).unlock();
    }

    bool has(const HashableKeyType &key) const {
      lock(key);
      const bool hasKey = (
        store.find(key) != store.end()
      );
      unlock(key);

      return hasKey;
    }

    size_t size() const {
      mutex_t &mutex = mutexList[0];
      size_t mapSize;

      mutex.lock();
      mapSize = store.size();
      mutex.unlock();

      return mapSize;
    }

    ValueSharedPtr unsafeGet(const HashableKeyType &key) {
      typename storeMapType::iterator it = store.find(key);
      if (it != store.end()) {
        return it->second;
      }
      return nullptr;
    }

    ValueSharedPtr get(const HashableKeyType &key) {
      ValueSharedPtr valuePtr;

      lock(key);
      valuePtr = unsafeGet(key);
      unlock(key);

      return valuePtr;
    }

    bool unsafeGetOrCreate(const HashableKeyType &key, ValueSharedPtr &valuePtr) {
      typename storeMapType::iterator it = store.find(key);

      if (it != store.end()) {
        valuePtr = it->second;
        return false;
      }

      valuePtr = std::make_shared<ValueType>();
      store[key] = valuePtr;

      return true;
    }
  };
}

#endif
