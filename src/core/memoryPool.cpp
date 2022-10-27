#include <occa/core/base.hpp>
#include <occa/core/memory.hpp>
#include <occa/core/device.hpp>
#include <occa/internal/core/device.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/core/memoryPool.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  memoryPool::memoryPool() :
      modeMemoryPool(NULL) {}

  memoryPool::memoryPool(modeMemoryPool_t *modeMemoryPool_) :
      modeMemoryPool(NULL) {
    setModeMemoryPool(modeMemoryPool_);
  }

  memoryPool::memoryPool(const memoryPool &m) :
    modeMemoryPool(NULL) {
    setModeMemoryPool(m.modeMemoryPool);
  }

  memoryPool& memoryPool::operator = (const memoryPool &m) {
    setModeMemoryPool(m.modeMemoryPool);
    return *this;
  }

  memoryPool::~memoryPool() {
    removeMemoryPoolRef();
  }

  void memoryPool::assertInitialized() const {
    OCCA_ERROR("MemoryPool not initialized or has been freed",
               modeMemoryPool != NULL);
  }

  void memoryPool::setModeMemoryPool(modeMemoryPool_t *modeMemoryPool_) {
    if (modeMemoryPool != modeMemoryPool_) {
      removeMemoryPoolRef();
      modeMemoryPool = modeMemoryPool_;
      if (modeMemoryPool) {
        modeMemoryPool->addMemoryPoolRef(this);
      }
    }
  }

  void memoryPool::removeMemoryPoolRef() {
    if (!modeMemoryPool) {
      return;
    }
    modeMemoryPool->removeMemoryPoolRef(this);
    if (modeMemoryPool->modeMemoryPool_t::needsFree()) {
      delete modeMemoryPool;
      modeMemoryPool = NULL;
    }
  }

  void memoryPool::dontUseRefs() {
    if (modeMemoryPool) {
      modeMemoryPool->modeMemoryPool_t::dontUseRefs();
    }
  }

  bool memoryPool::isInitialized() const {
    return (modeMemoryPool != NULL);
  }

  memoryPool& memoryPool::swap(memoryPool &m) {
    modeMemoryPool_t *modeMemoryPool_ = modeMemoryPool;
    modeMemoryPool   = m.modeMemoryPool;
    m.modeMemoryPool = modeMemoryPool_;
    return *this;
  }

  modeMemoryPool_t* memoryPool::getModeMemoryPool() const {
    return modeMemoryPool;
  }

  modeDevice_t* memoryPool::getModeDevice() const {
    return (modeMemoryPool
            ? modeMemoryPool->modeDevice
            : nullptr);
  }

  occa::device memoryPool::getDevice() const {
    return occa::device(modeMemoryPool
                        ? modeMemoryPool->modeDevice
                        : NULL);
  }

  const std::string& memoryPool::mode() const {
    static const std::string noMode = "No Mode";
    return (modeMemoryPool
            ? modeMemoryPool->modeDevice->mode
            : noMode);
  }

  const occa::json& memoryPool::properties() const {
    static const occa::json noProperties;
    return (modeMemoryPool
            ? modeMemoryPool->properties
            : noProperties);
  }

  udim_t memoryPool::size() const {
    if (modeMemoryPool == NULL) {
      return 0;
    }
    return modeMemoryPool->size;
  }

  udim_t memoryPool::reserved() const {
    if (modeMemoryPool == NULL) {
      return 0;
    }
    return modeMemoryPool->reserved;
  }

  udim_t memoryPool::numReservations() const {
    if (modeMemoryPool == NULL) {
      return 0;
    }
    return modeMemoryPool->numReservations();
  }

  udim_t memoryPool::alignment() const {
    if (modeMemoryPool == NULL) {
      return 0;
    }
    return modeMemoryPool->alignment;
  }

  bool memoryPool::operator == (const occa::memoryPool &other) const {
    return (modeMemoryPool == other.modeMemoryPool);
  }

  bool memoryPool::operator != (const occa::memoryPool &other) const {
    return (modeMemoryPool != other.modeMemoryPool);
  }

  void memoryPool::resize(const udim_t bytes) {
    assertInitialized();
    modeMemoryPool->resize(bytes);
  }

  void memoryPool::shrinkToFit() {
    resize(reserved());
  }

  void memoryPool::free() {
    if (modeMemoryPool == NULL) return;
    delete modeMemoryPool;
  }

  memory memoryPool::reserve(const dim_t entries,
                             const dtype_t &dtype) {
    assertInitialized();

    if (entries == 0) {
      return memory();
    }

    const dim_t bytes = entries * dtype.bytes();
    OCCA_ERROR("Trying to reserve negative bytes (" << bytes << ")",
               bytes >= 0);

    memory mem(modeMemoryPool->reserve(bytes));
    mem.setDtype(dtype);

    return mem;
  }

  template <>
  memory memoryPool::reserve<void>(const dim_t entries) {
    return reserve(entries, dtype::byte);
  }

  void memoryPool::setAlignment(const udim_t alignment) {
    assertInitialized();
    modeMemoryPool->setAlignment(alignment);
  }
}
