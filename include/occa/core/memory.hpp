#ifndef OCCA_CORE_MEMORY_HEADER
#define OCCA_CORE_MEMORY_HEADER

#include <iostream>

#include <occa/defines.hpp>
#include <occa/dtype.hpp>
#include <occa/types.hpp>

// Unfortunately we need to expose this in include
#include <occa/utils/gc.hpp>

namespace occa {
  class modeMemory_t; class memory;
  class modeDevice_t; class device;
  class kernelArg;

  typedef std::map<hash_t,occa::memory>   hashedMemoryMap;
  typedef hashedMemoryMap::iterator       hashedMemoryMapIterator;
  typedef hashedMemoryMap::const_iterator cHashedMemoryMapIterator;

  /**
   * @startDoc{memory}
   *
   * Description:
   *
   *   A [[memory]] object is a handle to memory allocated by a device.
   *   For example, in `Serial` and `OpenMP` modes it is analogous to a `void*` pointer that comes out of `malloc` or `new`.
   *   Check [[device.malloc]] for more information about how to allocate memory and build a memory object.
   *
   *   # Data transfer
   *
   *   There are 2 helper methods to help with data transfer:
   *   - [[memory.copyTo]] which helpes copy data from the memory object to the input.
   *   - [[memory.copyFrom]] which helpes copy data from the input to the memory object.
   *
   *   > Note that because we know the type and size of the underlying data allocated, passing the bytes to copy defaults to the full array.
   *
   *   # Transformations
   *
   *   ## Slices
   *
   *   Sometimes we want to pass a subsection of the memory to a kernel.
   *   Rather than passing the memory and the offset to the kernel, we support slicing the memory object through [[memory.slice]].
   *   The returned memory object will be a reference to the original but will keep track of the offset and size change.
   *
   *   ## Cloning
   *
   *   The [[memory.clone]] method is a quick way to create a copy of a memory object.
   *
   *   ## Casting
   *
   *   Calling [[memory.cast]] will return a reference to the original memory object but with a different type.
   *   This can be used to assert type at runtime when passed to kernel as arguments.
   *
   *   # Garbage collection
   *
   *   The [[memory.free]] function can be called to free the memory.
   *   OCCA implemented reference counting by default so calling [[memory.free]] is not required.
   *
   * @endDoc
   */
  class memory : public gc::ringEntry_t {
    friend class occa::modeMemory_t;
    friend class occa::device;
    friend class occa::kernelArg;

  private:
    modeMemory_t *modeMemory;

  public:
    /**
     * @startDoc{constructor}
     *
     * Description:
     *   TODO
     *
     * @endDoc
     */
    memory();
    memory(void *uvaPtr);
    memory(modeMemory_t *modeMemory_);

    memory(const memory &m);
    memory& operator = (const memory &m);
    ~memory();

  private:
    void assertInitialized() const;
    void setModeMemory(modeMemory_t *modeMemory_);
    void removeMemoryRef();

  public:
    void dontUseRefs();

    bool isInitialized() const;

    memory& swap(memory &m);

    /**
     * @startDoc{ptr}
     *
     * TODO
     *
     * @endDoc
     */
    template <class T = void>
    T* ptr();

    template <class T = void>
    const T* ptr() const;

    modeMemory_t* getModeMemory() const;
    modeDevice_t* getModeDevice() const;

    occa::device getDevice() const;

    operator kernelArg() const;

    const std::string& mode() const;
    const occa::json& properties() const;

    void setDtype(const dtype_t &dtype__);

    const dtype_t& dtype() const;

    udim_t size() const;
    udim_t length() const;

    template <class T>
    udim_t length() const {
      return size() / sizeof(T);
    }

    //---[ UVA ]------------------------
    bool isManaged() const;
    bool inDevice() const;
    bool isStale() const;

    void setupUva();
    void startManaging();
    void stopManaging();

    void syncToDevice(const dim_t bytes, const dim_t offset);
    void syncToHost(const dim_t bytes, const dim_t offset);

    bool uvaIsStale() const;
    void uvaMarkStale();
    void uvaMarkFresh();
    //==================================

    bool operator == (const occa::memory &other) const;
    bool operator != (const occa::memory &other) const;

    occa::memory operator + (const dim_t offset) const;
    occa::memory& operator += (const dim_t offset);

    /**
     * @startDoc{slice}
     *
     * Description:
     *   TODO
     *
     * @endDoc
     */
    occa::memory slice(const dim_t offset,
                       const dim_t count = -1) const;

    /**
     * @startDoc{copyFrom}
     *
     * Description:
     *   TODO
     *
     * @endDoc
     */
    void copyFrom(const void *src,
                  const dim_t bytes = -1,
                  const dim_t offset = 0,
                  const occa::json &props = occa::json());

    void copyFrom(const memory src,
                  const dim_t bytes = -1,
                  const dim_t destOffset = 0,
                  const dim_t srcOffset = 0,
                  const occa::json &props = occa::json());

    /**
     * @startDoc{copyTo}
     *
     * Description:
     *   TODO
     *
     * @endDoc
     */
    void copyTo(void *dest,
                const dim_t bytes = -1,
                const dim_t offset = 0,
                const occa::json &props = occa::json()) const;

    void copyTo(const memory dest,
                const dim_t bytes = -1,
                const dim_t destOffset = 0,
                const dim_t srcOffset = 0,
                const occa::json &props = occa::json()) const;

    void copyFrom(const void *src,
                  const occa::json &props);

    void copyFrom(const memory src,
                  const occa::json &props);

    void copyTo(void *dest,
                const occa::json &props) const;

    void copyTo(const memory dest,
                const occa::json &props) const;

    /**
     * @startDoc{cast}
     *
     * Description:
     *   TODO
     *
     * @endDoc
     */
    occa::memory cast(const dtype_t &dtype_) const;

    /**
     * @startDoc{clone}
     *
     * Description:
     *   TODO
     *
     * @endDoc
     */
    occa::memory clone() const;

    /**
     * @startDoc{free}
     *
     * Description:
     *   TODO
     *
     * @endDoc
     */
    void free();

    void detach();

    void deleteRefs(const bool freeMemory = false);
  };

  extern memory null;

  std::ostream& operator << (std::ostream &out,
                           const occa::memory &memory);

  template <>
  void* memory::ptr<void>();

  template <>
  const void* memory::ptr<void>() const;
}

#include "memory.tpp"

#endif
