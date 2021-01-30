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

    /**
     * @startDoc{isInitialized}
     *
     * Description:
     *   Check whether the [[memory]] has been intialized.
     *
     * Returns:
     *   Returns `true` if the [[memory]] has been built successfully
     *
     * @endDoc
     */
    bool isInitialized() const;

    memory& swap(memory &m);

    /**
     * @startDoc{ptr[0]}
     *
     * Description:
     *   Return the backend pointer
     *
     *   - _Serial_, _OpenMP_: Host pointer, which can be used in the host application
     *   - _CUDA_, _HIP_: Allocated device pointer. If allocated with the `host: true` flag it will return the host pointer
     *   - _OpenCL_: `cl_mem` pointer
     *   - _Metal_: Metal buffer pointer
     *
     * @endDoc
     */
    template <class T = void>
    T* ptr();

    /**
     * @doc{ptr[1]}
     */
    template <class T = void>
    const T* ptr() const;

    modeMemory_t* getModeMemory() const;
    modeDevice_t* getModeDevice() const;

    /**
     * @startDoc{getDevice}
     *
     * Description:
     *   Returns the [[device]] used to build the [[memory]].
     *
     * Returns:
     *   The [[device]] used to build the [[memory]]
     *
     * @endDoc
     */
    occa::device getDevice() const;

    /**
     * @startDoc{operator_kernelArg}
     *
     * Description:
     *   Casts to [[kernelArg]] for it to be taken as a [[kernel]] argument
     *
     * Returns:
     *   The [[kernelArg]]
     *
     * @endDoc
     */
    operator kernelArg() const;

    /**
     * @startDoc{mode}
     *
     * Description:
     *   Returns the mode of the [[device]] used to build the [[memory]].
     *
     * Returns:
     *   The `mode` string, such as `"Serial"`, `"CUDA"`, or `"HIP"`.
     *
     * @endDoc
     */
    const std::string& mode() const;

    /**
     * @startDoc{properties}
     *
     * Description:
     *   Get the properties used to build the [[memory]].
     *
     * Description:
     *   Returns the properties used to build the [[memory]].
     *
     * @endDoc
     */
    const occa::json& properties() const;

    void setDtype(const dtype_t &dtype__);

    /**
     * @startDoc{dtype}
     *
     * Description:
     *   Get the [[dtype_t]] from when the [[memory]] was allocated or [[casted manually|memory.cast]]
     *
     * Description:
     *   Returns the [[dtype_t]]
     *
     * @endDoc
     */
    const dtype_t& dtype() const;

    /**
     * @startDoc{size}
     *
     * Description:
     *   Get the byte size of the allocated memory
     *
     * @endDoc
     */
    udim_t size() const;

    /**
     * @startDoc{length[0]}
     *
     * Description:
     *   Get the length of the memory object, using its underlying [[dtype_t]].
     *   This [[dtype_t]] can be fetched through the [[memory.dtype]] method
     *
     *   If no type was given during [[allocation|device.malloc]] or was ever set
     *   through [[casting it|memory.cast]], it will return the bytes just like [[memory.size]].
     *
     * @endDoc
     */
    udim_t length() const;

    /**
     * @startDoc{length[1]}
     *
     * Overloaded Description:
     *   Same as above but explicitly chose the type (`T`)
     *
     * @endDoc
     */
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

    /**
     * @startDoc{operator_equals[0]}
     *
     * Description:
     *   Compare if two memory objects have the same references.
     *
     * Returns:
     *   If the references are the same, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator == (const occa::memory &other) const;

    /**
     * @startDoc{operator_equals[1]}
     *
     * Description:
     *   Compare if two memory objects have different references.
     *
     * Returns:
     *   If the references are different, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator != (const occa::memory &other) const;

    /**
     * @startDoc{operator_add[0]}
     *
     * Description:
     *   Same as calling [[memory.slice]]`(offset)`
     *
     * Returns:
     *   A [[memory]] object shifted by `offset` bytes
     *
     * @endDoc
     */
    occa::memory operator + (const dim_t offset) const;

    /**
     * @doc{operator_add[1]}
     */
    occa::memory& operator += (const dim_t offset);

    /**
     * @startDoc{slice}
     *
     * Description:
     *   Returns a [[memory]] object with the same reference as the caller,
     *   but has its start and end pointer values shifted.
     *
     *   For example:
     *
     *   ```cpp
     *   // mem = {?, ?, ?, ?}
     *   occa::memory mem = device.malloc<float>(4);
     *
     *   occa::memory firstHalf = mem.slice(0, 2);
     *   occa::memory lastHalf = mem.slice(2, 4); // Or just mem.slice(2)
     *
     *   int values[4] = {1, 2, 3, 4}
     *
     *   // mem = {1, 2, ?, ?}
     *   firstHalf.copyFrom(values);
     *
     *   // mem = {1, 2, 3, 4}
     *   secondtHalf.copyFrom(values + 2);
     *   ```
     *
     * @endDoc
     */
    occa::memory slice(const dim_t offset,
                       const dim_t count = -1) const;

    /**
     * @startDoc{copyFrom[0]}
     *
     * Description:
     *   Copies data from the input `src` to the caller [[memory]] object
     *
     * Arguments:
     *   src:
     *     Data source.
     *
     *   bytes:
     *     How many bytes to copy.
     *
     *   offset:
     *     The [[memory]] offset where data transfer will start.
     *
     *   props:
     *     Any backend-specific properties for memory transfer.
     *     For example, `async: true`.
     *
     * @endDoc
     */
    void copyFrom(const void *src,
                  const dim_t bytes = -1,
                  const dim_t offset = 0,
                  const occa::json &props = occa::json());

    /**
     * @doc{copyFrom[1]}
     */
    void copyFrom(const void *src,
                  const occa::json &props);

    /**
     * @startDoc{copyFrom[2]}
     *
     * Description:
     *   Same as above, but uses a [[memory]] source
     *
     * Arguments:
     *   destOffset:
     *     The [[memory]] offset for the caller [[memory]]
     *
     *   srcOffset:
     *     The [[memory]] offset for the source [[memory]] (`src`)
     *
     * @endDoc
     */
    void copyFrom(const memory src,
                  const dim_t bytes = -1,
                  const dim_t destOffset = 0,
                  const dim_t srcOffset = 0,
                  const occa::json &props = occa::json());

    /**
     * @doc{copyFrom[3]}
     */
    void copyFrom(const memory src,
                  const occa::json &props);

    /**
     * @startDoc{copyTo[0]}
     *
     * Description:
     *   Copies data from the input `src` to the caller [[memory]] object
     *
     * Arguments:
     *   dest:
     *     Where to copy the [[memory]] data to.
     *
     *   bytes:
     *     How many bytes to copy
     *
     *   offset:
     *     The [[memory]] offset where data transfer will start.
     *
     *   props:
     *     Any backend-specific properties for memory transfer.
     *     For example, `async: true`.
     *
     * @endDoc
     */
    void copyTo(void *dest,
                const dim_t bytes = -1,
                const dim_t offset = 0,
                const occa::json &props = occa::json()) const;

    /**
     * @doc{copyTo[1]}
     */
    void copyTo(void *dest,
                const occa::json &props) const;

    /**
     * @startDoc{copyTo[2]}
     *
     * Description:
     *   Same as above, but uses a [[memory]] source
     *
     * Arguments:
     *   destOffset:
     *     The [[memory]] offset for the destination [[memory]] (`dest`)
     *
     *   srcOffset:
     *     The [[memory]] offset for the caller [[memory]]
     *
     * @endDoc
     */
    void copyTo(const memory dest,
                const dim_t bytes = -1,
                const dim_t destOffset = 0,
                const dim_t srcOffset = 0,
                const occa::json &props = occa::json()) const;

    /**
     * @doc{copyTo[3]}
     */
    void copyTo(const memory dest,
                const occa::json &props) const;

    /**
     * @startDoc{cast}
     *
     * Description:
     *   Return a reference to the caller [[memory]] object but
     *   with a different data type.
     *
     * Arguments:
     *   dtype_:
     *     What the return [[memory]]'s data type should be
     *
     * @endDoc
     */
    occa::memory cast(const dtype_t &dtype_) const;

    /**
     * @startDoc{clone}
     *
     * Description:
     *   Allocate a new [[memory]] object with the same data copied to it
     *
     * @endDoc
     */
    occa::memory clone() const;

    /**
     * @startDoc{free}
     *
     * Description:
     *   Free the device memory.
     *   Calling [[memory.isInitialized]] will return `false` now.
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
