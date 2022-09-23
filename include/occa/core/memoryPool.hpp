#ifndef OCCA_CORE_MEMORYPOOL_HEADER
#define OCCA_CORE_MEMORYPOOL_HEADER

#include <occa/core/memory.hpp>

namespace occa {
  class modeBuffer_t; class buffer;
  class modeMemory_t; class memory;
  class modeDevice_t; class device;
  class modeMemoryPool_t;

  namespace experimental {

  class memoryPool : public gc::ringEntry_t {
    friend class occa::modeMemoryPool_t;

  private:
    modeMemoryPool_t *modeMemoryPool;

  public:
    memoryPool();
    memoryPool(modeMemoryPool_t *modeMemoryPool_);

    memoryPool(const memoryPool &m);
    memoryPool& operator = (const memoryPool &m);
    ~memoryPool();

  private:
    void assertInitialized() const;
    void setModeMemoryPool(modeMemoryPool_t *modeMemoryPool_);
    void removeMemoryPoolRef();

  public:
    void dontUseRefs();

    /**
     * @startDoc{isInitialized}
     *
     * Description:
     *   Check whether the [[memoryPool]] has been intialized.
     *
     * Returns:
     *   Returns `true` if the [[memoryPool]] has been built successfully
     *
     * @endDoc
     */
    bool isInitialized() const;

    memoryPool& swap(memoryPool &m);

    modeMemoryPool_t* getModeMemoryPool() const;
    modeDevice_t* getModeDevice() const;

    /**
     * @startDoc{getDevice}
     *
     * Description:
     *   Returns the [[device]] used to build the [[memoryPool]].
     *
     * Returns:
     *   The [[device]] used to build the [[memoryPool]]
     *
     * @endDoc
     */
    occa::device getDevice() const;

    /**
     * @startDoc{mode}
     *
     * Description:
     *   Returns the mode of the [[device]] used to build the [[memoryPool]].
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
     *   Get the properties used to build the [[memoryPool]].
     *
     * Description:
     *   Returns the properties used to build the [[memoryPool]].
     *
     * @endDoc
     */
    const occa::json& properties() const;

    /**
     * @startDoc{size}
     *
     * Description:
     *   Get the byte size of the allocated memoryPool
     *
     * @endDoc
     */
    udim_t size() const;

    /**
     * @startDoc{reserved}
     *
     * Description:
     *   Get the byte size of the memoryPool currently reserved
     *
     * @endDoc
     */
    udim_t reserved() const;

    /**
     * @startDoc{numReservations}
     *
     * Description:
     *   Get the number of currently active reservations in the memoryPool
     *
     * @endDoc
     */
    udim_t numReservations() const;

    /**
     * @startDoc{alignment}
     *
     * Description:
     *   Get the byte size of the memoryPool alignment
     *
     * @endDoc
     */
    udim_t alignment() const;

    /**
     * @startDoc{operator_equals[0]}
     *
     * Description:
     *   Compare if two memoryPool objects have the same references.
     *
     * Returns:
     *   If the references are the same, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator == (const occa::experimental::memoryPool &other) const;

    /**
     * @startDoc{operator_equals[1]}
     *
     * Description:
     *   Compare if two memoryPool objects have different references.
     *
     * Returns:
     *   If the references are different, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator != (const occa::experimental::memoryPool &other) const;

    /**
     * @startDoc{resize}
     *
     * Description:
     *   Resize the underlying device memory buffer in the memoryPool.
     *   An error will be thrown if the currently reserved space in the memoryPool
     *   is larger than bytes.
     *
     * Arguments:
     *   bytes:
     *     The size of the memory pool to allocate.
     * @endDoc
     */
    void resize(const udim_t bytes);

    /**
     * @startDoc{shrinkToFit}
     *
     * Description:
     *   Resize the underlying device memory buffer in the memoryPool to fit only
     *   The currently active reservations.
     *
     * @endDoc
     */
    void shrinkToFit();

    /**
     * @startDoc{free}
     *
     * Description:
     *   Free the device memoryPool.
     *   Calling [[memoryPool.isInitialized]] will return `false` now.
     *
     * @endDoc
     */
    void free();

    //  |===============================

    //  |---[ Memory ]------------------
    /**
     * @startDoc{reserve[0]}
     *
     * Description:
     *   Reserves memory on the device from memory pool and returns the [[memory]] handle.
     *
     * Overloaded Description:
     *   Uses the templated type to determine the type and bytes.
     *
     * Arguments:
     *   entries:
     *     The length of the allocated memory
     *
     * Returns:
     *   The reserved [[memory]]
     *
     * @endDoc
     */
    template <class T = void>
    occa::memory reserve(const dim_t entries);

    /**
     * @startDoc{reserve[1]}
     *
     * Overloaded Description:
     *   Same but takes a [[dtype_t]] rather than a template parameter.
     *
     * Arguments:
     *   entries:
     *     The length of the allocated memory
     *   dtype:
     *     The [[dtype_t]] of what will be allocated, which defines the length of each entry
     *
     * Returns:
     *   The reserved [[memory]]
     *
     * @endDoc
     */
    occa::memory reserve(const dim_t entries,
                         const dtype_t &dtype);\

    /**
     * @startDoc{setAlignment}
     *
     * Description:
     *   Set the buffer aligment of the memoryPool.
     *   May trigger a re-allocation of the memory pool if there
     *   are currently active reservations
     *
     * Arguments:
     *   alignment:
     *     The size of the alignment in bytes.
     * @endDoc
     */
    void setAlignment(const udim_t alignment);
  };

  }
}

#include "memoryPool.tpp"

#endif
