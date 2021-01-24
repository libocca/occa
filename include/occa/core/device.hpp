#ifndef OCCA_CORE_DEVICE_HEADER
#define OCCA_CORE_DEVICE_HEADER

#include <iostream>
#include <sstream>

#include <occa/core/kernel.hpp>
#include <occa/core/memory.hpp>
#include <occa/core/stream.hpp>
#include <occa/defines.hpp>
#include <occa/dtype.hpp>
#include <occa/types.hpp>

// Unfortunately we need to expose this in include
#include <occa/utils/gc.hpp>

namespace occa {
  class modeKernel_t; class kernel;
  class modeDevice_t; class device;
  class modeStreamTag_t; class streamTag;
  class deviceInfo;

  typedef std::map<std::string, kernel>   cachedKernelMap;
  typedef cachedKernelMap::iterator       cachedKernelMapIterator;
  typedef cachedKernelMap::const_iterator cCachedKernelMapIterator;

  /**
   * @id{device}
   *
   * @descriptionStart
   *
   * # Description
   *
   * A [[device]] object maps to a physical device we want to program on.
   * Examples include a CPU, GPU, or other physical accelerator like an FPGA.
   *
   * There are 2 main uses of a device:
   * - Memory allocation ([[memory]])
   * - Compile and run code ([[kernel]])
   *
   * # Setup
   *
   * Setting up [[device]] objects is done through JSON properties.
   * Here's an example of a CUDA device picking device `0` through its [[device.constructor]]:
   *
   * ```cpp
   * occa::device device({
   *   {"mode", "CUDA"},
   *   {"device_id", 0}
   * })
   * ```
   *
   * JSON formatted strings can also be passed directly, which can be useful when loading from a config file.
   *
   * ```cpp
   * occa::device device(
   *   "{ mode: 'CUDA", device_id: 0 }"
   * );
   * ```
   *
   * We can achieve the same using the [[device.setup]] method which take similar arguments as the constructors.
   * For example:
   *
   * ```cpp
   * occa::device device;
   * // ...
   * device.setup({
   *   {"mode", "CUDA"},
   *   {"device_id", 0}
   * })
   * ```
   *
   * # Memory allocation
   *
   * We suggest allocating through the templated [[device.malloc]] method which will keep type information around.
   * Here's an example which allocates memory on the device to fit a `float` array of size 10:
   *
   * ```cpp
   * occa::memory mem = device.malloc<float>(10);
   * ```
   *
   * # Kernel compilation
   *
   * Kernel allocation can be done two ways, through [[a file|device.buildKernel]]) or [[string source|device.buildKernelFromString]].
   * Here's an example which builds a [[kernel]] from a file:
   *
   * ```cpp
   * occa::kernel addVectors = (
   *   device.buildKernel("addVectors.okl",
   *                      "addVectors")
   * );
   * ```
   *
   * # Interoperability
   *
   * Lastly, we allow for interoperability with supported backends/libraries by wrapping and unwrapping memory objects.
   *
   * Here's an example which takes a native pointer and wraps it as a [[memory]] object through the [[device.wrapMemory]] method:
   *
   * ```cpp
   * occa::memory occaPtr = (
   *   device.wrapMemory<float>(ptr, 10)
   * );
   * ```
   *
   * # Garbage collection
   *
   * The [[device.free]] function can be called to free the device along with any other object allocated by it, such as [[memory]] and [[kernel] objects.
   * OCCA implemented reference counting by default so calling [[device.free]] is not required.   *
   *
   * @descriptionEnd
   */
  class device : public gc::ringEntry_t {
    friend class modeDevice_t;
    friend class kernel;
    friend class memory;

  private:
    mutable modeDevice_t *modeDevice;

  public:
    /**
     * @id{constructor}
     *
     * @descriptionStart
     *
     * Creates a handle to a physical device we want to program on, such as a CPU, GPU, or other accelerator.
     *
     * @descriptionEnd
     *
     * @instanceDescriptionStart
     *
     * Default constructor
     *
     * @instanceDescriptionEnd
     */
    device();

    /**
     * @alias{constructor}
     *
     * @instanceDescriptionStart
     *
     * Takes a JSON-formatted string for the device props.
     *
     * @instanceDescriptionEnd
     */
    device(const std::string &props);

    /**
     * @alias{constructor}
     *
     * @instanceDescriptionStart
     *
     * Takes an [[json]] argument for the device props.
     *
     * @instanceDescriptionEnd
     */
    device(const occa::json &props);

    device(modeDevice_t *modeDevice_);

    device(jsonInitializerList initializer);

    device(const occa::device &other);

    device& operator = (const occa::device &other);

    ~device();

  private:
    void assertInitialized() const;
    void setModeDevice(modeDevice_t *modeDevice_);
    void removeDeviceRef();

  public:
    /**
     * @id{dontUseRefs}
     *
     * @descriptionStart
     *
     * By default, a [device] will automatically call [[device.free]] through reference counting.
     * Turn off automatic garbage collection through this method.
     *
     * @descriptionEnd
     */
    void dontUseRefs();

    /**
     * @id{==}
     *
     * @descriptionStart
     *
     * Comparison operators
     *
     * @descriptionEnd
     */
    bool operator == (const occa::device &other) const;

    /**
     * @alias{==}
     */
    bool operator != (const occa::device &other) const;

    /**
     * @id{isInitialized}
     *
     * @descriptionStart
     *
     * Returns `true` if the device has been intialized, through either the [[device.constructor]] or [[device.setup]].
     *
     * @descriptionEnd
     */
    bool isInitialized();

    modeDevice_t* getModeDevice() const;

    /**
     * @id{setup}
     *
     * @descriptionStart
     *
     * Similar to [[device.constructor]] but can be called after creating the initial [[device]] object.
     *
     * @descriptionEnd
     */
    void setup(const std::string &props);

    /**
     * @alias{setup}
     */
    void setup(const occa::json &props);

    /**
     * @id{free}
     *
     * @descriptionStart
     *
     * Free the device, which will also free:
     * - Allocated [[memory]]
     * - Built [[kernel]]
     * - Created [[stream]] and [[streamTag]]
     *
     * @descriptionEnd
     */
    void free();

    /**
     * @id{mode}
     *
     * @descriptionStart
     *
     * Returns the device `mode`, such as `"CUDA"` or `"HIP"`.
     *
     * @descriptionEnd
     */
    const std::string& mode() const;

    /**
     * @id{properties}
     *
     * @descriptionStart
     *
     * Returns the properties used to build the [[device]].
     *
     * @descriptionEnd
     */
    const occa::json& properties() const;

    const occa::json& kernelProperties() const;
    occa::json kernelProperties(const occa::json &additionalProps) const;

    const occa::json& memoryProperties() const;
    occa::json memoryProperties(const occa::json &additionalProps) const;

    const occa::json& streamProperties() const;
    occa::json streamProperties(const occa::json &additionalProps) const;

    /**
     * @id{hash}
     *
     * @descriptionStart
     *
     * Two devices should have the same [[hash|hash_t]] if they point to the same hardware device and setup with the same properties.
     *
     * @descriptionEnd
     */
    hash_t hash() const;

    /**
     * @id{memorySize}
     *
     * @descriptionStart
     *
     * Returns the memory size of the device, not accounting for allocated memory.
     *
     * @descriptionEnd
     */
    udim_t memorySize() const;

    /**
     * @id{memoryAllocated}
     *
     * @descriptionStart
     *
     * Returns the memory allocated using just this [[device]].
     *
     * @descriptionEnd
     */
    udim_t memoryAllocated() const;

    /**
     * @id{finish}
     *
     * @descriptionStart
     *
     * Finishes any asynchronous operation queued up on the device, such as [[async memory allocations|malloc]] or [[kernel calls|kernel.()]].
     *
     * @descriptionEnd
     */
    void finish();

    /**
     * @id{hasSeparateMemorySpace}
     *
     * @descriptionStart
     *
     * Returns `true` if the memory is directly accesible through the host.
     *
     * @descriptionEnd
     */
    bool hasSeparateMemorySpace();

    //  |---[ Stream ]------------------
    /**
     * @id{createStream}
     *
     * @descriptionStart
     *
     * Creates a new [[steam]] to queue operations on.
     * If the backend supports streams, out-of-order work can be achieved through the use of streams.
     *
     * @descriptionEnd
     */
    stream createStream(const occa::json &props = occa::json());

    /**
     * @id{getStream}
     *
     * @descriptionStart
     *
     * Returns the active [[stream]].
     *
     * @descriptionEnd
     */
    stream getStream();

    /**
     * @id{setStream}
     *
     * @descriptionStart
     *
     * Sets the active [[stream]].
     *
     * @descriptionEnd
     */
    void setStream(stream s);

    /**
     * @id{tagStream}
     *
     * @descriptionStart
     *
     * Tag the stream and returns the created [[streamTag]].
     *
     * @descriptionEnd
     */
    streamTag tagStream();

    /**
     * @id{waitFor}
     *
     * @descriptionStart
     *
     * Wait for all operations queued up until the [[tag|streamTag]].
     *
     * @descriptionEnd
     */
    void waitFor(streamTag tag);

    /**
     * @id{timeBetween}
     *
     * @descriptionStart
     *
     * Returns the time taken in seconds between the two [[tags|streamTag]].
     *
     * @descriptionEnd
     */
    double timeBetween(const streamTag &startTag,
                       const streamTag &endTag);
    //  |===============================

    //  |---[ Kernel ]------------------
    void setupKernelInfo(const occa::json &props,
                         const hash_t &sourceHash,
                         occa::json &kernelProps,
                         hash_t &kernelHash) const;

    hash_t applyDependencyHash(const hash_t &kernelHash) const;

    /**
     * @id{buildKernel}
     *
     * @descriptionStart
     *
     * Builds a [[kernel]] given a filename, kernel name, and optional properties
     *
     * defines
     * includes
     * headers
     * functions
     *
     * @descriptionEnd
     */
    occa::kernel buildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::json &props = occa::json()) const;

    /**
     * @id{buildkernelFromString}
     *
     * @descriptionStart
     *
     * Same as [[device.buildKernel]] but given the kernel source code rather than the filename.
     *
     * @descriptionEnd
     */
    occa::kernel buildKernelFromString(const std::string &content,
                                       const std::string &kernelName,
                                       const occa::json &props = occa::json()) const;

    occa::kernel buildKernelFromBinary(const std::string &filename,
                                       const std::string &kernelName,
                                       const occa::json &props = occa::json()) const;
    //  |===============================

    //  |---[ Memory ]------------------
    /**
     * @id{malloc}
     *
     * @descriptionStart
     *
     * Allocates memory on the device and returns the [[memory]] handle.
     * If a `src` pointer is passed, its data will be automatically copied to the allocated [[memory]].
     *
     * The `props` argument is dependent on the backend.
     * For example, we can pass the following on `CUDA` and `HIP` backends to use a shared host pointer:
     *
     * ```cpp
     * {"host", true}
     * ```
     *
     * @descriptionEnd
     *
     * @instanceDescriptionStart
     *
     * Uses the templated type to determine the type and bytes.
     *
     * @instanceDescriptionEnd
     */
    template <class TM = void>
    occa::memory malloc(const dim_t entries,
                        const void *src = NULL,
                        const occa::json &props = occa::json());

    /**
     * @alias{malloc}
     */
    template <class TM = void>
    occa::memory malloc(const dim_t entries,
                        const occa::memory src,
                        const occa::json &props = occa::json());

    /**
     * @alias{malloc}
     */
    template <class TM = void>
    occa::memory malloc(const dim_t entries,
                        const occa::json &props);

    /**
     * @alias{malloc}
     *
     * @instanceDescriptionStart
     *
     * Same but takes a [[dtype_t]] rather than a template parameter.
     *
     * @instanceDescriptionEnd
     */
    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const void *src = NULL,
                        const occa::json &props = occa::json());

    /**
     * @alias{malloc}
     */
    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const occa::memory src,
                        const occa::json &props = occa::json());

    /**
     * @alias{malloc}
     */
    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const occa::json &props);

    void* umalloc(const dim_t entries,
                  const dtype_t &dtype,
                  const void *src = NULL,
                  const occa::json &props = occa::json());

    void* umalloc(const dim_t entries,
                  const dtype_t &dtype,
                  const occa::memory src,
                  const occa::json &props = occa::json());

    void* umalloc(const dim_t entries,
                  const dtype_t &dtype,
                  const occa::json &props);

    template <class TM = void>
    TM* umalloc(const dim_t entries,
                const void *src = NULL,
                const occa::json &props = occa::json());

    template <class TM = void>
    TM* umalloc(const dim_t entries,
                const occa::memory src,
                const occa::json &props = occa::json());

    template <class TM = void>
    TM* umalloc(const dim_t entries,
                const occa::json &props);

    /**
     * @id{wrapMemory}
     *
     * @descriptionStart
     *
     * Wrap a native backend pointer inside a [[memory]] for the device.
     * The simplest example would be on a `Serial` or `OpenMP` device, where a regular pointer allocated through `malloc` or `new` is passed in.
     * For other modes, such as CUDA or HIP, it takes the pointer allocated through their API.
     *
     * @descriptionEnd
     *
     * @instanceDescriptionStart
     *
     * Uses the templated type to determine the type and bytes.
     *
     * @instanceDescriptionEnd
     */
    template <class TM = void>
    occa::memory wrapMemory(const TM *ptr,
                            const dim_t entries,
                            const occa::json &props = occa::json());

    /**
     * @alias{wrapMemory}
     *
     * @instanceDescriptionStart
     *
     * Same but takes a [[dtype_t]] rather than a template parameter.
     *
     * @instanceDescriptionEnd
     *
     */
    occa::memory wrapMemory(const void *ptr,
                            const dim_t entries,
                            const dtype_t &dtype,
                            const occa::json &props = occa::json());
    //  |===============================
  };

  template <>
  hash_t hash(const occa::device &device);

  std::ostream& operator << (std::ostream &out,
                             const occa::device &device);

  template <>
  occa::memory device::malloc<void>(const dim_t entries,
                                    const void *src,
                                    const occa::json &props);

  template <>
  occa::memory device::malloc<void>(const dim_t entries,
                                    const occa::memory src,
                                    const occa::json &props);

  template <>
  occa::memory device::malloc<void>(const dim_t entries,
                                    const occa::json &props);
}

#include "device.tpp"

#endif
