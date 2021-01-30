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
   * @startDoc{device}
   *
   * Description:
   *   A [[device]] object maps to a physical device we want to program on.
   *   Examples include a CPU, GPU, or other physical accelerator like an FPGA.
   *
   *   There are 2 main uses of a device:
   *   - Memory allocation ([[memory]])
   *   - Compile and run code ([[kernel]])
   *
   *   # Setup
   *
   *   Setting up [[device]] objects is done through JSON properties.
   *   Here's an example of a CUDA device picking device `0` through its [[device.constructor]]:
   *
   *   ```cpp
   *   occa::device device({
   *     {"mode", "CUDA"},
   *     {"device_id", 0}
   *   })
   *   ```
   *
   *   JSON formatted strings can also be passed directly, which can be useful when loading from a config file.
   *
   *   ```cpp
   *   occa::device device(
   *     "{ mode: 'CUDA', device_id: 0 }"
   *   );
   *   ```
   *
   *   We can achieve the same using the [[device.setup]] method which take similar arguments as the constructors.
   *   For example:
   *
   *   ```cpp
   *   occa::device device;
   *   // ...
   *   device.setup({
   *     {"mode", "CUDA"},
   *     {"device_id", 0}
   *   })
   *   ```
   *
   *   # Memory allocation
   *
   *   We suggest allocating through the templated [[device.malloc]] method which will keep type information around.
   *   Here's an example which allocates memory on the device to fit a `float` array of size 10:
   *
   *   ```cpp
   *   occa::memory mem = device.malloc<float>(10);
   *   ```
   *
   *   # Kernel compilation
   *
   *   Kernel allocation can be done two ways, through [[a file|device.buildKernel]] or [[string source|device.buildKernelFromString]].
   *   Here's an example which builds a [[kernel]] from a file:
   *
   *   ```cpp
   *   occa::kernel addVectors = (
   *     device.buildKernel("addVectors.okl",
   *                        "addVectors")
   *   );
   *   ```
   *
   *   # Interoperability
   *
   *   Lastly, we allow for interoperability with supported backends/libraries by wrapping and unwrapping memory objects.
   *
   *   Here's an example which takes a native pointer and wraps it as a [[memory]] object through the [[device.wrapMemory]] method:
   *
   *   ```cpp
   *   occa::memory occaPtr = (
   *     device.wrapMemory<float>(ptr, 10)
   *   );
   *   ```
   *
   *   # Garbage collection
   *
   *   The [[device.free]] function can be called to free the device along with any other object allocated by it, such as [[memory]] and [[kernel]] objects.
   *   OCCA implemented reference counting by default so calling [[device.free]] is not required.
   *
   * @endDoc
   */
  class device : public gc::ringEntry_t {
    friend class modeDevice_t;
    friend class kernel;
    friend class memory;

  private:
    mutable modeDevice_t *modeDevice;

  public:
    /**
     * @startDoc{constructor[0]}
     *
     * Description:
     *   Creates a handle to a physical device we want to program on, such as a CPU, GPU, or other accelerator.
     *
     * Overloaded Description:
     *   Default constructor
     *
     * @endDoc
     */
    device();

    /**
     * @startDoc{constructor[1]}
     *
     * Overloaded Description:
     *   Takes a JSON-formatted string for the device props.
     *
     * Arguments:
     *   props:
     *     JSON-formatted string that defines the device properties
     *
     * @endDoc
     */
    device(const std::string &props);

    /**
     * @startDoc{constructor[2]}
     *
     * Overloaded Description:
     *   Takes an [[json]] argument for the device props.
     *
     * Arguments:
     *   props:
     *     Device properties
     *
     * @endDoc
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
     * @startDoc{dontUseRefs}
     *
     * Description:
     *   By default, a [[device]] will automatically call [[device.free]] through reference counting.
     *   Turn off automatic garbage collection through this method.
     *
     * @endDoc
     */
    void dontUseRefs();

    /**
     * @startDoc{operator_equals[0]}
     *
     * Description:
     *   Compare if two devices have the same references.
     *
     * Returns:
     *   If the references are the same, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator == (const occa::device &other) const;

    /**
     * @startDoc{operator_equals[1]}
     *
     * Description:
     *   Compare if two devices have different references.
     *
     * Returns:
     *   If the references are different, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator != (const occa::device &other) const;

    /**
     * @startDoc{isInitialized}
     *
     * Description:
     *   Check whether the [[device]] has been intialized.
     *
     * Returns:
     *   Returns `true` if the [[device]] has been intialized, through either the [[device.constructor]] or [[device.setup]].
     *
     * @endDoc
     */
    bool isInitialized();

    modeDevice_t* getModeDevice() const;

    /**
     * @startDoc{setup[0]}
     *
     * Description:
     *   Similar to [[device.constructor]] but can be called after creating the initial [[device]] object.
     *
     * @endDoc
     */
    void setup(const std::string &props);

    /**
     * @doc{setup[1]}
     */
    void setup(const occa::json &props);

    /**
     * @startDoc{free}
     *
     * Description:
     *   Free the device, which will also free:
     *   - Allocated [[memory]]
     *   - Built [[kernel]]
     *   - Created [[stream]] and [[streamTag]]
     *
     *   Calling [[device.isInitialized]] will return `false` now.
     *
     * @endDoc
     */
    void free();

    /**
     * @startDoc{mode}
     *
     * Description:
     *   Returns the device mode, matching the backend the device is targeting.
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
     *   Get the properties used to build the device.
     *
     * Description:
     *   Returns the properties used to build the [[device]].
     *
     * @endDoc
     */
    const occa::json& properties() const;

    const occa::json& kernelProperties() const;
    occa::json kernelProperties(const occa::json &additionalProps) const;

    const occa::json& memoryProperties() const;
    occa::json memoryProperties(const occa::json &additionalProps) const;

    const occa::json& streamProperties() const;
    occa::json streamProperties(const occa::json &additionalProps) const;

    /**
     * @startDoc{hash}
     *
     * Description:
     *   Gets the [[hash|hash_t]] of the device.
     *   Two devices should have the same hash if they point to the same hardware device
     *   and setup with the same properties.
     *
     * Returns:
     *   The device [[hash|hash_t]]
     *
     * @endDoc
     */
    hash_t hash() const;

    /**
     * @startDoc{memorySize}
     *
     * Description:
     *   Finds the device's memory capacity, not accounting for allocated memory.
     *
     * Description:
     *   Returns the memory capacity in bytes.
     *
     * @endDoc
     */
    udim_t memorySize() const;

    /**
     * @startDoc{memoryAllocated}
     *
     * Description:
     *   Find how much memory has been allocated by this specific device.
     *
     * Description:
     *   Returns the memory allocated in bytes.
     *
     * @endDoc
     */
    udim_t memoryAllocated() const;

    /**
     * @startDoc{finish}
     *
     * Description:
     *   Finishes any asynchronous operation queued up on the device, such as
     *   [[async memory allocations|device.malloc]] or [[kernel calls|kernel.operator_parentheses]].
     *
     * @endDoc
     */
    void finish();

    /**
     * @startDoc{hasSeparateMemorySpace}
     *
     * Description:
     *   Checks if the device memory is in a separate memory space than the host.
     *   If they are not in a separate space, it should be safe to access the memory directly
     *   in the host application.
     *   For example, accesses of the [[memory.ptr]] return pointer.
     *
     * Returns:
     *   Returns `true` if the memory is directly accesible through the host.
     *
     * @endDoc
     */
    bool hasSeparateMemorySpace();

    //  |---[ Stream ]------------------
    /**
     * @startDoc{createStream}
     *
     * Description:
     *   Creates and returns a new [[stream]] to queue operations on.
     *   If the backend supports streams, out-of-order work can be achieved through
     *   the use of streams.
     *
     *   > Note that the stream is created but not set as the active stream.
     *
     * Returns:
     *   Newly created [[stream]]
     *
     * @endDoc
     */
    stream createStream(const occa::json &props = occa::json());

    /**
     * @startDoc{getStream}
     *
     * Description:
     *   Returns the active [[stream]].
     *
     * Returns:
     *   Returns the active [[stream]].
     *
     * @endDoc
     */
    stream getStream();

    /**
     * @startDoc{setStream}
     *
     * Description:
     *   Sets the active [[stream]].
     *
     * @endDoc
     */
    void setStream(stream s);

    /**
     * @startDoc{tagStream}
     *
     * Description:
     *   Tag the active stream and return the created [[streamTag]].
     *
     * Returns:
     *   The created [[streamTag]].
     *
     * @endDoc
     */
    streamTag tagStream();

    /**
     * @startDoc{waitFor}
     *
     * Description:
     *   Wait for all operations queued up until the [[tag|streamTag]].
     *
     * @endDoc
     */
    void waitFor(streamTag tag);

    /**
     * @startDoc{timeBetween}
     *
     * Description:
     *   Returns the time taken between two [[tags|streamTag]].
     *
     * Returns:
     *   Returns the time in seconds.
     *
     * @endDoc
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
     * @startDoc{buildKernel}
     *
     * Arguments:
     *   filename:
     *     Location of the file to compile
     *   kernelName
     *     Specify the `@kernel` function name to use
     *   props:
     *     Backend-specific [[properties|json]] on how to compile the kernel.
     *
     * Returns:
     *   The compiled [[kernel]].
     *
     * Description:
     *   Builds a [[kernel]] given a filename, kernel name, and optional properties.
     *
     *   # Defines
     *
     *   Compile-time definitions can be passed through the `defines` path.
     *   For example:
     *
     *   ```cpp
     *   occa::json props;
     *   props["defines/TWO"] = 2;
     *   ```
     *
     *   # Includes
     *
     *   Headers can be `#include`-ed through the `includes` path.
     *   For example:
     *
     *   ```cpp
     *   occa::json props;
     *   props["includes"].asArray();
     *   props["includes"] += "my_header.hpp";
     *   ```
     *
     *   # Headers
     *
     *   Source code can be injected through the `headers` path.
     *   For example:
     *
     *   ```cpp
     *   occa::json props;
     *   props["headers"].asArray();
     *   props["headers"] += "#define TWO 2";
     *   ```
     *
     *   # Functions
     *
     *   Lastly, [[function]]'s can be captured through the `functions` path.
     *   For example:
     *
     *   ```cpp
     *   occa::json props;
     *   props["functions/add"] = (
     *     OCCA_FUNCTION({}, [=](float a, float b) -> float {
     *       return a + b;
     *     }
     *   );
     *   ```
     *
     * @endDoc
     */
    occa::kernel buildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::json &props = occa::json()) const;

    /**
     * @startDoc{buildKernelFromString}
     *
     * Description:
     *   Same as [[device.buildKernel]] but given the kernel source code rather than the filename.
     *
     * Arguments:
     *   content:
     *     Source code to complile
     *   kernelName
     *     Specify the `@kernel` function name to use
     *   props:
     *     Backend-specific [[properties|json]] on how to compile the kernel.
     *     More information in [[device.buildKernel]]
     *
     * Returns:
     *   The compiled [[kernel]].
     *
     * @endDoc
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
     * @startDoc{malloc[0]}
     *
     * Description:
     *   Allocates memory on the device and returns the [[memory]] handle.
     *   If a `src` pointer is passed, its data will be automatically copied to the allocated [[memory]].
     *
     *   The `props` argument is dependent on the backend.
     *   For example, we can pass the following on `CUDA` and `HIP` backends to use a shared host pointer:
     *
     *   ```cpp
     *   {"host", true}
     *   ```
     *
     * Overloaded Description:
     *   Uses the templated type to determine the type and bytes.
     *
     * Arguments:
     *   entries:
     *     The length of the allocated memory
     *   src:
     *     If non-`NULL`, copy the `src` contents to the newly allocated [[memory]]
     *   props:
     *     Backend-specific [[properties|json]] to describe allocation strategies
     *
     * Returns:
     *   The allocated [[memory]]
     *
     * @endDoc
     */
    template <class T = void>
    occa::memory malloc(const dim_t entries,
                        const void *src = NULL,
                        const occa::json &props = occa::json());

    /**
     * @doc{malloc[1]}
     */
    template <class T = void>
    occa::memory malloc(const dim_t entries,
                        const occa::memory src,
                        const occa::json &props = occa::json());

    /**
     * @doc{malloc[2]}
     */
    template <class T = void>
    occa::memory malloc(const dim_t entries,
                        const occa::json &props);

    /**
     * @startDoc{malloc[3]}
     *
     * Overloaded Description:
     *   Same but takes a [[dtype_t]] rather than a template parameter.
     *
     * Arguments:
     *   entries:
     *     The length of the allocated memory
     *   dtype:
     *     The [[dtype_t]] of what will be allocated, which defines the length of each entry
     *   src:
     *     If non-`NULL`, copy the `src` contents to the newly allocated [[memory]]
     *   props:
     *     Backend-specific [[properties|json]] to describe allocation strategies
     *
     * Returns:
     *   The allocated [[memory]]
     *
     * @endDoc
     */
    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const void *src = NULL,
                        const occa::json &props = occa::json());

    /**
     * @doc{malloc[4]}
     */
    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const occa::memory src,
                        const occa::json &props = occa::json());

    /**
     * @doc{malloc[5]}
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

    template <class T = void>
    T* umalloc(const dim_t entries,
                const void *src = NULL,
                const occa::json &props = occa::json());

    template <class T = void>
    T* umalloc(const dim_t entries,
                const occa::memory src,
                const occa::json &props = occa::json());

    template <class T = void>
    T* umalloc(const dim_t entries,
                const occa::json &props);

    /**
     * @startDoc{wrapMemory}
     *
     * Description:
     *   Wrap a native backend pointer inside a [[memory]] for the device.
     *   The simplest example would be on a `Serial` or `OpenMP` device, where a regular pointer allocated through `malloc` or `new` is passed in.
     *   For other modes, such as CUDA or HIP, it takes the pointer allocated through their API.
     *
     *   > Note that automatic garbage collection is not set for wrapped memory objects.
     *
     * Overloaded Description:
     *   Uses the templated type to determine the type and bytes.
     *
     * Returns:
     *   The wrapped [[memory]]
     *
     * @endDoc
     */
    template <class T = void>
    occa::memory wrapMemory(const T *ptr,
                            const dim_t entries,
                            const occa::json &props = occa::json());

    /**
     * @startDoc{wrapMemory}
     *
     * Overloaded Description:
     *   Same but takes a [[dtype_t]] rather than a template parameter.
     *
     * @endDoc
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
