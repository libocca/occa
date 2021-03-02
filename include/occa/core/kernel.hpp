#ifndef OCCA_CORE_KERNEL_HEADER
#define OCCA_CORE_KERNEL_HEADER

#include <initializer_list>
#include <iostream>
#include <stdint.h>
#include <vector>

#include <occa/defines.hpp>
#include <occa/core/kernelArg.hpp>
#include <occa/types.hpp>

// Unfortunately we need to expose this in include
#include <occa/utils/gc.hpp>

namespace occa {
  class modeKernel_t; class kernel;
  class modeMemory_t; class memory;
  class modeDevice_t; class device;
  class kernelBuilder;

  namespace lang {
    class parser_t;
  }

  typedef std::map<hash_t, kernel>            hashedKernelMap;
  typedef hashedKernelMap::iterator           hashedKernelMapIterator;
  typedef hashedKernelMap::const_iterator     cHashedKernelMapIterator;

  typedef std::vector<kernelBuilder>          kernelBuilderVector;
  typedef kernelBuilderVector::iterator       kernelBuilderVectorIterator;
  typedef kernelBuilderVector::const_iterator cKernelBuilderVectorIterator;

  /**
   * @startDoc{kernel}
   *
   * Description:
   *
   *   A [[kernel]] object is a handle to a device function for the device it was built in.
   *   For example, in `Serial` and `OpenMP` modes it is analogous to a calling a C++ function.
   *   For GPU modes, it means launching work on a more granular and parallized manner.
   *
   *   # Launch
   *
   *   There are 2 ways to launching kernels:
   *   - [[kernel.operator_parentheses]] which can be used to call a kernel like a regular function.
   *   - [[kernel.run]] which requires the user to push the arguments one-by-one before running it.
   *
   *   # Garbage collection
   *
   *   The [[kernel.free]] function can be called to free the kernel.
   *   OCCA implemented reference counting by default so calling [[kernel.free]] is not required.
   *
   * @endDoc
   */
  class kernel : public gc::ringEntry_t {
    friend class occa::modeKernel_t;
    friend class occa::device;

  private:
    modeKernel_t *modeKernel;

  public:
    kernel();
    kernel(modeKernel_t *modeKernel_);

    kernel(const kernel &k);
    kernel& operator = (const kernel &k);
    kernel& operator = (modeKernel_t *modeKernel_);
    ~kernel();

  private:
    void assertInitialized() const;
    void setModeKernel(modeKernel_t *modeKernel_);
    void removeKernelRef();

  public:
    /**
     * @startDoc{dontUseRefs}
     *
     * Description:
     *   By default, a [[kernel]] will automatically call [[kernel.free]] through reference counting.
     *   Turn off automatic garbage collection through this method.
     *
     * @endDoc
     */
    void dontUseRefs();

    /**
     * @startDoc{isInitialized}
     *
     * Description:
     *   Check whether the [[kernel]] has been intialized.
     *
     * Returns:
     *   Returns `true` if the [[kernel]] has been built successfully
     *
     * @endDoc
     */
    bool isInitialized();

    /**
     * @startDoc{mode}
     *
     * Description:
     *   Returns the mode of the [[device]] used to build the [[kernel]].
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
     *   Get the properties used to build the [[kernel]].
     *
     * Description:
     *   Returns the properties used to build the [[kernel]].
     *
     * @endDoc
     */
    const occa::json& properties() const;

    modeKernel_t* getModeKernel() const;

    /**
     * @startDoc{getDevice}
     *
     * Description:
     *   Returns the [[device]] used to build the [[kernel]].
     *
     * Returns:
     *   The [[device]] used to build the [[kernel]]
     *
     * @endDoc
     */
    occa::device getDevice();

    /**
     * @startDoc{operator_equals[0]}
     *
     * Description:
     *   Compare if two kernels have the same references.
     *
     * Returns:
     *   If the references are the same, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator == (const occa::kernel &other) const;

    /**
     * @startDoc{operator_equals[1]}
     *
     * Description:
     *   Compare if two kernels have different references.
     *
     * Returns:
     *   If the references are different, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator != (const occa::kernel &other) const;

    /**
     * @startDoc{name}
     *
     * Description:
     *   Get the kernel name.
     *
     * Returns:
     *   The name of the kernel
     *
     * @endDoc
     */
    const std::string& name();

    /**
     * @startDoc{sourceFilename}
     *
     * Description:
     *   Find where the kernel source came from.
     *
     * Returns:
     *   The location of the kernel source code.
     *
     * @endDoc
     */
    const std::string& sourceFilename();

    /**
     * @startDoc{binaryFilename}
     *
     * Description:
     *   Find where the kernel binary was compiled to.
     *
     * Returns:
     *   The location of the kernel binary.
     *
     * @endDoc
     */
    const std::string& binaryFilename();

    /**
     * @startDoc{hash}
     *
     * Description:
     *   Gets the [[hash|hash_t]] of the kernel.
     *   Two kernels should have the same hash if they were compiled with the same source
     *   and setup with the same properties.
     *
     * Returns:
     *   The kernel [[hash|hash_t]]
     *
     * @endDoc
     */
    hash_t hash();

    int maxDims();
    dim maxOuterDims();
    dim maxInnerDims();

    /**
     * @startDoc{setRunDims}
     *
     * Description:
     *   If the [[kernel]] was compiled without OKL, the outer and inner dimensions
     *   need to be manually set.
     *   The dimensions are required when running modes such as `CUDA`, `HIP`, and `OpenCL`.
     *
     * @endDoc
     */
    void setRunDims(dim outerDims, dim innerDims);

    /**
     * @startDoc{pushArg}
     *
     * Description:
     *   Push the next argument that will be passed to the backend kernel.
     *
     *   See [[kernel.run]] for more information.
     *
     * @endDoc
     */
    void pushArg(const kernelArg &arg);

    /**
     * @startDoc{clearArgs}
     *
     * Description:
     *   Clear arguments pushed through the [[kernel.pushArg]] method.
     *
     *   See [[kernel.run]] for more information.
     *
     * @endDoc
     */
    void clearArgs();

    /**
     * @startDoc{run[0]}
     *
     * Description:
     *   The more common way to run a [[kernel]] is through [[kernel.operator_parentheses]].
     *   However, we also offer a way to run a kernel by manually pushing arguments to it.
     *   This is useful when building a kernel dynamically.
     *
     *   Manually push arguments through [[kernel.pushArg]], followed by calling this `run` function.
     *
     *   To clear the arguments, use [[kernel.clearArgs]].
     *
     * @endDoc
     */
    void run() const;

    /**
     * @doc{run[1]}
     */
    void run(std::initializer_list<kernelArg> args) const;

    /**
     * @startDoc{operator_parentheses}
     *
     * Description:
     *   Pass [[kernelArg]] arguments to launch the kernel.
     *
     *   To manually push arguments, look at the [[kernel.run]] method.
     *
     * Argument Override:
     *    [[kernelArg]]... args
     *
     * @endDoc
     */
#include "kernelOperators.hpp_codegen"

    /**
     * @startDoc{free}
     *
     * Description:
     *   Free the kernel object.
     *   Calling [[kernel.isInitialized]] will return `false` now.
     *
     * @endDoc
     */
    void free();
  };


  //---[ Kernel Properties ]------------
  // Properties:
  //   defines       : Object
  //   includes      : Array
  //   header        : Array
  //   include_paths : Array
  hash_t kernelHeaderHash(const occa::json &props);

  std::string assembleKernelHeader(const occa::json &props);
  //====================================
}

#endif
