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
#include <occa/internal/utils/gc.hpp>

namespace occa {
  class modeKernel_t; class kernel;
  class modeDevice_t; class device;
  class modeStreamTag_t; class streamTag;
  class deviceInfo;

  typedef std::map<std::string, kernel>   cachedKernelMap;
  typedef cachedKernelMap::iterator       cachedKernelMapIterator;
  typedef cachedKernelMap::const_iterator cCachedKernelMapIterator;

  //---[ device ]-----------------------
  class device : public gc::ringEntry_t {
    friend class modeDevice_t;
    friend class kernel;
    friend class memory;

  private:
    mutable modeDevice_t *modeDevice;

  public:
    device();
    device(modeDevice_t *modeDevice_);
    device(const occa::properties &props);

    device(const occa::device &other);
    device& operator = (const occa::device &other);
    ~device();

  private:
    void assertInitialized() const;
    void setModeDevice(modeDevice_t *modeDevice_);
    void removeDeviceRef();

  public:
    void dontUseRefs();

    bool operator == (const occa::device &other) const;
    bool operator != (const occa::device &other) const;

    bool isInitialized();

    modeDevice_t* getModeDevice() const;

    void setup(const occa::properties &props);

    void free();

    const std::string& mode() const;

    const occa::properties& properties() const;

    const occa::properties& kernelProperties() const;
    occa::properties kernelProperties(const occa::properties &additionalProps) const;

    const occa::properties& memoryProperties() const;
    occa::properties memoryProperties(const occa::properties &additionalProps) const;

    const occa::properties& streamProperties() const;
    occa::properties streamProperties(const occa::properties &additionalProps) const;

    hash_t hash() const;

    udim_t memorySize() const;
    udim_t memoryAllocated() const;

    void finish();

    bool hasSeparateMemorySpace();

    //  |---[ Stream ]------------------
    stream createStream(const occa::properties &props = occa::properties());

    stream getStream();
    void setStream(stream s);

    streamTag tagStream();
    void waitFor(streamTag tag);
    double timeBetween(const streamTag &startTag,
                       const streamTag &endTag);
    //  |===============================

    //  |---[ Kernel ]------------------
    void setupKernelInfo(const occa::properties &props,
                         const hash_t &sourceHash,
                         occa::properties &kernelProps,
                         hash_t &kernelHash) const;

    hash_t applyDependencyHash(const hash_t &kernelHash) const;

    occa::kernel buildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::properties &props = occa::properties()) const;

    occa::kernel buildKernelFromString(const std::string &content,
                                       const std::string &kernelName,
                                       const occa::properties &props = occa::properties()) const;

    occa::kernel buildKernelFromBinary(const std::string &filename,
                                       const std::string &kernelName,
                                       const occa::properties &props = occa::properties()) const;

    void loadKernels(const std::string &library = "");
    //  |===============================

    //  |---[ Memory ]------------------
    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const void *src = NULL,
                        const occa::properties &props = occa::properties());

    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const occa::memory src,
                        const occa::properties &props = occa::properties());

    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const occa::properties &props);

    template <class TM = void>
    occa::memory malloc(const dim_t bytes,
                        const void *src = NULL,
                        const occa::properties &props = occa::properties());

    template <class TM = void>
    occa::memory malloc(const dim_t bytes,
                        const occa::memory src,
                        const occa::properties &props = occa::properties());

    template <class TM = void>
    occa::memory malloc(const dim_t bytes,
                        const occa::properties &props);

    void* umalloc(const dim_t entries,
                  const dtype_t &dtype,
                  const void *src = NULL,
                  const occa::properties &props = occa::properties());

    void* umalloc(const dim_t entries,
                  const dtype_t &dtype,
                  const occa::memory src,
                  const occa::properties &props = occa::properties());

    void* umalloc(const dim_t entries,
                  const dtype_t &dtype,
                  const occa::properties &props);

    template <class TM = void>
    TM* umalloc(const dim_t entries,
                const void *src = NULL,
                const occa::properties &props = occa::properties());

    template <class TM = void>
    TM* umalloc(const dim_t entries,
                const occa::memory src,
                const occa::properties &props = occa::properties());

    template <class TM = void>
    TM* umalloc(const dim_t entries,
                const occa::properties &props);
    //  |===============================
  };

  template <>
  hash_t hash(const occa::device &device);

  std::ostream& operator << (std::ostream &out,
                           const occa::device &device);

  template <>
  occa::memory device::malloc<void>(const dim_t entries,
                                    const void *src,
                                    const occa::properties &props);

  template <>
  occa::memory device::malloc<void>(const dim_t entries,
                                    const occa::memory src,
                                    const occa::properties &props);

  template <>
  occa::memory device::malloc<void>(const dim_t entries,
                                    const occa::properties &props);
  //====================================

  //---[ Utils ]------------------------
  occa::properties getModeSpecificProps(const std::string &mode,
                                        const occa::properties &props);

  occa::properties getObjectSpecificProps(const std::string &mode,
                                          const std::string &object,
                                          const occa::properties &props);

  occa::properties initialObjectProps(const std::string &mode,
                                      const std::string &object,
                                      const occa::properties &props);
  //====================================
}

#include "device.tpp"

#endif
