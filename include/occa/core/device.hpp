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
    device(const std::string &props);
    device(const occa::json &props);
    device(jsonInitializerList initializer);

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

    void setup(const std::string &props);
    void setup(const occa::json &props);

    void free();

    const std::string& mode() const;

    const occa::json& properties() const;

    const occa::json& kernelProperties() const;
    occa::json kernelProperties(const occa::json &additionalProps) const;

    const occa::json& memoryProperties() const;
    occa::json memoryProperties(const occa::json &additionalProps) const;

    const occa::json& streamProperties() const;
    occa::json streamProperties(const occa::json &additionalProps) const;

    hash_t hash() const;

    udim_t memorySize() const;
    udim_t memoryAllocated() const;

    void finish();

    bool hasSeparateMemorySpace();

    //  |---[ Stream ]------------------
    stream createStream(const occa::json &props = occa::json());

    stream getStream();
    void setStream(stream s);

    streamTag tagStream();
    void waitFor(streamTag tag);
    double timeBetween(const streamTag &startTag,
                       const streamTag &endTag);
    //  |===============================

    //  |---[ Kernel ]------------------
    void setupKernelInfo(const occa::json &props,
                         const hash_t &sourceHash,
                         occa::json &kernelProps,
                         hash_t &kernelHash) const;

    hash_t applyDependencyHash(const hash_t &kernelHash) const;

    occa::kernel buildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::json &props = occa::json()) const;

    occa::kernel buildKernelFromString(const std::string &content,
                                       const std::string &kernelName,
                                       const occa::json &props = occa::json()) const;

    occa::kernel buildKernelFromBinary(const std::string &filename,
                                       const std::string &kernelName,
                                       const occa::json &props = occa::json()) const;
    //  |===============================

    //  |---[ Memory ]------------------
    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const void *src = NULL,
                        const occa::json &props = occa::json());

    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const occa::memory src,
                        const occa::json &props = occa::json());

    occa::memory malloc(const dim_t entries,
                        const dtype_t &dtype,
                        const occa::json &props);

    template <class TM = void>
    occa::memory malloc(const dim_t bytes,
                        const void *src = NULL,
                        const occa::json &props = occa::json());

    template <class TM = void>
    occa::memory malloc(const dim_t bytes,
                        const occa::memory src,
                        const occa::json &props = occa::json());

    template <class TM = void>
    occa::memory malloc(const dim_t bytes,
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

    template <class TM = void>
    occa::memory wrapMemory(const TM *ptr,
                            const dim_t entries,
                            const occa::json &props = occa::json());

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
  //====================================

  //---[ Utils ]------------------------
  occa::json getModeSpecificProps(const std::string &mode,
                                        const occa::json &props);

  occa::json getObjectSpecificProps(const std::string &mode,
                                          const std::string &object,
                                          const occa::json &props);

  occa::json initialObjectProps(const std::string &mode,
                                      const std::string &object,
                                      const occa::json &props);
  //====================================
}

#include "device.tpp"

#endif
