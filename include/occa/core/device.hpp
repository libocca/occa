#ifndef OCCA_CORE_DEVICE_HEADER
#define OCCA_CORE_DEVICE_HEADER

#include <iostream>
#include <sstream>

#include <occa/core/kernel.hpp>
#include <occa/core/memory.hpp>
#include <occa/core/stream.hpp>
#include <occa/defines.hpp>
#include <occa/dtype.hpp>
#include <occa/io/output.hpp>
#include <occa/tools/gc.hpp>
#include <occa/tools/uva.hpp>

namespace occa {
  class modeKernel_t; class kernel;
  class modeDevice_t; class device;
  class modeStreamTag_t; class streamTag;
  class deviceInfo;

  typedef std::map<std::string, kernel>   cachedKernelMap;
  typedef cachedKernelMap::iterator       cachedKernelMapIterator;
  typedef cachedKernelMap::const_iterator cCachedKernelMapIterator;

  //---[ modeDevice_t ]---------------------
  class modeDevice_t {
  public:
    std::string mode;
    occa::properties properties;
    bool needsLauncherKernel;

    gc::ring_t<device> deviceRing;
    gc::ring_t<modeKernel_t> kernelRing;
    gc::ring_t<modeMemory_t> memoryRing;
    gc::ring_t<modeStream_t> streamRing;
    gc::ring_t<modeStreamTag_t> streamTagRing;

    ptrRangeMap uvaMap;
    memoryVector uvaStaleMemory;

    stream currentStream;
    std::vector<modeStream_t*> streams;

    udim_t bytesAllocated;

    cachedKernelMap cachedKernels;

    modeDevice_t(const occa::properties &properties_);

    template <class modeType_t>
    void freeRing(gc::ring_t<modeType_t> ring) {
      while (ring.head) {
        modeType_t *ptr = (modeType_t*) ring.head;
        ring.removeRef(ptr);
        delete ptr;
      }
    }

    // Must be called before ~modeDevice_t()!
    void freeResources();

    void dontUseRefs();
    void addDeviceRef(device *dev);
    void removeDeviceRef(device *dev);
    bool needsFree() const;

    void addKernelRef(modeKernel_t *kernel);
    void removeKernelRef(modeKernel_t *kernel);

    void addMemoryRef(modeMemory_t *memory);
    void removeMemoryRef(modeMemory_t *memory);

    void addStreamRef(modeStream_t *stream);
    void removeStreamRef(modeStream_t *stream);

    void addStreamTagRef(modeStreamTag_t *streamTag);
    void removeStreamTagRef(modeStreamTag_t *streamTag);

    //---[ Virtual Methods ]------------
    virtual ~modeDevice_t() = 0;

    virtual void finish() const = 0;

    virtual bool hasSeparateMemorySpace() const = 0;

    hash_t versionedHash() const;
    virtual hash_t hash() const = 0;
    virtual hash_t kernelHash(const occa::properties &props) const = 0;

    //  |---[ Stream ]------------------
    virtual modeStream_t* createStream(const occa::properties &props) = 0;

    virtual streamTag tagStream() = 0;
    virtual void waitFor(streamTag tag) = 0;
    virtual double timeBetween(const streamTag &startTag,
                               const streamTag &endTag) = 0;
    //  |===============================

    //  |---[ Kernel ]------------------
    void writeKernelBuildFile(const std::string &filename,
                              const hash_t &kernelHash,
                              const occa::properties &kernelProps,
                              const lang::sourceMetadata_t &metadataMap) const;

    std::string getKernelHash(const std::string &fullHash,
                              const std::string &kernelName);

    std::string getKernelHash(const hash_t &kernelHash,
                              const std::string &kernelName);

    std::string getKernelHash(modeKernel_t *kernel);

    kernel& getCachedKernel(const hash_t &kernelHash,
                            const std::string &kernelName);

    void removeCachedKernel(modeKernel_t *kernel);

    virtual modeKernel_t* buildKernel(const std::string &filename,
                                      const std::string &kernelName,
                                      const hash_t hash,
                                      const occa::properties &props) = 0;

    virtual modeKernel_t* buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::properties &props) = 0;
    //  |===============================

    //  |---[ Memory ]------------------
    virtual modeMemory_t* malloc(const udim_t bytes,
                                 const void* src,
                                 const occa::properties &props) = 0;

    virtual udim_t memorySize() const = 0;
    //  |===============================
    //==================================
  };
  //====================================

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
