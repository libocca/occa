#ifndef OCCA_INTERNAL_CORE_DEVICE_HEADER
#define OCCA_INTERNAL_CORE_DEVICE_HEADER

#include <occa/core/device.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/utils/gc.hpp>
#include <occa/internal/utils/uva.hpp>
#include <occa/internal/lang/kernelMetadata.hpp>

namespace occa {
  class modeDevice_t {
   public:
    std::string mode;
    occa::json properties;
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

    modeDevice_t(const occa::json &json_);

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
    virtual hash_t kernelHash(const occa::json &props) const = 0;

    //  |---[ Stream ]------------------
    virtual modeStream_t* createStream(const occa::json &props) = 0;

    virtual streamTag tagStream() = 0;
    virtual void waitFor(streamTag tag) = 0;
    virtual double timeBetween(const streamTag &startTag,
                               const streamTag &endTag) = 0;
    //  |===============================

    //  |---[ Kernel ]------------------
    void writeKernelBuildFile(const std::string &filename,
                              const hash_t &kernelHash,
                              const occa::json &kernelProps,
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
                                      const occa::json &props) = 0;

    virtual modeKernel_t* buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::json &props) = 0;
    //  |===============================

    //  |---[ Memory ]------------------
    virtual modeMemory_t* malloc(const udim_t bytes,
                                 const void* src,
                                 const occa::json &props) = 0;

    virtual modeMemory_t* wrapMemory(const void *ptr,
                                     const udim_t bytes,
                                     const occa::json &props) = 0;

    virtual udim_t memorySize() const = 0;
    //  |===============================
    //==================================
  };
}

#endif
