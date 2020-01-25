#include <occa/modes/metal/memory.hpp>
#include <occa/modes/metal/device.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  namespace metal {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::properties &properties_) :
        occa::modeMemory_t(modeDevice_, size_, properties_),
        bufferOffset(0) {}

    memory::~memory() {
      if (isOrigin) {
        metalBuffer.free();
      }
      size = 0;
    }

    kernelArg memory::makeKernelArg() const {
      kernelArgData arg;

      arg.modeMemory = const_cast<memory*>(this);
      arg.data.void_ = (void*) NULL;
      arg.size       = sizeof(void*);
      arg.info       = kArgInfo::usePointer;

      return kernelArg(arg);
    }

    modeMemory_t* memory::addOffset(const dim_t offset) {
      memory *m = new memory(modeDevice,
                             size - offset,
                             properties);
      m->metalBuffer = metalBuffer;
      m->bufferOffset = bufferOffset + offset;
      return m;
    }

    const api::metal::buffer_t& memory::getMetalBuffer() {
      return metalBuffer;
    }

    void* memory::getPtr(const occa::properties &props) {
      if (!ptr) {
        ptr = (char*) metalBuffer.getPtr();
      }
      return ptr;
    }

    udim_t memory::getOffset() const {
      return bufferOffset;
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::properties &props) {
      const bool async = props.get("async", false);

      api::metal::commandQueue_t &metalCommandQueue = (
        ((metal::device*) modeDevice)->metalCommandQueue
      );
      metalCommandQueue.memcpy(metalBuffer,
                               offset,
                               src,
                               bytes,
                               async);
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::properties &props) {
      const bool async = props.get("async", false);

      api::metal::commandQueue_t &metalCommandQueue = (
        ((metal::device*) modeDevice)->metalCommandQueue
      );
      metalCommandQueue.memcpy(metalBuffer,
                               destOffset,
                               ((const metal::memory*) src)->metalBuffer,
                               srcOffset,
                               bytes,
                               async);
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset,
                        const occa::properties &props) const {

      const bool async = props.get("async", false);

      api::metal::commandQueue_t &metalCommandQueue = (
        ((metal::device*) modeDevice)->metalCommandQueue
      );
      metalCommandQueue.memcpy(dest,
                               metalBuffer,
                               offset,
                               bytes,
                               async);
    }

    void memory::detach() {
      size = 0;
    }
  }
}
