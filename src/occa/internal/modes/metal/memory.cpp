#include <occa/internal/modes/metal/device.hpp>
#include <occa/internal/modes/metal/memory.hpp>
#include <occa/internal/modes/metal/buffer.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace metal {
    memory::memory(modeBuffer_t *modeBuffer_,
                   udim_t size_, dim_t offset_) :
      occa::modeMemory_t(modeBuffer_, size_, offset_),
      bufferOffset(offset) {
      buffer *b = dynamic_cast<buffer*>(modeBuffer);
      metalBuffer = b->metalBuffer;
      ptr = (char*) metalBuffer.getPtr();
    }

    memory::~memory() {
      metalBuffer = NULL;
      bufferOffset = 0;
      size = 0;
    }

    void* memory::getKernelArgPtr() const {
      return nullptr;
    }

    const api::metal::buffer_t& memory::getMetalBuffer() {
      return metalBuffer;
    }

    void* memory::getPtr() const {
      return ptr;
    }

    udim_t memory::getOffset() const {
      return bufferOffset;
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset_,
                          const occa::json &props) {
      const bool async = props.get("async", false);

      api::metal::commandQueue_t &metalCommandQueue = (
        ((metal::device*) getModeDevice())->metalCommandQueue
      );
      metalCommandQueue.memcpy(metalBuffer,
                               offset_,
                               src,
                               bytes,
                               async);
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::json &props) {
      const bool async = props.get("async", false);

      api::metal::commandQueue_t &metalCommandQueue = (
        ((metal::device*) getModeDevice())->metalCommandQueue
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
                        const udim_t offset_,
                        const occa::json &props) const {

      const bool async = props.get("async", false);

      api::metal::commandQueue_t &metalCommandQueue = (
        ((metal::device*) getModeDevice())->metalCommandQueue
      );
      metalCommandQueue.memcpy(dest,
                               metalBuffer,
                               offset_,
                               bytes,
                               async);
    }
  }
}
