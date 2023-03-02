#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/modes/dpcpp/device.hpp>
#include <occa/internal/modes/dpcpp/buffer.hpp>
#include <occa/internal/modes/dpcpp/memory.hpp>
#include <occa/internal/modes/dpcpp/stream.hpp>
#include <occa/internal/modes/dpcpp/streamTag.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa
{
  namespace dpcpp
  {
    memory::memory(buffer *b,
                   udim_t size_, dim_t offset_) :
      occa::modeMemory_t(b, size_, offset_) {
      ptr = b->ptr + offset;
    }

    memory::memory(memoryPool *memPool,
                   udim_t size_, dim_t offset_) :
      occa::modeMemory_t(memPool, size_, offset_) {
      ptr = memPool->buffer->ptr + offset;
    }

    memory::~memory()
    {
      ptr = nullptr;
    }

    void *memory::getKernelArgPtr() const
    {
      return ptr;
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset_,
                          const occa::json &props)
    {
      const bool async = props.get("async", false);

      occa::dpcpp::stream& q = getDpcppStream(getModeDevice()->currentStream);
      occa::dpcpp::streamTag e = q.memcpy(&(this->ptr)[offset_],src, bytes);

      if(!async)
        e.waitFor();
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::json &props)
    {
      const bool async = props.get("async", false);

      occa::dpcpp::stream& q = getDpcppStream(getModeDevice()->currentStream);
      occa::dpcpp::streamTag e = q.memcpy(&(this->ptr)[destOffset],&(src->ptr)[srcOffset], bytes);

      if(!async)
        e.waitFor();
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset_,
                        const occa::json &props) const
    {

      const bool async = props.get("async", false);

      occa::dpcpp::stream& q = getDpcppStream(getModeDevice()->currentStream);
      occa::dpcpp::streamTag e = q.memcpy(dest, &(this->ptr)[offset_], bytes);

      if(!async)
        e.waitFor();
    }

    void* memory::unwrap() {
      return static_cast<void*>(&ptr);
    }
  } // namespace dpcpp
} // namespace occa
