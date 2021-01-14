#include <occa/internal/modes/dpcpp/memory.hpp>
#include <occa/internal/modes/dpcpp/device.hpp>
#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa
{
  namespace dpcpp
  {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::properties &properties_)
        : occa::modeMemory_t(modeDevice_, size_, properties_) {}

    memory::~memory()
    {
      if (isOrigin)
      {
        ::sycl::queue *q = getCommandQueue();
        free(this->ptr, *q);
      }
      this->ptr = nullptr;
      size = 0;
    }

    ::sycl::queue *memory::getCommandQueue() const
    {
      return ((::occa::dpcpp::device *)modeDevice)->getCommandQueue();
    }

    kernelArg memory::makeKernelArg() const
    {
      kernelArgData arg;
      arg.modeMemory = const_cast<memory *>(this);
      arg.data.void_ = (void *)this->ptr;
      arg.size = sizeof(void *);
      arg.info = kArgInfo::usePointer;

      return kernelArg(arg);
    }

    modeMemory_t *memory::addOffset(const dim_t offset)
    {
      memory *m = new memory(modeDevice,
                             size - offset,
                             properties);

      m->ptr = this->ptr + offset;
      return m;
    }

    void *memory::getPtr(const occa::properties &props)
    {
      return this->ptr;
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::properties &props)
    {
      const bool async = props.get("async", false);
      ::sycl::queue *q = getCommandQueue();

      q->memcpy(&(this->ptr)[offset], &((char *)src)[offset], bytes);
      if (!async)
        q->wait();
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::properties &props)
    {
      const bool async = props.get("async", false);
      ::sycl::queue *q = getCommandQueue();
      q->memcpy(&(this->ptr)[destOffset], &(src->ptr)[srcOffset], bytes);

      if (!async)
        q->wait();
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset,
                        const occa::properties &props) const
    {

      const bool async = props.get("async", false);
      ::sycl::queue *q = getCommandQueue();

      q->memcpy(&((char *)dest)[offset], &(this->ptr)[offset], bytes);
      if (!async)
        q->wait();
    }

    void memory::detach()
    {
      size = 0;
    }

  } // namespace dpcpp
} // namespace occa
