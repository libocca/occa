#include <occa/modes/dpcpp/memory.hpp>
#include <occa/modes/dpcpp/device.hpp>
#include <occa/modes/dpcpp/utils.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  namespace dpcpp {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::properties &properties_) :
        occa::modeMemory_t(modeDevice_, size_, properties_),
        rootDpcppMem(NULL),
        rootOffset(0),
        mappedPtr(NULL) {}

    memory::~memory() {
      if (isOrigin) {
        // Free mapped-host pointer
	::sycl::queue *q = getCommandQueue();
	free(rootDpcppMem, *q);
	
	//TODO: Cedric check if there are cases where we can have mapped pointer
        if (mappedPtr) {
/*          OCCA_DPCPP_ERROR("Mapped Free: clEnqueueUnmapMemObject",
                            clEnqueueUnmapMemObject(getCommandQueue(),
                                                    dpcppMem,
                                                    mappedPtr,
                                                    0, NULL, NULL));*/
        }
      }

      // Is the root cl_mem or the root cl_mem hasn't been freed yet
      if (size && (isOrigin || (rootDpcppMem != NULL))) {
//        OCCA_OPENCL_ERROR("Mapped Free: free()",
//          free(rootDpcppMem);
      }

      rootDpcppMem = NULL;
      rootOffset = 0;

      dpcppMem = NULL;
      mappedPtr = NULL;
      size = 0;
    }

    ::sycl::queue* memory::getCommandQueue() const {
	return ((::occa::dpcpp::device*) modeDevice)->getCommandQueue();
    }

    kernelArg memory::makeKernelArg() const {
      kernelArgData arg;
      arg.modeMemory = const_cast<memory*>(this);
      arg.data.void_ = (void*) dpcppMem;
      arg.size       = sizeof(void*);
      arg.info       = kArgInfo::usePointer;

      return kernelArg(arg);
    }

    modeMemory_t* memory::addOffset(const dim_t offset) {
      memory *m = new memory(modeDevice,
                                             size - offset,
                                             properties);

      m->rootDpcppMem = rootDpcppMem;
      m->rootOffset = rootOffset + offset;

      /* Need to check this */
      m->dpcppMem = (void*)(((char*)rootDpcppMem) + offset);	
      return m;
    }

    void* memory::getPtr(const occa::properties &props) {
      if (props.get("mapped", false)) {
        return mappedPtr;
      }
      return dpcppMem;
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::properties &props) {
      const bool async = props.get("async", false);
      ::sycl::queue *q = getCommandQueue();
      if(async){
            q->memcpy(&((char*) dpcppMem)[offset], &((char*) src)[offset], bytes);
      }else{
            q->memcpy(&((char*) dpcppMem)[offset], &((char*) src)[offset], bytes);
            q->wait();

      }
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::properties &props) {
      const bool async = props.get("async", false);
      ::sycl::queue *q = getCommandQueue();
      if(async){
            q->memcpy(&((char*) dpcppMem)[destOffset], &(src->ptr)[srcOffset], bytes);
      }else{
            q->memcpy(&((char*) dpcppMem)[destOffset], &(src->ptr)[srcOffset], bytes);
            q->wait();
      
      }
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset,
                        const occa::properties &props) const {

      const bool async = props.get("async", false);
      ::sycl::queue *q = getCommandQueue();
      if(async){
            q->memcpy(&((char*) dest)[offset],&((const char*) dpcppMem)[offset], bytes);
      }else{
            q->memcpy(&((char*) dest)[offset], &((const char*) dpcppMem)[offset], bytes);
            q->wait();
      }

    }

    void memory::detach() {
      size = 0;
    }
  }
}
