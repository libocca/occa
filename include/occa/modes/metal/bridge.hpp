#ifndef OCCA_MODES_METAL_UTILS_HEADER
#define OCCA_MODES_METAL_UTILS_HEADER

#include <occa/types.hpp>

namespace occa {
  class kernelArgData;
  namespace io {
    class lock_t;
  }

  namespace metal {
    class metalEvent_t {
     public:
      void *obj;

      metalEvent_t();
      metalEvent_t(const metalEvent_t &other);

      void free();

      double getTime() const;
    };

    class metalBuffer_t {
     public:
      void *obj;

      metalBuffer_t();
      metalBuffer_t(const metalBuffer_t &other);

      void free();

      void* getPtr() const;
    };

    class metalCommandQueue_t {
     public:
      void *obj;

      metalCommandQueue_t();
      metalCommandQueue_t(const metalCommandQueue_t &other);

      void free();
    };

    class metalKernel_t {
     public:
      void *obj;

      metalKernel_t();
      metalKernel_t(const metalKernel_t &other);

      void clearArguments();

      void addArgument(const int index,
                       const kernelArgData &arg);

      void run(occa::dim outerDims,
               occa::dim innerDims);

      void free();
    };

    class metalDevice_t {
     public:
      void *deviceObj;
      void *libraryObj;

      metalDevice_t();
      metalDevice_t(const metalDevice_t &other);

      void free();

      static int getCount();
      static metalDevice_t fromId(const int id);

      std::string getName() const;
      udim_t getMemorySize() const;
      dim getMaxOuterDims() const;
      dim getMaxInnerDims() const;

      metalCommandQueue_t createCommandQueue() const;
      metalEvent_t createEvent() const;

      metalKernel_t buildKernel(const std::string &source,
                                const std::string &kernelName,
                                io::lock_t &lock) const;

      metalBuffer_t malloc(const udim_t bytes,
                           const void *src) const;

      void memcpy(metalBuffer_t &dest,
                  const udim_t destOffset,
                  const metalBuffer_t &src,
                  const udim_t srcOffset,
                  const udim_t bytes,
                  const bool async) const;

      void memcpy(void *dest,
                  const metalBuffer_t &src,
                  const udim_t srcOffset,
                  const udim_t bytes,
                  const bool async) const;

      void memcpy(metalBuffer_t &dest,
                  const udim_t destOffset,
                  const void *src,
                  const udim_t bytes,
                  const bool async) const;

      void waitFor(metalEvent_t &event) const;

      void finish() const;
    };
  }
}

#endif
