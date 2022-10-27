#ifndef OCCA_INTERNAL_CORE_BUFFER_HEADER
#define OCCA_INTERNAL_CORE_BUFFER_HEADER

#include <occa/core/memory.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/utils/gc.hpp>

namespace occa {
  class modeMemory_t;

  class modeBuffer_t : public gc::ringEntry_t {
   public:
    occa::json properties;

    gc::ring_t<modeMemory_t> modeMemoryRing;

    char *ptr;

    occa::modeDevice_t *modeDevice;

    udim_t size;

    bool isWrapped;

    modeBuffer_t(modeDevice_t *modeDevice_,
                 udim_t size_,
                 const occa::json &json_);


    //---[ Virtual Methods ]------------
    virtual ~modeBuffer_t();

    virtual bool needsFree() const;
    virtual void addModeMemoryRef(modeMemory_t *mem);
    virtual void removeModeMemoryRef(modeMemory_t *mem);

    virtual void malloc(udim_t bytes) {};

    virtual modeMemory_t* slice(const dim_t offset_,
                                const udim_t bytes) = 0;

    virtual void detach() {};
  };
}

#endif
