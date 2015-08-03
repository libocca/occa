#ifndef OCCA_ARRAY_HEADER
#define OCCA_ARRAY_HEADER

#include "occa/base.hpp"

namespace occa {
  typedef uintptr_t dim_t;

  static const int copyOnHost          = (1 << 0);
  static const int copyOnDevice        = (1 << 1);
  static const int copyOnHostAndDevice = (1 << 1);

  static const int dontUseIdxOrder = (1 << 0);
  static const int useIdxOrder     = (1 << 1);

  template <class TM, const int idxType = occa::dontUseIdxOrder>
  class array {
  public:
    occa::device device;
    occa::memory memory;

    TM *data_;

    int ks_[6];     // Passed to the kernel, not a problem for 32-bit
    dim_t s_[6];    // Strides

    int idxCount;
    dim_t fs_[7];   // Full Strides (used with idxOrder)
    int sOrder_[6]; // Stride Ordering

    array();

    template <class TM2, const int idxType2>
    array(const array<TM2,idxType2> &v);

    void initSOrder(const int idxCount_ = 1);

    template <class TM2, const int idxType2>
    array& operator = (const array<TM2,idxType2> &v);

    void free();

    //---[ Info ]-----------------------
    inline TM* data(){
      return data_;
    }

    inline dim_t entries(){
      return s_[0] * s_[1] * s_[2] * s_[3] * s_[4] * s_[5];
    }

    inline dim_t bytes(){
      return (entries() * sizeof(TM));
    }

    inline dim_t dim(const int i){
      return s_[i];
    }

    inline operator occa::kernelArg () {
      occa::kernelArg ret;

      ret.argc = 2;

      ret.args[0].mHandle = memory.getMHandle();
      ret.args[0].dHandle = memory.getDHandle();

      ret.args[0].data.void_ = memory.getMemoryHandle();
      ret.args[0].size       = sizeof(void*);
      ret.args[0].info       = kArgInfo::usePointer;

      ret.args[1].data.void_ = (void*) ks_;
      ret.args[1].size       = maxBase2(idxCount) * sizeof(int);
      ret.args[1].info       = kArgInfo::usePointer;

      return ret;
    }

    std::string idxOrderStr();

    //---[ clone() ]--------------------
    array<TM,idxType> clone(const int copyOn = copyOnHostAndDevice);

    array<TM,idxType> cloneOnCurrentDevice(const int copyOn = copyOnHostAndDevice);

    array<TM,idxType> cloneOn(occa::device device_,
                              const int copyOn = copyOnHostAndDevice);

    template <class TM2, const int idxType2>
    array<TM2,idxType2> clone(const int copyOn = copyOnHostAndDevice);

    template <class TM2, const int idxType2>
    array<TM2,idxType2> cloneOnCurrentDevice(const int copyOn = copyOnHostAndDevice);

    template <class TM2, const int idxType2>
    array<TM2,idxType2> cloneOn(occa::device device_,
                                const int copyOn = copyOnHostAndDevice);

    //---[ array(...) ]----------------
    array(const int dim, const dim_t *d);

    array(const dim_t d0);
    array(const dim_t d0, const dim_t d1);
    array(const dim_t d0, const dim_t d1, const dim_t d2);
    array(const dim_t d0, const dim_t d1, const dim_t d2,
          const dim_t d3);
    array(const dim_t d0, const dim_t d1, const dim_t d2,
          const dim_t d3, const dim_t d4);
    array(const dim_t d0, const dim_t d1, const dim_t d2,
          const dim_t d3, const dim_t d4, const dim_t d5);

    //---[ array(device, ...) ]--------
    array(occa::device device_, const int dim, const dim_t *d);

    array(occa::device device_,
          const dim_t d0);
    array(occa::device device_,
          const dim_t d0, const dim_t d1);
    array(occa::device device_,
          const dim_t d0, const dim_t d1, const dim_t d2);
    array(occa::device device_,
          const dim_t d0, const dim_t d1, const dim_t d2,
          const dim_t d3);
    array(occa::device device_,
          const dim_t d0, const dim_t d1, const dim_t d2,
          const dim_t d3, const dim_t d4);
    array(occa::device device_,
          const dim_t d0, const dim_t d1, const dim_t d2,
          const dim_t d3, const dim_t d4, const dim_t d5);

    //---[ allocate(...) ]--------------
  private:
    void allocate();

  public:
    void allocate(const int dim, const dim_t *d);

    void allocate(const dim_t d0);
    void allocate(const dim_t d0, const dim_t d1);
    void allocate(const dim_t d0, const dim_t d1, const dim_t d2);
    void allocate(const dim_t d0, const dim_t d1, const dim_t d2,
                  const dim_t d3);
    void allocate(const dim_t d0, const dim_t d1, const dim_t d2,
                  const dim_t d3, const dim_t d4);
    void allocate(const dim_t d0, const dim_t d1, const dim_t d2,
                  const dim_t d3, const dim_t d4, const dim_t d5);

    //---[ allocate(device, ...) ]------
    void allocate(occa::device device_, const int dim, const dim_t *d);

    void allocate(occa::device device_,
                  const dim_t d0);
    void allocate(occa::device device_,
                  const dim_t d0, const dim_t d1);
    void allocate(occa::device device_,
                  const dim_t d0, const dim_t d1, const dim_t d2);
    void allocate(occa::device device_,
                  const dim_t d0, const dim_t d1, const dim_t d2,
                  const dim_t d3);
    void allocate(occa::device device_,
                  const dim_t d0, const dim_t d1, const dim_t d2,
                  const dim_t d3, const dim_t d4);
    void allocate(occa::device device_,
                  const dim_t d0, const dim_t d1, const dim_t d2,
                  const dim_t d3, const dim_t d4, const dim_t d5);

    //---[ reshape(...) ]---------------
    void reshape(const int dim, const dim_t *d);

    void reshape(const dim_t d0);
    void reshape(const dim_t d0, const dim_t d1);
    void reshape(const dim_t d0, const dim_t d1, const dim_t d2);
    void reshape(const dim_t d0, const dim_t d1, const dim_t d2,
                 const dim_t d3);
    void reshape(const dim_t d0, const dim_t d1, const dim_t d2,
                 const dim_t d3, const dim_t d4);
    void reshape(const dim_t d0, const dim_t d1, const dim_t d2,
                 const dim_t d3, const dim_t d4, const dim_t d5);

    //---[ setIdxOrder(...) ]-----------
    void updateFS(const int idxCount_ = 1);

    void setIdxOrder(const int dim, const int *o);
    void setIdxOrder(const std::string &default_,
                     const std::string &given);

    void setIdxOrder(const int o0);
    void setIdxOrder(const int o0, const int o1);
    void setIdxOrder(const int o0, const int o1, const int o2);
    void setIdxOrder(const int o0, const int o1, const int o2,
                     const int o3);
    void setIdxOrder(const int o0, const int o1, const int o2,
                     const int o3, const int o4);
    void setIdxOrder(const int o0, const int o1, const int o2,
                     const int o3, const int o4, const int o5);

    //---[ Operators ]------------------
    inline TM& operator [] (const dim_t i0);

    inline TM& operator () (const dim_t i0);
    inline TM& operator () (const dim_t i0, const dim_t i1);
    inline TM& operator () (const dim_t i0, const dim_t i1, const dim_t i2);
    inline TM& operator () (const dim_t i0, const dim_t i1, const dim_t i2,
                            const dim_t i3);
    inline TM& operator () (const dim_t i0, const dim_t i1, const dim_t i2,
                            const dim_t i3, const dim_t i4);
    inline TM& operator () (const dim_t i0, const dim_t i1, const dim_t i2,
                            const dim_t i3, const dim_t i4, const dim_t i5);

    //---[ Syncs ]----------------------
    void startManaging();
    void stopManaging();

    void syncToDevice();
    void syncFromDevice();

    bool needsSync();
    void sync();
    void dontSync();
  };
}

#include "occa/array/array.tpp"

#endif
