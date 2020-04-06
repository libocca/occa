#include <occa/defines.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  template <class TM, const int idxType>
  array<TM,idxType>::array() :
    device(),
    memory_(),
    ptr_(NULL) {

    for (int i = 0; i < 6; ++i) {
      ks_[i]     = 0;
      s_[i]      = 0;
      sOrder_[i] = i;
    }
    initSOrder();
  }


  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM,idxType>::array(const array<TM2,idxType2> &v) {
    *this = v;
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM,idxType>& array<TM,idxType>::operator = (const array<TM2,idxType2> &v) {
    device  = v.device;
    memory_ = v.memory_;

    ptr_ = v.ptr_;

    initSOrder(v.idxCount);

    for (int i = 0; i < idxCount; ++i) {
      ks_[i]     = v.ks_[i];
      s_[i]      = v.s_[i];
      sOrder_[i] = v.sOrder_[i];
    }

    if (idxType == occa::dynamic) {
      updateFS(v.idxCount);
    }
    return *this;
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::initSOrder(int idxCount_) {
    idxCount = idxCount_;

    if (idxType == occa::dynamic) {
      for (int i = 0; i < 6; ++i) {
        sOrder_[i] = i;
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::free() {
    if (ptr_ == NULL) {
      return;
    }

    occa::free(ptr_);
    ptr_ = NULL;

    for (int i = 0; i < 6; ++i) {
      ks_[i]     = 0;
      s_[i]      = 0;
      sOrder_[i] = i;
    }
  }

  //---[ Info ]-------------------------
  template <class TM, const int idxType>
  std::string array<TM,idxType>::indexingStr() {
    if (idxType == occa::dynamic) {
      std::string str(2*idxCount - 1, ',');
      for (int i = 0; i < idxCount; ++i) {
        str[2*i] = ('0' + sOrder_[idxCount - i - 1]);
      }
      return str;
    } else {
      OCCA_FORCE_ERROR("Only occa::array<TM, occa::dynamic> can use reindex()");
    }
    return "";
  }

  //---[ clone() ]----------------------
  template <class TM, const int idxType>
  array<TM,idxType> array<TM,idxType>::clone(const int copyOn) {
    return cloneOn(device, copyOn);
  }

  template <class TM, const int idxType>
  array<TM,idxType> array<TM,idxType>::cloneOnCurrentDevice(const int copyOn) {
    return cloneOn(occa::getDevice(), copyOn);
  }

  template <class TM, const int idxType>
  array<TM,idxType> array<TM,idxType>::cloneOn(occa::device device_, const int copyOn) {
    return cloneOn<TM,idxType>(device_, copyOn);
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM2,idxType2> array<TM,idxType>::clone(const int copyOn) {
    return cloneOn<TM2>(device, copyOn);
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM2,idxType2> array<TM,idxType>::cloneOnCurrentDevice(const int copyOn) {
    return cloneOn<TM2>(occa::getDevice(), copyOn);
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM2,idxType2> array<TM,idxType>::cloneOn(occa::device device_, const int copyOn) {
    array<TM2,idxType2> clone_ = *this;

    clone_.allocate(device_, idxCount, s_);
    occa::memcpy(clone_.ptr_, ptr_, bytes());

    return clone_;
  }

  //---[ array(...) ]------------------
  template <class TM, const int idxType>
  array<TM,idxType>::array(const int dim_, const udim_t *d,
                           const TM *src) {
    initSOrder(dim_);

    switch(dim_) {
    case 1: allocate(d[0], src);                               break;
    case 2: allocate(d[0], d[1], src);                         break;
    case 3: allocate(d[0], d[1], d[2], src);                   break;
    case 4: allocate(d[0], d[1], d[2], d[3], src);             break;
    case 5: allocate(d[0], d[1], d[2], d[3], d[4], src);       break;
    case 6: allocate(d[0], d[1], d[2], d[3], d[4], d[5], src); break;
    default:
      if (dim_ <= 0) {
        OCCA_FORCE_ERROR("Number of dimensions must be [1-6]");
      } else {
        OCCA_FORCE_ERROR("occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const int dim_, const udim_t *d,
                           const occa::memory src) {
    initSOrder(dim_);

    switch(dim_) {
    case 1: allocate(d[0], src);                               break;
    case 2: allocate(d[0], d[1], src);                         break;
    case 3: allocate(d[0], d[1], d[2], src);                   break;
    case 4: allocate(d[0], d[1], d[2], d[3], src);             break;
    case 5: allocate(d[0], d[1], d[2], d[3], d[4], src);       break;
    case 6: allocate(d[0], d[1], d[2], d[3], d[4], d[5], src); break;
    default:
      if (dim_ <= 0) {
        OCCA_FORCE_ERROR("Number of dimensions must be [1-6]");
      } else {
        OCCA_FORCE_ERROR("occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0,
                           const TM *src) {
    initSOrder(1);
    allocate(d0, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1,
                           const TM *src) {
    initSOrder(2);
    allocate(d0, d1, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const TM *src) {
    initSOrder(3);
    allocate(d0, d1, d2, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3,
                           const TM *src) {
    initSOrder(4);
    allocate(d0, d1, d2, d3, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4,
                           const TM *src) {
    initSOrder(5);
    allocate(d0, d1, d2, d3, d4, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4, const udim_t d5,
                           const TM *src) {
    initSOrder(6);
    allocate(d0, d1, d2, d3, d4, d5, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0,
                           const occa::memory src) {
    initSOrder(1);
    allocate(d0, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1,
                           const occa::memory src) {
    initSOrder(2);
    allocate(d0, d1, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const occa::memory src) {
    initSOrder(3);
    allocate(d0, d1, d2, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3,
                           const occa::memory src) {
    initSOrder(4);
    allocate(d0, d1, d2, d3, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4,
                           const occa::memory src) {
    initSOrder(5);
    allocate(d0, d1, d2, d3, d4, src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4, const udim_t d5,
                           const occa::memory src) {
    initSOrder(6);
    allocate(d0, d1, d2, d3, d4, d5, src);
  }

  //---[ array(device, ...) ]----------
  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const int dim_, const udim_t *d,
                           const TM *src) {
    switch(dim_) {
    case 1: allocate(device_, d[0], src);                               break;
    case 2: allocate(device_, d[0], d[1], src);                         break;
    case 3: allocate(device_, d[0], d[1], d[2], src);                   break;
    case 4: allocate(device_, d[0], d[1], d[2], d[3], src);             break;
    case 5: allocate(device_, d[0], d[1], d[2], d[3], d[4], src);       break;
    case 6: allocate(device_, d[0], d[1], d[2], d[3], d[4], d[5], src); break;
    default:
      if (dim_ <= 0) {
        OCCA_FORCE_ERROR("Number of dimensions must be [1-6]");
      } else {
        OCCA_FORCE_ERROR("occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const int dim_, const udim_t *d,
                           const occa::memory src) {
    switch(dim_) {
    case 1: allocate(device_, d[0], src);                               break;
    case 2: allocate(device_, d[0], d[1], src);                         break;
    case 3: allocate(device_, d[0], d[1], d[2], src);                   break;
    case 4: allocate(device_, d[0], d[1], d[2], d[3], src);             break;
    case 5: allocate(device_, d[0], d[1], d[2], d[3], d[4], src);       break;
    case 6: allocate(device_, d[0], d[1], d[2], d[3], d[4], d[5], src); break;
    default:
      if (dim_ <= 0) {
        OCCA_FORCE_ERROR("Number of dimensions must be [1-6]");
      } else {
        OCCA_FORCE_ERROR("occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0,
                           const TM *src) {
    initSOrder(1);
    allocate(device_,
             d0,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1,
                           const TM *src) {
    initSOrder(2);
    allocate(device_,
             d0, d1,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const TM *src) {
    initSOrder(3);
    allocate(device_,
             d0, d1, d2,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3,
                           const TM *src) {
    initSOrder(4);
    allocate(device_,
             d0, d1, d2, d3,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4,
                           const TM *src) {
    initSOrder(5);
    allocate(device_,
             d0, d1, d2, d3, d4,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4, const udim_t d5,
                           const TM *src) {
    initSOrder(6);
    allocate(device_,
             d0, d1, d2, d3, d4, d5,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0,
                           const occa::memory src) {
    initSOrder(1);
    allocate(device_,
             d0,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1,
                           const occa::memory src) {
    initSOrder(2);
    allocate(device_,
             d0, d1,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const occa::memory src) {
    initSOrder(3);
    allocate(device_,
             d0, d1, d2,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3,
                           const occa::memory src) {
    initSOrder(4);
    allocate(device_,
             d0, d1, d2, d3,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4,
                           const occa::memory src) {
    initSOrder(5);
    allocate(device_,
             d0, d1, d2, d3, d4,
             src);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4, const udim_t d5,
                           const occa::memory src) {
    initSOrder(6);
    allocate(device_,
             d0, d1, d2, d3, d4, d5,
             src);
  }

  //---[ allocate(...) ]----------------
  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const TM *src) {
    ptr_    = (TM*) device.umalloc(size(), dtype::get<TM>(), src);
    memory_ = occa::memory(ptr_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const occa::memory src) {
    ptr_    = (TM*) device.umalloc(size(), dtype::get<TM>(), src);
    memory_ = occa::memory(ptr_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const int dim_, const udim_t *d,
                                   const TM *src) {
    switch(dim_) {
    case 1: allocate(d[0], src);                               break;
    case 2: allocate(d[0], d[1], src);                         break;
    case 3: allocate(d[0], d[1], d[2], src);                   break;
    case 4: allocate(d[0], d[1], d[2], d[3], src);             break;
    case 5: allocate(d[0], d[1], d[2], d[3], d[4], src);       break;
    case 6: allocate(d[0], d[1], d[2], d[3], d[4], d[5], src); break;
    default:
      if (dim_ <= 0) {
        OCCA_FORCE_ERROR("Number of dimensions must be [1-6]");
      } else {
        OCCA_FORCE_ERROR("occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const int dim_, const udim_t *d, occa::memory src) {
    switch(dim_) {
    case 1: allocate(d[0], src);                               break;
    case 2: allocate(d[0], d[1], src);                         break;
    case 3: allocate(d[0], d[1], d[2], src);                   break;
    case 4: allocate(d[0], d[1], d[2], d[3], src);             break;
    case 5: allocate(d[0], d[1], d[2], d[3], d[4], src);       break;
    case 6: allocate(d[0], d[1], d[2], d[3], d[4], d[5], src); break;
    default:
      if (dim_ <= 0) {
        OCCA_FORCE_ERROR("Number of dimensions must be [1-6]");
      } else {
        OCCA_FORCE_ERROR("occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0,
                                   const TM *src) {
    allocate(occa::getDevice(),
             d0,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1,
                                   const TM *src) {
    allocate(occa::getDevice(),
             d0, d1,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const TM *src) {
    allocate(occa::getDevice(),
             d0, d1, d2,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3,
                                   const TM *src) {
    allocate(occa::getDevice(),
             d0, d1, d2, d3,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4,
                                   const TM *src) {
    allocate(occa::getDevice(),
             d0, d1, d2, d3, d4,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4, const udim_t d5,
                                   const TM *src) {
    allocate(occa::getDevice(),
             d0, d1, d2, d3, d4, d5,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0,
                                   const occa::memory src) {
    allocate(occa::getDevice(),
             d0,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1,
                                   const occa::memory src) {
    allocate(occa::getDevice(),
             d0, d1,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const occa::memory src) {
    allocate(occa::getDevice(),
             d0, d1, d2,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3,
                                   const occa::memory src) {
    allocate(occa::getDevice(),
             d0, d1, d2, d3,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4,
                                   const occa::memory src) {
    allocate(occa::getDevice(),
             d0, d1, d2, d3, d4,
             src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4, const udim_t d5,
                                   const occa::memory src) {
    allocate(occa::getDevice(),
             d0, d1, d2, d3, d4, d5,
             src);
  }

  //---[ allocate(device, ...) ]--------
  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_, const int dim_, const udim_t *d,
                                   const TM *src) {
    switch(dim_) {
    case 1: allocate(device_, d[0], src);                               break;
    case 2: allocate(device_, d[0], d[1], src);                         break;
    case 3: allocate(device_, d[0], d[1], d[2], src);                   break;
    case 4: allocate(device_, d[0], d[1], d[2], d[3], src);             break;
    case 5: allocate(device_, d[0], d[1], d[2], d[3], d[4], src);       break;
    case 6: allocate(device_, d[0], d[1], d[2], d[3], d[4], d[5], src); break;
    default:
      if (dim_ <= 0) {
        OCCA_FORCE_ERROR("Number of dimensions must be [1-6]");
      } else {
        OCCA_FORCE_ERROR("occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_, const int dim_, const udim_t *d,
                                   const occa::memory src) {
    switch(dim_) {
    case 1: allocate(device_, d[0], src);                               break;
    case 2: allocate(device_, d[0], d[1], src);                         break;
    case 3: allocate(device_, d[0], d[1], d[2], src);                   break;
    case 4: allocate(device_, d[0], d[1], d[2], d[3], src);             break;
    case 5: allocate(device_, d[0], d[1], d[2], d[3], d[4], src);       break;
    case 6: allocate(device_, d[0], d[1], d[2], d[3], d[4], d[5], src); break;
    default:
      if (dim_ <= 0) {
        OCCA_FORCE_ERROR("Number of dimensions must be [1-6]");
      } else {
        OCCA_FORCE_ERROR("occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0,
                                   const TM *src) {
    device = device_;
    reshape(d0);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1,
                                   const TM *src) {
    device = device_;
    reshape(d0, d1);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const TM *src) {
    device = device_;
    reshape(d0, d1, d2);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3,
                                   const TM *src) {
    device = device_;
    reshape(d0, d1, d2, d3);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4,
                                   const TM *src) {
    device = device_;
    reshape(d0, d1, d2, d3, d4);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4, const udim_t d5,
                                   const TM *src) {
    device = device_;
    reshape(d0, d1, d2, d3, d4, d5);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0,
                                   const occa::memory src) {
    device = device_;
    reshape(d0);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1,
                                   const occa::memory src) {
    device = device_;
    reshape(d0, d1);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const occa::memory src) {
    device = device_;
    reshape(d0, d1, d2);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3,
                                   const occa::memory src) {
    device = device_;
    reshape(d0, d1, d2, d3);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4,
                                   const occa::memory src) {
    device = device_;
    reshape(d0, d1, d2, d3, d4);
    allocate(src);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4, const udim_t d5,
                                   const occa::memory src) {
    device = device_;
    reshape(d0, d1, d2, d3, d4, d5);
    allocate(src);
  }

  //---[ reshape(...) ]-----------------
  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const int dim_, const udim_t *d) {
    switch(dim_) {
    case 1: reshape(d[0]);                               break;
    case 2: reshape(d[0], d[1]);                         break;
    case 3: reshape(d[0], d[1], d[2]);                   break;
    case 4: reshape(d[0], d[1], d[2], d[3]);             break;
    case 5: reshape(d[0], d[1], d[2], d[3], d[4]);       break;
    case 6: reshape(d[0], d[1], d[2], d[3], d[4], d[5]); break;
    default:
      if (dim_ <= 0) {
        OCCA_FORCE_ERROR("Number of dimensions must be [1-6]");
      } else {
        OCCA_FORCE_ERROR("occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const udim_t d0) {
    s_[0] = d0; s_[1] =  1; s_[2] =  1;
    s_[3] =  1; s_[4] =  1; s_[5] =  1;
    updateFS(6);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const udim_t d0, const udim_t d1) {
    s_[0] = d0; s_[1] = d1; s_[2] =  1;
    s_[3] =  1; s_[4] =  1; s_[5] =  1;
    updateFS(6);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const udim_t d0, const udim_t d1, const udim_t d2) {
    s_[0] = d0; s_[1] = d1; s_[2] = d2;
    s_[3] =  1; s_[4] =  1; s_[5] =  1;
    updateFS(6);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const udim_t d0, const udim_t d1, const udim_t d2,
                                  const udim_t d3) {
    s_[0] = d0; s_[1] = d1; s_[2] = d2;
    s_[3] = d3; s_[4] =  1; s_[5] =  1;
    updateFS(6);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const udim_t d0, const udim_t d1, const udim_t d2,
                                  const udim_t d3, const udim_t d4) {
    s_[0] = d0; s_[1] = d1; s_[2] = d2;
    s_[3] = d3; s_[4] = d4; s_[5] =  1;
    updateFS(6);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const udim_t d0, const udim_t d1, const udim_t d2,
                                  const udim_t d3, const udim_t d4, const udim_t d5) {
    s_[0] = d0; s_[1] = d1; s_[2] = d2;
    s_[3] = d3; s_[4] = d4; s_[5] = d5;
    updateFS(6);
  }

  //---[ reindex(...) ]-------------
  template <class TM, const int idxType>
  void array<TM,idxType>::updateFS(const int idxCount_) {
    idxCount = idxCount_;

    for (int i = 0; i < idxCount; ++i) {
      ks_[i] = s_[i];
    }
    if (idxType == occa::dynamic) {
      udim_t fs2[7];
      fs2[0] = 1;

      for (int i = 0; i < 6; ++i) {
        const int i2 = (sOrder_[i] + 1);
        fs2[i2] = s_[i];
      }
      for (int i = 1; i < 7; ++i) {
        fs2[i] *= fs2[i - 1];
      }
      for (int i = 0; i < 6; ++i) {
        fs_[i] = fs2[sOrder_[i]];
      }
    }
  }

  //  |---[ dynamic ]---------------
  template <class TM, const int idxType>
  void array<TM,idxType>::reindex(const int dim_, const int *o) {
    if (idxType == occa::dynamic) {
      switch(dim_) {
      case 1:                                                  break;
      case 2: reindex(o[0], o[1]);                         break;
      case 3: reindex(o[0], o[1], o[2]);                   break;
      case 4: reindex(o[0], o[1], o[2], o[3]);             break;
      case 5: reindex(o[0], o[1], o[2], o[3], o[4]);       break;
      case 6: reindex(o[0], o[1], o[2], o[3], o[4], o[5]); break;
      default:
        if (dim_ <= 0) {
          OCCA_FORCE_ERROR("Number of dimensions must be [1-6]");
        } else {
          OCCA_FORCE_ERROR("occa::array can only take up to 6 dimensions");
        }
      }
    } else {
      OCCA_FORCE_ERROR("Only occa::array<TM, occa::dynamic> can use reindex()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reindex(const int o0, const int o1) {
    if (idxType == occa::dynamic) {
      OCCA_ERROR("occa::array::reindex("
                 << o1 << ','
                 << o0 << ") has index out of bounds",
                 (0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1));

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] =  2;
      sOrder_[3] =  3; sOrder_[4] =  4; sOrder_[5] =  5;
      updateFS(2);
    } else {
      OCCA_FORCE_ERROR("Only occa::array<TM, occa::dynamic> can use reindex()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reindex(const int o0, const int o1, const int o2) {
    if (idxType == occa::dynamic) {
      OCCA_ERROR("occa::array::reindex("
                 << o0 << ','
                 << o1 << ','
                 << o2 << ") has index out of bounds",
                 (0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1) &&
                 (0 <= o2) && (o2 <= 1));

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] = o2;
      sOrder_[3] =  3; sOrder_[4] =  4; sOrder_[5] =  5;
      updateFS(3);
    } else {
      OCCA_FORCE_ERROR("Only occa::array<TM, occa::dynamic> can use reindex()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reindex(const int o0, const int o1, const int o2,
                                  const int o3) {

    if (idxType == occa::dynamic) {
      OCCA_ERROR("occa::array::reindex("
                 << o0 << ','
                 << o1 << ','
                 << o2 << ','
                 << o3 << ") has index out of bounds",
                 (0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1) &&
                 (0 <= o2) && (o2 <= 1) &&
                 (0 <= o3) && (o3 <= 1));

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] = o2;
      sOrder_[3] = o3; sOrder_[4] =  4; sOrder_[5] =  5;
      updateFS(4);
    } else {
      OCCA_FORCE_ERROR("Only occa::array<TM, occa::dynamic> can use reindex()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reindex(const int o0, const int o1, const int o2,
                                  const int o3, const int o4) {
    if (idxType == occa::dynamic) {
      OCCA_ERROR("occa::array::reindex("
                 << o0 << ','
                 << o1 << ','
                 << o2 << ','
                 << o3 << ','
                 << o4 << ") has index out of bounds",
                 (0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1) &&
                 (0 <= o2) && (o2 <= 1) &&
                 (0 <= o3) && (o3 <= 1) &&
                 (0 <= o4) && (o4 <= 1));

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] = o2;
      sOrder_[3] = o3; sOrder_[4] = o4; sOrder_[5] =  5;
      updateFS(5);
    } else {
      OCCA_FORCE_ERROR("Only occa::array<TM, occa::dynamic> can use reindex()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reindex(const int o0, const int o1, const int o2,
                                  const int o3, const int o4, const int o5) {
    if (idxType == occa::dynamic) {
      OCCA_ERROR("occa::array::reindex("
                 << o0 << ','
                 << o1 << ','
                 << o2 << ','
                 << o3 << ','
                 << o4 << ','
                 << o5 << ") has index out of bounds",
                 (0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1) &&
                 (0 <= o2) && (o2 <= 1) &&
                 (0 <= o3) && (o3 <= 1) &&
                 (0 <= o4) && (o4 <= 1) &&
                 (0 <= o5) && (o5 <= 1));

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] = o2;
      sOrder_[3] = o3; sOrder_[4] = o4; sOrder_[5] = o5;
      updateFS(6);
    } else {
      OCCA_FORCE_ERROR("Only occa::array<TM, occa::dynamic> can use reindex()");
    }
  }

  //---[ Access Operators ]-------------
  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator [] (const udim_t i0) {
    return ptr_[i0];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0) {
    return ptr_[i0];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0, const udim_t i1) {
    if (idxType == occa::fixed) {
      return ptr_[i0 + s_[0]*i1];
    }
    return ptr_[fs_[0]*i0 + fs_[1]*i1];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0, const udim_t i1, const udim_t i2) {
    if (idxType == occa::fixed) {
      return ptr_[i0 + s_[0]*(i1 + s_[1]*i2)];
    }
    return ptr_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0, const udim_t i1, const udim_t i2,
                                             const udim_t i3) {
    if (idxType == occa::fixed) {
      return ptr_[i0 + s_[0]*(i1 + s_[1]*(i2 + s_[2]*i3))];
    }
    return ptr_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2 + fs_[3]*i3];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0, const udim_t i1, const udim_t i2,
                                             const udim_t i3, const udim_t i4) {
    if (idxType == occa::fixed) {
      return ptr_[i0 + s_[0]*(i1 + s_[1]*(i2 + s_[2]*(i3 + s_[3]*i4)))];
    }
    return ptr_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2 + fs_[3]*i3 + fs_[4]*i4];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0, const udim_t i1, const udim_t i2,
                                             const udim_t i3, const udim_t i4, const udim_t i5) {
    if (idxType == occa::fixed) {
      return ptr_[i0 + s_[0]*(i1 + s_[1]*(i2 + s_[2]*(i3 + s_[3]*(i4 + s_[4]*i5))))];
    }
    return ptr_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2 + fs_[3]*i3 + fs_[4]*i4 + fs_[5]*i5];
  }
  //====================================

  //---[ Subarray ]---------------------
  template <class TM, const int idxType>
  inline array<TM,idxType> array<TM,idxType>::operator + (const udim_t offset) {
    array<TM,idxType> ret = *this;
    ret.memory_ += offset;
    ret.ptr_   += offset;
    ret.reshape(size() - offset);
    return ret;
  }
  //====================================

  //---[ Assignment Operators ]-------
  template <class TM, const int idxType>
  array<TM,idxType>& array<TM,idxType>::operator = (const TM value) {
    linalg::operator_eq<TM>(memory_, value);
    return *this;
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  bool array<TM,idxType>::operator == (const array<TM2,idxType2> &vec) {
    return (memory_ == vec.memory_);
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  bool array<TM,idxType>::operator != (const array<TM2,idxType2> &vec) {
    return (memory_ != vec.memory_);
  }

  template <class TM, const int idxType>
  array<TM,idxType>& array<TM,idxType>::operator += (const TM value) {
    linalg::operator_plus_eq<TM>(memory_, value);
    return *this;
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM,idxType>& array<TM,idxType>::operator += (const array<TM2,idxType2> &vec) {
    linalg::operator_plus_eq<TM2,TM>(vec.memory_, memory_);
    return *this;
  }

  template <class TM, const int idxType>
  array<TM,idxType>& array<TM,idxType>::operator -= (const TM value) {
    linalg::operator_sub_eq<TM>(memory_, value);
    return *this;
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM,idxType>& array<TM,idxType>::operator -= (const array<TM2,idxType2> &vec) {
    linalg::operator_sub_eq<TM2,TM>(vec.memory_, memory_);
    return *this;
  }

  template <class TM, const int idxType>
  array<TM,idxType>& array<TM,idxType>::operator *= (const TM value) {
    linalg::operator_mult_eq<TM>(memory_, value);
    return *this;
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM,idxType>& array<TM,idxType>::operator *= (const array<TM2,idxType2> &vec) {
    linalg::operator_mult_eq<TM2,TM>(vec.memory_, memory_);
    return *this;
  }

  template <class TM, const int idxType>
  array<TM,idxType>& array<TM,idxType>::operator /= (const TM value) {
    linalg::operator_div_eq<TM>(memory_, value);
    return *this;
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM,idxType>& array<TM,idxType>::operator /= (const array<TM2,idxType2> &vec) {
    linalg::operator_div_eq<TM2,TM>(vec.memory_, memory_);
    return *this;
  }
  //====================================

  //---[ Linear Algebra ]---------------
  template <class TM, const int idxType>
  TM array<TM,idxType>::l1Norm() {
    return linalg::l1Norm<TM,TM>(memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE>
  RETTYPE array<TM,idxType>::l1Norm() {
    return linalg::l1Norm<TM,RETTYPE>(memory_);
  }

  template <class TM, const int idxType>
  TM array<TM,idxType>::l2Norm() {
    return linalg::l2Norm<TM,TM>(memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE>
  RETTYPE array<TM,idxType>::l2Norm() {
    return linalg::l2Norm<TM,RETTYPE>(memory_);
  }

  template <class TM, const int idxType>
  TM array<TM,idxType>::lpNorm(const float p) {
    return linalg::lpNorm<TM,TM>(p, memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE>
  RETTYPE array<TM,idxType>::lpNorm(const float p) {
    return linalg::lpNorm<TM,RETTYPE>(p, memory_);
  }

  template <class TM, const int idxType>
  TM array<TM,idxType>::lInfNorm() {
    return linalg::lInfNorm<TM,TM>(memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE>
  RETTYPE array<TM,idxType>::lInfNorm() {
    return linalg::lInfNorm<TM,RETTYPE>(memory_);
  }

  template <class TM, const int idxType>
  TM array<TM,idxType>::max() {
    return linalg::max<TM,TM>(memory_);
  }

  template <class TM, const int idxType>
  TM array<TM,idxType>::min() {
    return linalg::min<TM,TM>(memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE, class TM2, const int idxType2>
  RETTYPE array<TM,idxType>::dot(const array<TM2, idxType2> &vec) {
    return linalg::dot<TM,TM2,RETTYPE>(memory_, vec.memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE, class TM2, const int idxType2>
  RETTYPE array<TM,idxType>::distance(const array<TM2, idxType2> &vec) {
    return linalg::distance<TM,TM2,RETTYPE>(memory_, vec.memory_);
  }

  template <class TM, const int idxType>
  template <class TYPE_A, class TM2, const int idxType2>
  array<TM,idxType>& array<TM,idxType>::sum(const TYPE_A alpha,
                                            const array<TM2, idxType2> &vec) {
    linalg::axpy<TYPE_A,TM2,TM>(alpha, vec.memory_, memory_);
    return *this;
  }
  //==================================

  //---[ Syncs ]------------------------
  template <class TM, const int idxType>
  void array<TM,idxType>::startManaging() {
    occa::startManaging(ptr_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::stopManaging() {
    occa::stopManaging(ptr_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::syncToDevice() {
    occa::syncToDevice(ptr_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::syncToHost() {
    occa::syncToHost(ptr_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::keepInDevice() {
    syncToDevice();
    stopManaging();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::keepInHost() {
    syncToHost();
    stopManaging();
  }

  template <class TM, const int idxType>
  bool array<TM,idxType>::needsSync() {
    return occa::needsSync(ptr_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::sync() {
    occa::sync(ptr_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::dontSync() {
    occa::dontSync(ptr_);
  }
}
