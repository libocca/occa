/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  template <class TM, const int idxType>
  array<TM,idxType>::array() :
    data_(NULL) {

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

    data_ = v.data_;

    initSOrder(v.idxCount);

    for (int i = 0; i < idxCount; ++i) {
      ks_[i]     = v.ks_[i];
      s_[i]      = v.s_[i];
      sOrder_[i] = v.sOrder_[i];
    }

    if (idxType == occa::useIdxOrder) {
      updateFS(v.idxCount);
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::initSOrder(int idxCount_) {
    idxCount = idxCount_;

    if (idxType == occa::useIdxOrder) {
      for (int i = 0; i < 6; ++i) {
        sOrder_[i] = i;
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::free() {
    if (data_ == NULL) {
      return;
    }

    occa::free(data_);
    data_ = NULL;

    for (int i = 0; i < 6; ++i) {
      ks_[i]     = 0;
      s_[i]      = 0;
      sOrder_[i] = i;
    }
  }

  //---[ Info ]-------------------------
  template <class TM, const int idxType>
  std::string array<TM,idxType>::idxOrderStr() {
    if (idxType == occa::useIdxOrder) {
      std::string str(2*idxCount - 1, ',');
      for (int i = 0; i < idxCount; ++i) {
        str[2*i] = ('0' + sOrder_[idxCount - i - 1]);
      }
      return str;
    } else {
      OCCA_ERROR("Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()",
                 false);
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
    return cloneOn(occa::currentDevice(), copyOn);
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
    return cloneOn<TM2>(occa::currentDevice(), copyOn);
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM2,idxType2> array<TM,idxType>::cloneOn(occa::device device_, const int copyOn) {
    array<TM2,idxType2> clone_ = *this;

    clone_.allocate(device_, idxCount, s_);
    occa::memcpy(clone_.data_, data_, bytes());

    return clone_;
  }

  //---[ array(...) ]------------------
  template <class TM, const int idxType>
  array<TM,idxType>::array(const int dim_, const udim_t *d) {
    initSOrder(dim_);

    switch(dim_) {
    case 1: allocate(d[0]);                               break;
    case 2: allocate(d[0], d[1]);                         break;
    case 3: allocate(d[0], d[1], d[2]);                   break;
    case 4: allocate(d[0], d[1], d[2], d[3]);             break;
    case 5: allocate(d[0], d[1], d[2], d[3], d[4]);       break;
    case 6: allocate(d[0], d[1], d[2], d[3], d[4], d[5]); break;
    default:
      if (dim_ <= 0) {
        OCCA_ERROR("Number of dimensions must be [1-6]",
                   false);
      } else {
        OCCA_ERROR("occa::array can only take up to 6 dimensions",
                   false);
      }
    }
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0) {
    initSOrder(1);
    allocate(d0);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1) {
    initSOrder(2);
    allocate(d0, d1);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2) {
    initSOrder(3);
    allocate(d0, d1, d2);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3) {
    initSOrder(4);
    allocate(d0, d1, d2, d3);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4) {
    initSOrder(5);
    allocate(d0, d1, d2, d3, d4);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4, const udim_t d5) {
    initSOrder(6);
    allocate(d0, d1, d2, d3, d4, d5);
  }

  //---[ array(device, ...) ]----------
  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_, const int dim_, const udim_t *d) {
    switch(dim_) {
    case 1: allocate(device_, d[0]);                               break;
    case 2: allocate(device_, d[0], d[1]);                         break;
    case 3: allocate(device_, d[0], d[1], d[2]);                   break;
    case 4: allocate(device_, d[0], d[1], d[2], d[3]);             break;
    case 5: allocate(device_, d[0], d[1], d[2], d[3], d[4]);       break;
    case 6: allocate(device_, d[0], d[1], d[2], d[3], d[4], d[5]); break;
    default:
      if (dim_ <= 0) {
        OCCA_ERROR("Number of dimensions must be [1-6]",
                   false);
      } else {
        OCCA_ERROR("occa::array can only take up to 6 dimensions",
                   false);
      }
    }
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0) {
    initSOrder(1);
    allocate(device_,
             d0);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1) {
    initSOrder(2);
    allocate(device_,
             d0, d1);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2) {
    initSOrder(3);
    allocate(device_,
             d0, d1, d2);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3) {
    initSOrder(4);
    allocate(device_,
             d0, d1, d2, d3);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4) {
    initSOrder(5);
    allocate(device_,
             d0, d1, d2, d3, d4);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const udim_t d0, const udim_t d1, const udim_t d2,
                           const udim_t d3, const udim_t d4, const udim_t d5) {
    initSOrder(6);
    allocate(device_,
             d0, d1, d2, d3, d4, d5);
  }

  //---[ allocate(...) ]----------------
  template <class TM, const int idxType>
  void array<TM,idxType>::allocate() {
    data_   = (TM*) device.uvaAlloc(bytes());
    memory_ = occa::memory(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const int dim_, const udim_t *d) {
    switch(dim_) {
    case 1: allocate(d[0]);                               break;
    case 2: allocate(d[0], d[1]);                         break;
    case 3: allocate(d[0], d[1], d[2]);                   break;
    case 4: allocate(d[0], d[1], d[2], d[3]);             break;
    case 5: allocate(d[0], d[1], d[2], d[3], d[4]);       break;
    case 6: allocate(d[0], d[1], d[2], d[3], d[4], d[5]); break;
    default:
      if (dim_ <= 0) {
        OCCA_ERROR("Number of dimensions must be [1-6]",
                   false);
      } else {
        OCCA_ERROR("occa::array can only take up to 6 dimensions",
                   false);
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0) {
    allocate(occa::currentDevice(),
             d0);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1) {
    allocate(occa::currentDevice(),
             d0, d1);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2) {
    allocate(occa::currentDevice(),
             d0, d1, d2);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3) {
    allocate(occa::currentDevice(),
             d0, d1, d2, d3);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4) {
    allocate(occa::currentDevice(),
             d0, d1, d2, d3, d4);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4, const udim_t d5) {
    allocate(occa::currentDevice(),
             d0, d1, d2, d3, d4, d5);
  }

  //---[ allocate(device, ...) ]--------
  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_, const int dim_, const udim_t *d) {
    switch(dim_) {
    case 1: allocate(device_, d[0]);                               break;
    case 2: allocate(device_, d[0], d[1]);                         break;
    case 3: allocate(device_, d[0], d[1], d[2]);                   break;
    case 4: allocate(device_, d[0], d[1], d[2], d[3]);             break;
    case 5: allocate(device_, d[0], d[1], d[2], d[3], d[4]);       break;
    case 6: allocate(device_, d[0], d[1], d[2], d[3], d[4], d[5]); break;
    default:
      if (dim_ <= 0) {
        OCCA_ERROR("Number of dimensions must be [1-6]",
                   false);
      } else {
        OCCA_ERROR("occa::array can only take up to 6 dimensions",
                   false);
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0) {
    device = device_;
    reshape(d0);
    allocate();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1) {
    device = device_;
    reshape(d0, d1);
    allocate();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2) {
    device = device_;
    reshape(d0, d1, d2);
    allocate();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3) {
    device = device_;
    reshape(d0, d1, d2, d3);
    allocate();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4) {
    device = device_;
    reshape(d0, d1, d2, d3, d4);
    allocate();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const udim_t d0, const udim_t d1, const udim_t d2,
                                   const udim_t d3, const udim_t d4, const udim_t d5) {
    device = device_;
    reshape(d0, d1, d2, d3, d4, d5);
    allocate();
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
        OCCA_ERROR("Number of dimensions must be [1-6]",
                   false);
      } else {
        OCCA_ERROR("occa::array can only take up to 6 dimensions",
                   false);
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

  //---[ setIdxOrder(...) ]-------------
  template <class TM, const int idxType>
  void array<TM,idxType>::updateFS(const int idxCount_) {
    idxCount = idxCount_;

    for (int i = 0; i < idxCount; ++i) {
      ks_[i] = s_[i];
    }
    if (idxType == occa::useIdxOrder) {
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

  //  |---[ useIdxOrder ]---------------
  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int dim_, const int *o) {
    if (idxType == occa::useIdxOrder) {
      switch(dim_) {
      case 1:                                                  break;
      case 2: setIdxOrder(o[0], o[1]);                         break;
      case 3: setIdxOrder(o[0], o[1], o[2]);                   break;
      case 4: setIdxOrder(o[0], o[1], o[2], o[3]);             break;
      case 5: setIdxOrder(o[0], o[1], o[2], o[3], o[4]);       break;
      case 6: setIdxOrder(o[0], o[1], o[2], o[3], o[4], o[5]); break;
      default:
        if (dim_ <= 0) {
          OCCA_ERROR("Number of dimensions must be [1-6]",
                     false);
        } else {
          OCCA_ERROR("occa::array can only take up to 6 dimensions",
                     false);
        }
      }
    } else {
      OCCA_ERROR("Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()",
                 false);
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const std::string &default_,
                                      const std::string &given) {

    const int dim_ = (int) default_.size();
    int o[6];

    OCCA_ERROR("occa::array::setIdxOrder(default, given) must have matching sized strings of size [1-6]",
               (dim_ == ((int) given.size())) &&
               (1 <= dim_) && (dim_ <= 6));

    for (int j = 0; j < dim_; ++j) {
      o[j] = -1;
    }
    for (int j_ = (dim_ - 1); 0 <= j_; --j_) {
      const int j = (dim_ - j_ - 1);
      const char C = default_[j_];

      for (int i_ = (dim_ - 1); 0 <= i_; --i_) {
        const int i = (dim_ - i_ - 1);

        if (C == given[i_]) {
          OCCA_ERROR("occa::array::setIdxOrder(default, given) must have strings with unique characters",
                     o[j] == -1);

          o[j] = i;
          break;
        }
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int o0, const int o1) {
    if (idxType == occa::useIdxOrder) {
      OCCA_ERROR("occa::array::setIdxOrder("
                 << o1 << ','
                 << o0 << ") has index out of bounds",
                 (0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1));

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] =  2;
      sOrder_[3] =  3; sOrder_[4] =  4; sOrder_[5] =  5;
      updateFS(2);
    } else {
      OCCA_ERROR(false,
                 "Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int o0, const int o1, const int o2) {
    if (idxType == occa::useIdxOrder) {
      OCCA_ERROR("occa::array::setIdxOrder("
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
      OCCA_ERROR("Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()",
                 false);
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int o0, const int o1, const int o2,
                                      const int o3) {

    if (idxType == occa::useIdxOrder) {
      OCCA_ERROR("occa::array::setIdxOrder("
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
      OCCA_ERROR("Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()",
                 false);
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int o0, const int o1, const int o2,
                                      const int o3, const int o4) {
    if (idxType == occa::useIdxOrder) {
      OCCA_ERROR("occa::array::setIdxOrder("
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
      OCCA_ERROR("Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()",
                 false);
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int o0, const int o1, const int o2,
                                      const int o3, const int o4, const int o5) {
    if (idxType == occa::useIdxOrder) {
      OCCA_ERROR("occa::array::setIdxOrder("
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
      OCCA_ERROR("Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()",
                 false);
    }
  }

  //---[ Operators ]--------------------
  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator [] (const udim_t i0) {
    return data_[i0];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0) {
    return data_[i0];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0, const udim_t i1) {
    if (idxType == occa::dontUseIdxOrder) {
      return data_[i0 + s_[0]*i1];
    }
    return data_[fs_[0]*i0 + fs_[1]*i1];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0, const udim_t i1, const udim_t i2) {
    if (idxType == occa::dontUseIdxOrder) {
      return data_[i0 + s_[0]*(i1 + s_[1]*i2)];
    }
    return data_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0, const udim_t i1, const udim_t i2,
                                             const udim_t i3) {
    if (idxType == occa::dontUseIdxOrder) {
      return data_[i0 + s_[0]*(i1 + s_[1]*(i2 + s_[2]*i3))];
    }
    return data_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2 + fs_[3]*i3];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0, const udim_t i1, const udim_t i2,
                                             const udim_t i3, const udim_t i4) {
    if (idxType == occa::dontUseIdxOrder) {
      return data_[i0 + s_[0]*(i1 + s_[1]*(i2 + s_[2]*(i3 + s_[3]*i4)))];
    }
    return data_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2 + fs_[3]*i3 + fs_[4]*i4];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const udim_t i0, const udim_t i1, const udim_t i2,
                                             const udim_t i3, const udim_t i4, const udim_t i5) {
    if (idxType == occa::dontUseIdxOrder) {
      return data_[i0 + s_[0]*(i1 + s_[1]*(i2 + s_[2]*(i3 + s_[3]*(i4 + s_[4]*i5))))];
    }
    return data_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2 + fs_[3]*i3 + fs_[4]*i4 + fs_[5]*i5];
  }
  //====================================

  //---[ Assignment Operators ]-------
  template <class TM, const int idxType>
  array<TM,idxType>& array<TM,idxType>::operator = (const TM value) {
    linalg::operator_eq<TM>(memory_, value);
    return *this;
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
  template <class RETTYPE>
  RETTYPE array<TM,idxType>::l1Norm() {
    return linalg::l1Norm<TM,RETTYPE>(memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE>
  RETTYPE array<TM,idxType>::l2Norm() {
    return linalg::l2Norm<TM,RETTYPE>(memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE>
  RETTYPE array<TM,idxType>::lpNorm(const float p) {
    return linalg::lpNorm<TM,RETTYPE>(p, memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE>
  RETTYPE array<TM,idxType>::lInfNorm() {
    return linalg::lInfNorm<TM,RETTYPE>(memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE>
  RETTYPE array<TM,idxType>::max() {
    return linalg::max<TM,RETTYPE>(memory_);
  }

  template <class TM, const int idxType>
  template <class RETTYPE>
  RETTYPE array<TM,idxType>::min() {
    return linalg::min<TM,RETTYPE>(memory_);
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
    occa::startManaging(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::stopManaging() {
    occa::stopManaging(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::syncToDevice() {
    occa::syncToDevice(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::syncFromDevice() {
    occa::syncFromDevice(data_);
  }

  template <class TM, const int idxType>
  bool array<TM,idxType>::needsSync() {
    return occa::needsSync(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::sync() {
    occa::sync(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::dontSync() {
    occa::dontSync(data_);
  }
}
