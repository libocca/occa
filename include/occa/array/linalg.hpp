#ifndef OCCA_ARRAY_LINALG_HEADER
#define OCCA_ARRAY_LINALG_HEADER

#include <cmath>

#include <occa/defines.hpp>
#include <occa/core/base.hpp>
#include <occa/core/kernelBuilder.hpp>
#include <occa/tools/env.hpp>

namespace occa {
  namespace linalg {
    static const int usedTileSizeCount = 7;
    static const int usedTileSizes[7] = {
      32, 64, 128, 256, 512, 1024, 2048
    };

    template <class VTYPE_IN, class VTYPE_OUT>
    kernelBuilder makeAssignmentBuilder(const std::string &kernelName,
                                        const int tileSize);

    template <class VTYPE_IN, class VTYPE_OUT>
    kernelBuilderVector makeAssignmentBuilders(const std::string &kernelName);

    inline occa::kernel getTiledKernel(kernelBuilderVector &builders,
                                       occa::device dev,
                                       const int tileSize) {
      int i;
      for (i = 1; i < usedTileSizeCount; ++i) {
        if (usedTileSizes[i] > tileSize) {
          break;
        }
      }
      return builders[i - 1].build(dev);
    }

    template <class VTYPE, class RETTYPE>
    kernelBuilder makeLinalgBuilder(const std::string &kernelName);

    template <class VTYPE1, class VTYPE2, class RETTYPE>
    kernelBuilder makeLinalgBuilder(const std::string &kernelName);

    //---[ Assignment ]-----------------
    template <class VTYPE_OUT>
    void operator_eq(occa::memory vec,
                     const VTYPE_OUT value,
                     const int tileSize = 128);

    template <class VTYPE_OUT>
    void operator_plus_eq(occa::memory vec,
                          const VTYPE_OUT value,
                          const int tileSize = 128);

    template <class VTYPE_IN, class VTYPE_OUT>
    void operator_plus_eq(occa::memory in,
                          occa::memory out,
                          const int tileSize = 128);

    template <class VTYPE_OUT>
    void operator_sub_eq(occa::memory vec,
                         const VTYPE_OUT value,
                         const int tileSize = 128);

    template <class VTYPE_IN, class VTYPE_OUT>
    void operator_sub_eq(occa::memory in,
                         occa::memory out,
                         const int tileSize = 128);

    template <class VTYPE_OUT>
    void operator_mult_eq(occa::memory vec,
                          const VTYPE_OUT value,
                          const int tileSize = 128);

    template <class VTYPE_IN, class VTYPE_OUT>
    void operator_mult_eq(occa::memory in,
                          occa::memory out,
                          const int tileSize = 128);

    template <class VTYPE_OUT>
    void operator_div_eq(occa::memory vec,
                         const VTYPE_OUT value,
                         const int tileSize = 128);

    template <class VTYPE_IN, class VTYPE_OUT>
    void operator_div_eq(occa::memory in,
                         occa::memory out,
                         const int tileSize = 128);
    //==================================

    //---[ Linear Algebra ]-------------
    template <class TM>
    inline std::map<int, TM*>& hostBufferMap() {
      static std::map<int, TM*> bufferMap;
      return bufferMap;
    }

    inline hashedMemoryMap& deviceBufferMap() {
      static hashedMemoryMap bufferMap;
      return bufferMap;
    }

    template <class TM>
    TM *hostReductionBuffer(const int size);

    template <class TM>
    occa::memory deviceReductionBuffer(occa::device device,
                                       const int size);

    template <class VTYPE, class RETTYPE>
    RETTYPE* reduce(occa::memory vec,
                    occa::kernelBuilder &builder);

    template <class VTYPE, class RETTYPE>
    RETTYPE l1Norm(occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE l2Norm(occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE lpNorm(const float p, occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE lInfNorm(occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE max(occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE min(occa::memory vec);

    template <class VTYPE1, class VTYPE2, class RETTYPE>
    RETTYPE dot(occa::memory vec1, occa::memory vec2);

    template <class VTYPE1, class VTYPE2, class RETTYPE>
    RETTYPE distance(occa::memory vec1, occa::memory vec2);

    template <class VTYPE, class RETTYPE>
    RETTYPE sum(occa::memory vec);

    template <class TYPE_A, class VTYPE_X, class VTYPE_Y>
    void axpy(const TYPE_A &alpha,
              occa::memory x,
              occa::memory y,
              const int tileSize = 128);

    kernelBuilder customLinearMethod(const std::string &kernelName,
                                     const std::string &formula,
                                     const occa::properties &props = occa::properties());
    //==================================
  }
}

#include "linalg.tpp"

#endif
