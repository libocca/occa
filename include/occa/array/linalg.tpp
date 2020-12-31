namespace occa {
  namespace linalg {
    template <class VTYPE_IN, class VTYPE_OUT>
    kernelBuilder makeAssignmentBuilder(const std::string &kernelName,
                                        const int tileSize) {
      return kernelBuilder::fromFile(env::OCCA_DIR + "include/occa/array/kernels/assignment.okl",
                                     kernelName,
                                     "defines: {"
                                     "  VTYPE_IN: '"  + primitiveinfo<VTYPE_IN>::name  + "',"
                                     "  VTYPE_OUT: '" + primitiveinfo<VTYPE_OUT>::name + "',"
                                     "  TILESIZE: '"  + std::to_string(tileSize) + "',"
                                     "}");
    }

    template <class VTYPE_IN, class VTYPE_OUT>
    kernelBuilderVector makeAssignmentBuilders(const std::string &kernelName) {
      kernelBuilderVector builders;
      for (int i = 0; i < usedTileSizeCount; ++i) {
        builders.push_back(makeAssignmentBuilder<VTYPE_IN,VTYPE_OUT>(kernelName,
                                                                     usedTileSizes[i]));
      }
      return builders;
    }

    template <class VTYPE, class RETTYPE>
    kernelBuilder makeLinalgBuilder(const std::string &kernelName) {
      return kernelBuilder::fromFile(env::OCCA_DIR + "include/occa/array/kernels/linalg.okl",
                                     kernelName,
                                     "defines: {"
                                     "  VTYPE: '" + primitiveinfo<VTYPE>::name + "',"
                                     "  VTYPE2: '" + primitiveinfo<VTYPE>::name + "',"
                                     "  RETTYPE: '" + primitiveinfo<RETTYPE>::name + "',"
                                     "  CPU_DOT_OUTER: 1024,"
                                     "  GPU_DOT_OUTER: 1024,"
                                     "  GPU_DOT_INNER: 128,"
                                     "}");
    }

    template <class VTYPE1, class VTYPE2, class RETTYPE>
    kernelBuilder makeLinalgBuilder(const std::string &kernelName) {
      return kernelBuilder::fromFile(env::OCCA_DIR + "include/occa/array/kernels/linalg.okl",
                                     kernelName,
                                     "defines: {"
                                     "  VTYPE: '"   + primitiveinfo<VTYPE1>::name  + "',"
                                     "  VTYPE2: '"  + primitiveinfo<VTYPE2>::name  + "',"
                                     "  RETTYPE: '" + primitiveinfo<RETTYPE>::name + "',"
                                     "  CPU_DOT_OUTER: 1024,"
                                     "  GPU_DOT_OUTER: 1024,"
                                     "  GPU_DOT_INNER: 128,"
                                     "}");
    }

    //---[ Assignment ]-----------------
    template <class VTYPE_OUT>
    void operator_eq(occa::memory vec,
                     const VTYPE_OUT value,
                     const int tileSize) {
      static kernelBuilderVector builders =
        makeAssignmentBuilders<VTYPE_OUT,VTYPE_OUT>("eq_const");

      const int entries = vec.size() / sizeof(VTYPE_OUT);
      getTiledKernel(builders,
                     vec.getDevice(),
                     tileSize)(entries, value, vec);
    }

    template <class VTYPE_OUT>
    void operator_plus_eq(occa::memory vec,
                          const VTYPE_OUT value,
                          const int tileSize) {
      static kernelBuilderVector builders =
        makeAssignmentBuilders<VTYPE_OUT,VTYPE_OUT>("plus_eq_const");

      const int entries = vec.size() / sizeof(VTYPE_OUT);
      getTiledKernel(builders,
                     vec.getDevice(),
                     tileSize)(entries, value, vec);
    }

    template <class VTYPE_IN, class VTYPE_OUT>
    void operator_plus_eq(occa::memory in,
                          occa::memory out,
                          const int tileSize) {
      static kernelBuilderVector builders =
        makeAssignmentBuilders<VTYPE_IN,VTYPE_OUT>("plus_eq");

      const int entries = in.size() / sizeof(VTYPE_OUT);
      getTiledKernel(builders,
                     in.getDevice(),
                     tileSize)(entries, in, out);
    }

    template <class VTYPE_OUT>
    void operator_sub_eq(occa::memory vec,
                         const VTYPE_OUT value,
                         const int tileSize) {
      static kernelBuilderVector builders =
        makeAssignmentBuilders<VTYPE_OUT,VTYPE_OUT>("sub_eq_const");

      const int entries = vec.size() / sizeof(VTYPE_OUT);
      getTiledKernel(builders,
                     vec.getDevice(),
                     tileSize)(entries, value, vec);
    }

    template <class VTYPE_IN, class VTYPE_OUT>
    void operator_sub_eq(occa::memory in,
                         occa::memory out,
                         const int tileSize) {
      static kernelBuilderVector builders =
        makeAssignmentBuilders<VTYPE_IN,VTYPE_OUT>("sub_eq");

      const int entries = in.size() / sizeof(VTYPE_OUT);
      getTiledKernel(builders,
                     in.getDevice(),
                     tileSize)(entries, in, out);
    }

    template <class VTYPE_OUT>
    void operator_mult_eq(occa::memory vec,
                          const VTYPE_OUT value,
                          const int tileSize) {
      static kernelBuilderVector builders =
        makeAssignmentBuilders<VTYPE_OUT,VTYPE_OUT>("mult_eq_const");

      const int entries = vec.size() / sizeof(VTYPE_OUT);
      getTiledKernel(builders,
                     vec.getDevice(),
                     tileSize)(entries, value, vec);
    }

    template <class VTYPE_IN, class VTYPE_OUT>
    void operator_mult_eq(occa::memory in,
                          occa::memory out,
                          const int tileSize) {
      static kernelBuilderVector builders =
        makeAssignmentBuilders<VTYPE_IN,VTYPE_OUT>("mult_eq");

      const int entries = in.size() / sizeof(VTYPE_OUT);
      getTiledKernel(builders,
                     in.getDevice(),
                     tileSize)(entries, in, out);
    }

    template <class VTYPE_OUT>
    void operator_div_eq(occa::memory vec,
                         const VTYPE_OUT value,
                         const int tileSize) {
      static kernelBuilderVector builders =
        makeAssignmentBuilders<VTYPE_OUT,VTYPE_OUT>("div_eq_const");

      const int entries = vec.size() / sizeof(VTYPE_OUT);
      getTiledKernel(builders,
                     vec.getDevice(),
                     tileSize)(entries, value, vec);
    }

    template <class VTYPE_IN, class VTYPE_OUT>
    void operator_div_eq(occa::memory in,
                         occa::memory out,
                         const int tileSize) {
      static kernelBuilderVector builders =
        makeAssignmentBuilders<VTYPE_IN,VTYPE_OUT>("div_eq");

      const int entries = in.size() / sizeof(VTYPE_OUT);
      getTiledKernel(builders,
                     in.getDevice(),
                     tileSize)(entries, in, out);
    }
    //==================================

    //---[ Linear Algebra ]-------------
    template <class TM>
    TM *hostReductionBuffer(const int size) {
      return new TM[size];
    }

    template <class TM>
    occa::memory deviceReductionBuffer(occa::device device,
                                       const int size) {
      memory &mapBuffer = deviceBufferMap()[hash(device)];
      const size_t bytes = size * sizeof(TM);
      // Not initialized or too small
      if (!mapBuffer.isInitialized() ||
          (mapBuffer.size() < bytes)) {
        mapBuffer = device.malloc(bytes);
        return mapBuffer;
      }
      if (mapBuffer.size() == bytes) {
        return mapBuffer;
      }
      return mapBuffer.slice(0, bytes);
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE* reduce(occa::memory vec,
                    occa::kernelBuilder &builder,
                    const int bufferSize) {

      device dev = vec.getDevice();
      RETTYPE *hostBuffer = hostReductionBuffer<RETTYPE>(bufferSize);
      memory deviceBuffer = deviceReductionBuffer<RETTYPE>(dev, bufferSize);
      const int entries = vec.size() / sizeof(VTYPE);
      builder.build(dev)(entries,
                         vec,
                         deviceBuffer);
      deviceBuffer.copyTo(hostBuffer);
      return hostBuffer;
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE l1Norm(occa::memory vec) {
      static kernelBuilder builder =
        makeLinalgBuilder<VTYPE, RETTYPE>("l1Norm");

      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = 0;
      for (int i = 0; i < 1024; ++i) {
        ret += partialReduction[i];
      }
      delete [] partialReduction;
      return ret;
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE l2Norm(occa::memory vec) {
      static kernelBuilder builder =
        makeLinalgBuilder<VTYPE, RETTYPE>("l2Norm");

      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = 0;
      for (int i = 0; i < 1024; ++i) {
        ret += partialReduction[i];
      }
      delete [] partialReduction;
      return sqrt(ret);
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE lpNorm(const float p,
                   occa::memory vec) {
      static kernelBuilder builder =
        makeLinalgBuilder<VTYPE, RETTYPE>("lpNorm");


      device dev = vec.getDevice();
      const int bufferSize = 1024;
      RETTYPE *hostBuffer = hostReductionBuffer<RETTYPE>(bufferSize);
      memory deviceBuffer = deviceReductionBuffer<RETTYPE>(dev, bufferSize);
      const int entries = vec.size() / sizeof(VTYPE);
      builder.build(dev)(entries,
                         p,
                         vec,
                         deviceBuffer);
      dev.finish();
      deviceBuffer.copyTo(hostBuffer);
      RETTYPE ret = 0;
      for (int i = 0; i < 1024; ++i) {
        ret += hostBuffer[i];
      }
      delete [] hostBuffer;
      return pow(ret, 1.0/p);
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE lInfNorm(occa::memory vec) {
      static kernelBuilder builder =
        makeLinalgBuilder<VTYPE, RETTYPE>("lInfNorm");

      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = partialReduction[0];
      for (int i = 1; i < 1024; ++i) {
        const RETTYPE abs_i = partialReduction[i];
        if (ret < abs_i) {
          ret = abs_i;
        }
      }
      delete [] partialReduction;
      return ret;
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE max(occa::memory vec) {
      static kernelBuilder builder =
        makeLinalgBuilder<VTYPE, RETTYPE>("vecMax");

      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = partialReduction[0];
      for (int i = 1; i < 1024; ++i) {
        if (ret < partialReduction[i]) {
          ret = partialReduction[i];
        }
      }
      delete [] partialReduction;
      return ret;
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE min(occa::memory vec) {
      static kernelBuilder builder =
        makeLinalgBuilder<VTYPE, RETTYPE>("vecMin");

      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = partialReduction[0];
      for (int i = 1; i < 1024; ++i) {
        if (ret > partialReduction[i]) {
          ret = partialReduction[i];
        }
      }
      delete [] partialReduction;
      return ret;
    }

    template <class VTYPE1, class VTYPE2, class RETTYPE>
    RETTYPE dot(occa::memory vec1, occa::memory vec2) {
      static kernelBuilder builder =
        makeLinalgBuilder<VTYPE1, VTYPE2, RETTYPE>("dot");

      OCCA_ERROR("Vectors must be in the same device",
                 vec1.getDevice() == vec2.getDevice());

      device dev = vec1.getDevice();
      const int bufferSize = 1024;
      RETTYPE *hostBuffer = hostReductionBuffer<RETTYPE>(bufferSize);
      memory deviceBuffer = deviceReductionBuffer<RETTYPE>(dev, bufferSize);
      const int entries = vec1.size() / sizeof(VTYPE1);
      builder.build(dev)(entries,
                         vec1,
                         vec2,
                         deviceBuffer);
      dev.finish();
      deviceBuffer.copyTo(hostBuffer);
      RETTYPE ret = 0;
      for (int i = 0; i < 1024; ++i) {
        ret += hostBuffer[i];
      }
      delete [] hostBuffer;
      return ret;
    }

    template <class VTYPE1, class VTYPE2, class RETTYPE>
    RETTYPE distance(occa::memory vec1, occa::memory vec2) {
      static kernelBuilder builder =
        makeLinalgBuilder<VTYPE1, VTYPE2, RETTYPE>("distance");

      OCCA_ERROR("Vectors must be in the same device",
                 vec1.getDevice() == vec2.getDevice());

      device dev = vec1.getDevice();
      const int bufferSize = 1024;
      RETTYPE *hostBuffer = hostReductionBuffer<RETTYPE>(bufferSize);
      memory deviceBuffer = deviceReductionBuffer<RETTYPE>(dev, bufferSize);
      const int entries = vec1.size() / sizeof(VTYPE1);
      builder.build(dev)(entries,
                         vec1,
                         vec2,
                         deviceBuffer);
      dev.finish();
      deviceBuffer.copyTo(hostBuffer);
      RETTYPE ret = 0;
      for (int i = 0; i < 1024; ++i) {
        ret += hostBuffer[i];
      }
      delete [] hostBuffer;
      return sqrt(ret);
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE sum(occa::memory vec) {
      static kernelBuilder builder =
        makeLinalgBuilder<VTYPE, RETTYPE>("sum");

      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = 0;
      for (int i = 0; i < 1024; ++i) {
        ret += partialReduction[i];
      }
      delete [] partialReduction;
      return ret;
    }

    template <class TYPE_A, class VTYPE_X, class VTYPE_Y>
    void axpy(const TYPE_A &alpha,
              occa::memory x,
              occa::memory y,
              const int tileSize) {

      static kernelBuilderVector builders;
      if (!builders.size()) {
        for (int i = 0; i < usedTileSizeCount; ++i) {
          kernelBuilder kerb =
            customLinearMethod("axpy",
                               "v0[i] += c0 * v1[i];",
                               "defines: {"
                               "  CTYPE0: '" + primitiveinfo<TYPE_A>::name + "',"
                               "  VTYPE0: '" + primitiveinfo<VTYPE_Y>::name + "',"
                               "  VTYPE1: '" + primitiveinfo<VTYPE_X>::name + "',"
                               "  TILESIZE: '" + std::to_string(usedTileSizes[i]) + "',"
                               "}");
          builders.push_back(kerb);
        }
      }

      const int entries = y.size() / sizeof(VTYPE_Y);
      getTiledKernel(builders,
                     y.getDevice(),
                     tileSize)(entries, alpha, y, x);
    }
    //==================================
  }
}
