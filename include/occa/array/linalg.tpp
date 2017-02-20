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

namespace occa {
  namespace linalg {
    template <class TM>
    TM *hostReductionBuffer(const int size) {
      std::map<int, TM*> &bufferMap = hostBufferMap<TM>();
      TM *&buffer = bufferMap[size];
      if (!buffer) {
        buffer = new TM[size];
      }
      return buffer;
    }

    template <class TM>
    occa::memory deviceReductionBuffer(occa::device device,
                                       const int size) {

      hashedMemoryMap_t &bufferMap = deviceBufferMap<TM>();
      occa::memory &buffer = bufferMap[hash(device) ^ size];
      if (!buffer.isInitialized()) {
        buffer = device.malloc(size * sizeof(TM));
      }
      return buffer;
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
      dev.finish();
      deviceBuffer.copyTo(hostBuffer);
      return hostBuffer;
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE l1Norm(occa::memory vec) {
      static kernelBuilder builder(getKernelFile(),
                                   "l1Norm",
                                   "defines: {"
                                   "  VTYPE: '" + primitiveinfo<VTYPE>::name + "',"
                                   "  RETTYPE: '" + primitiveinfo<RETTYPE>::name + "',"
                                   "  CPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_INNER: 128,"
                                   "}");
      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = 0;
      for (int i = 0; i < 1024; ++i) {
        ret += partialReduction[i];
      }
      return ret;
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE l2Norm(occa::memory vec) {
      static kernelBuilder builder(getKernelFile(),
                                   "l2Norm",
                                   "defines: {"
                                   "  VTYPE: '" + primitiveinfo<VTYPE>::name + "',"
                                   "  RETTYPE: '" + primitiveinfo<RETTYPE>::name + "',"
                                   "  CPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_INNER: 128,"
                                   "}");
      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = 0;
      for (int i = 0; i < 1024; ++i) {
        ret += partialReduction[i];
      }
      return sqrt(ret);
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE lpNorm(const float p,
              occa::memory vec) {
      static kernelBuilder builder(getKernelFile(),
                                   "lpNorm",
                                   "defines: {"
                                   "  VTYPE: '" + primitiveinfo<VTYPE>::name + "',"
                                   "  RETTYPE: '" + primitiveinfo<RETTYPE>::name + "',"
                                   "  CPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_INNER: 128,"
                                   "}");

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
      return pow(ret, 1.0/p);
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE lInfNorm(occa::memory vec) {
      static kernelBuilder builder(getKernelFile(),
                                   "lInfNorm",
                                   "defines: {"
                                   "  VTYPE: '" + primitiveinfo<VTYPE>::name + "',"
                                   "  RETTYPE: '" + primitiveinfo<RETTYPE>::name + "',"
                                   "  CPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_INNER: 128,"
                                   "}");
      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = std::abs(partialReduction[0]);
      for (int i = 1; i < 1024; ++i) {
        const RETTYPE abs_i = std::abs(partialReduction[i]);
        if (ret < abs_i) {
          ret = abs_i;
        }
      }
      return ret;
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE max(occa::memory vec) {
      static kernelBuilder builder(getKernelFile(),
                                   "vecMax",
                                   "defines: {"
                                   "  VTYPE: '" + primitiveinfo<VTYPE>::name + "',"
                                   "  RETTYPE: '" + primitiveinfo<RETTYPE>::name + "',"
                                   "  CPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_INNER: 128,"
                                   "}");
      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = partialReduction[0];
      for (int i = 1; i < 1024; ++i) {
        if (ret < partialReduction[i]) {
          ret = partialReduction[i];
        }
      }
      return ret;
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE min(occa::memory vec) {
      static kernelBuilder builder(getKernelFile(),
                                   "vecMin",
                                   "defines: {"
                                   "  VTYPE: '" + primitiveinfo<VTYPE>::name + "',"
                                   "  RETTYPE: '" + primitiveinfo<RETTYPE>::name + "',"
                                   "  CPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_INNER: 128,"
                                   "}");
      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = partialReduction[0];
      for (int i = 1; i < 1024; ++i) {
        if (ret > partialReduction[i]) {
          ret = partialReduction[i];
        }
      }
      return ret;
    }

    template <class VTYPE, class RETTYPE>
    RETTYPE dot(occa::memory vec) {
      static kernelBuilder builder(getKernelFile(),
                                   "dot",
                                   "defines: {"
                                   "  VTYPE: '" + primitiveinfo<VTYPE>::name + "',"
                                   "  RETTYPE: '" + primitiveinfo<RETTYPE>::name + "',"
                                   "  CPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_OUTER: 1024,"
                                   "  GPU_DOT_INNER: 128,"
                                   "}");
      RETTYPE *partialReduction = reduce<VTYPE,RETTYPE>(vec, builder, 1024);
      RETTYPE ret = 0;
      for (int i = 0; i < 1024; ++i) {
        ret += partialReduction[i];
      }
      return ret;
    }
  }
}
