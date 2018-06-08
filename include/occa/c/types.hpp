/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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

#include <occa/base.hpp>
#include <occa/tools/sys.hpp>

#include <occa/c/types.h>

namespace occa {
  namespace c {
    namespace typeType {
      static const int none       = (1 << 0);

      static const int ptr        = (1 << 1);

      static const int bool_      = (1 << 2);

      static const int int8_      = (1 << 3);
      static const int uint8_     = (1 << 4);
      static const int int16_     = (1 << 5);
      static const int uint16_    = (1 << 6);
      static const int int32_     = (1 << 7);
      static const int uint32_    = (1 << 8);
      static const int int64_     = (1 << 9);
      static const int uint64_    = (1 << 10);
      static const int float_     = (1 << 11);
      static const int double_    = (1 << 12);

      static const int struct_    = (1 << 13);
      static const int string     = (1 << 14);

      static const int device     = (1 << 15);
      static const int kernel     = (1 << 16);
      static const int memory     = (1 << 17);

      static const int properties = (1 << 18);
    }

    occaType defaultOccaType();

    template <class TM>
    occaType newOccaType(const TM &value) {
      OCCA_FORCE_ERROR("Unable to handle type");
      return occaType();
    }

    occaType newOccaType(void *value);

    template <>
    occaType newOccaType(const bool &value);

    template <>
    occaType newOccaType(const int8_t &value);

    template <>
    occaType newOccaType(const uint8_t &value);

    template <>
    occaType newOccaType(const int16_t &value);

    template <>
    occaType newOccaType(const uint16_t &value);

    template <>
    occaType newOccaType(const int32_t &value);

    template <>
    occaType newOccaType(const uint32_t &value);

    template <>
    occaType newOccaType(const int64_t &value);

    template <>
    occaType newOccaType(const uint64_t &value);

    template <>
    occaType newOccaType(const float &value);

    template <>
    occaType newOccaType(const double &value);

    occaType newOccaType(occa::device device);
    occaType newOccaType(occa::kernel kernel);
    occaType newOccaType(occa::memory memory);
    occaType newOccaType(occa::properties &properties);
    occaStream newOccaType(occa::stream value);
    occaStreamTag newOccaType(occa::streamTag value);

    bool isDefault(occaType value);

    occa::device device(occaType value);
    occa::kernel kernel(occaType value);
    occa::memory memory(occaType value);
    occa::properties& properties(occaType value);

    occa::stream stream(occaStream value);
    occa::streamTag streamTag(occaStreamTag value);
  }
}
