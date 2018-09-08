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

#include <occa/core/base.hpp>
#include <occa/tools/sys.hpp>

#include <occa/c/types.h>

namespace occa {
  namespace c {
    namespace typeType {
      static const int undefined  = 0;
      static const int default_   = 1;

      static const int ptr        = 2;

      static const int bool_      = 3;

      static const int int8_      = 4;
      static const int uint8_     = 5;
      static const int int16_     = 6;
      static const int uint16_    = 7;
      static const int int32_     = 8;
      static const int uint32_    = 9;
      static const int int64_     = 10;
      static const int uint64_    = 11;
      static const int float_     = 12;
      static const int double_    = 13;

      static const int struct_    = 14;
      static const int string     = 15;

      static const int device     = 16;
      static const int kernel     = 17;
      static const int memory     = 18;
      static const int stream     = 19;
      static const int streamTag  = 20;

      static const int json       = 21;
      static const int properties = 22;
    }

    occaType defaultOccaType();

    // Private API:
    //   Compile error if template specialization doesn't exist
    template <class TM>
    occaType newOccaType(const TM &value);

    occaType newOccaType(void *value);

    template <>
    occaType newOccaType(const occa::primitive &value);

    occaType newOccaType(const occa::primitive &value,
                         const int type);

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
    occaType newOccaType(occa::stream stream);
    occaType newOccaType(occa::streamTag streamTag);

    occaType newOccaType(const json &json,
                         const bool needsFree);

    occaType newOccaType(const occa::properties &properties,
                         const bool needsFree);

    bool isDefault(occaType value);

    occa::device device(occaType value);
    occa::kernel kernel(occaType value);
    occa::memory memory(occaType value);
    occa::stream stream(occaType value);
    occa::streamTag streamTag(occaType value);

    occa::kernelArg kernelArg(occaType value);

    occa::primitive primitive(occaType value);
    occa::primitive primitive(occaType value,
                              const int type);

    occa::json& json(occaType value);
    occa::json inferJson(occaType value);

    occa::properties& properties(occaType value);
    const occa::properties& constProperties(occaType value);
  }
}
