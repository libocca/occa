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

#if OCCA_MPI_ENABLED
#  ifndef OCCA_MPI_HEADER
#  define OCCA_MPI_HEADER

#include <mpi.h>
#include <vector>

#include "occa/defines.hpp"
#include "occa/tools/json.hpp"

namespace occa {
  namespace mpi {
    static const int defaultMessageID = 15331;

    class buffer_t {
    public:
      static int size;
      char *ptr;

      buffer_t();
      ~buffer_t();
    };

    int id();
    int size();
    char* getBuffer();

    //---[ Types ]----------------------
    template <class TM>
    MPI_Datatype type() {
      return MPI_BYTE;
    }
    template <>
    MPI_Datatype type<bool>();
    template <>
    MPI_Datatype type<uint8_t>();
    template <>
    MPI_Datatype type<int8_t>();
    template <>
    MPI_Datatype type<uint16_t>();
    template <>
    MPI_Datatype type<int16_t>();
    template <>
    MPI_Datatype type<uint32_t>();
    template <>
    MPI_Datatype type<int32_t>();
    template <>
    MPI_Datatype type<uint64_t>();
    template <>
    MPI_Datatype type<int64_t>();
    template <>
    MPI_Datatype type<float>();
    template <>
    MPI_Datatype type<double>();
    //==================================

    //---[ Tag ]------------------------
    class tag {
    public:
      MPI_Request mpiRequest;
      bool initialized;
      bool done;

      tag();
      tag(MPI_Request mpiRequest_);

      bool isInitialized();
      void wait();
      void updateStatus();
    };
    //==================================

    //---[ Tags ]-----------------------
    class tags {
    public:
      std::vector<tag> tags_;

      tags();

      int size();
      void wait();
      void updateStatus();

      tag operator [] (const int index);

      tags& operator += (const tag &tag_);
    };
    //==================================

    //---[ Methods ]--------------------
    template <class TM>
    tag send(const int receiverID,
             const occa::memory &data,
             const dim_t entries = -1,
             const int messageID = defaultMessageID);

    template <class TM>
    tag get(const int senderID,
            occa::memory &data,
            const dim_t entries = -1,
            const int messageID = defaultMessageID);
    //==================================
  }
}

#  endif
#endif
