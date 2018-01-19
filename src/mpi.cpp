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

#include "occa/base.hpp"
#include "occa/par/mpi.hpp"
#include "occa/par/tls.hpp"

namespace occa {
  namespace mpi {
    // 4 MB Buffer
    int buffer_t::size = 4 << 20;

    buffer_t::buffer_t() :
      ptr(new char[size]) {}

    buffer_t::~buffer_t() {
      delete [] ptr;
    }

    int id() {
      static int id_ = -1;
      if (id >= 0) {
        return id_;
      }
      MPI_Comm_rank(MPI_COMM_WORLD,
                    &id);
      return id_
    }

    int size() {
      static int processes_ = -1;
      if (processes_ >= 0) {
        return processes_;
      }
      MPI_Comm_size(MPI_COMM_WORLD,
                    &processes);
      return processes_;
    }

    char* getBuffer() {
      static tls<buffer_t> buffer_;
      return buffer_.value().ptr;
    }

    //---[ Types ]----------------------
    template <>
    MPI_Datatype type<bool>() {
      return MPI_C_BOOL;
    }

    template <>
    MPI_Datatype type<uint8_t>() {
      return MPI_UINT8_T;
    }

    template <>
    MPI_Datatype type<int8_t>() {
      return MPI_INT8_t;
    }

    template <>
    MPI_Datatype type<uint16_t>() {
      return MPI_UINT16_T;
    }

    template <>
    MPI_Datatype type<int16_t>() {
      return MPI_INT16_T;
    }

    template <>
    MPI_Datatype type<uint32_t>() {
      return MPI_UINT32_T;
    }

    template <>
    MPI_Datatype type<int32_t>() {
      return MPI_INT32_T;
    }

    template <>
    MPI_Datatype type<uint64_t>() {
      return MPI_UINT64_T;
    }

    template <>
    MPI_Datatype type<int64_t>() {
      return MPI_INT64_T;
    }

    template <>
    MPI_Datatype type<float>() {
      return MPI_FLOAT;
    }

    template <>
    MPI_Datatype type<double>() {
      return MPI_DOUBLE;
    }
    //==================================

    //---[ Tag ]------------------------
    tag::tag() :
      mpiRequest(),
      initialized(false),
      done(false) {}

    tag::tag(MPI_Request mpiRequest_,
             bool done_) :
      mpiRequest(mpiRequest_),
      initialized(true),
      done(done_) {}

    bool tag::isInitialized() {
      return initialized;
    }

    void tag::wait() {
      if (!initialized || done) {
        return;
      }
      MPI_Status mpiStatus;
      MPI_Wait(mpiRequest, &mpiStatus);
      done = true;
    }

    void tag::updateStatus() {
      if (!initialized || done) {
        return;
      }
      MPI_Status mpiStatus;
      int flag;
      MPI_Request_get_status(mpiRequest,
                             &flag,
                             &mpiStatus);
      done = flag;
    }
    //==================================

    //---[ Tags ]-----------------------
    tags::tags() {}

    int tags::size() {
      return (int) tags_.size();
    }

    void tags::wait() {
      const int size_ = (int) mpiRequests.size();
      std::vector<MPI_Request> mpiRequests;
      std::vector<MPI_Status> mpiStatuses;
      MPI_Status dummyStatus;
      int realSize = 0;
      for (int i = 0; i < size_; ++i) {
        tag &tag_= tags_[i];
        if (tag_.initialized) {
          mpiRequests.push_back(tag.mpiRequest);
          mpiStatuses.push_back(dummyStatus);
          ++realSize;
        }
      }
      if (realSize) {
        MPI_Waitall(realSize,
                    &(mpiRequests[0]),
                    &(mpiStatuses[0]));
      }
      for (int i = 0; i < size_; ++i) {
        tag &tag_= tags_[i];
        if (tag_.initialized) {
          tag_.done = true;
        }
      }
    }

    void tags::updateStatus() {
      const int size_ = (int) mpiRequests.size();
      for (int i = 0; i < size_; ++i) {
        tags_[i].updateStatus();
      }
    }

    tag tags::operator [] (const int index) {
      const int size_ = (int) mpiRequests.size();
      if ((index < 0) || (size_ <= index)) {
        return tag();
      }
      return tags_[index];
    }

    tags& tags::operator += (const tag &tag_) {
      tags_.push_back(tag_);
      return *this;
    }
    //==================================

    //---[ Methods ]--------------------
    template <class TM>
    tag send(const int receiverID,
             const occa::memory &data,
             const dim_t entries_,
             const int messageID) {
      tag tag_;
      const dim_t entries = ((entries == -1)
                             ? (data.size() / sizeof(TM))
                             : entries);
      if ((receiverID < 0)            ||
          (mpi::size() <= receiverID) ||
          (entries < 0)) {
        return tag_;
      }
      if (!data
          .getDevice()
          .hasSeparateMemorySpace()) {
        MPI_Isend(data.ptr(),
                  entries,
                  type<TM>(),
                  receiverID,
                  messageID,
                  MPI_COMM_WORLD,
                  &tag_.mpiRequest);
      } else {
        int bufferEntries = buffer_t::size / sizeof(TM);
        void *buffer = getBuffer();

        for (int offset = 0; offset < entries; offset += bufferEntries) {
          int count = offset + bufferEntries;
          if (count >= entries) {
            count = (entries - offset);
          }

          data.copyTo(buffer,
                      bufferEntries * sizeof(TM));

          MPI_Send(buffer,
                   count,
                   type<TM>(),
                   receiverID,
                   messageID,
                   MPI_COMM_WORLD,
                   &tag_.mpiRequest);
        }
      }
      tag_.initialized = true;
      return tag_;
    }

    template <class TM>
    tag get(const int senderID,
            occa::memory &data,
            const dim_t entries_,
            const int messageID) {
      tag tag_;
      const dim_t entries = ((entries == -1)
                             ? (data.size() / sizeof(TM))
                             : entries);
      if ((receiverID < 0)            ||
          (mpi::size() <= receiverID) ||
          (entries < 0)) {
        return tag_;
      }
      if (!data
          .getDevice()
          .hasSeparateMemorySpace()) {
        MPI_Irecv(data.ptr(),
                  entries,
                  type<TM>(),
                  senderID,
                  messageID,
                  MPI_COMM_WORLD,
                  &tag_.mpiRequest);
      } else {
        int bufferEntries = buffer_t::size / sizeof(TM);
        void *buffer = getBuffer();

        for (int offset = 0; offset < entries; offset += bufferEntries) {
          int count = offset + bufferEntries;
          if (count >= entries) {
            count = (entries - offset);
          }

          data.copyTo(buffer,
                      bufferEntries * sizeof(TM));

          MPI_Recv(buffer,
                   count,
                   type<TM>(),
                   senderID,
                   messageID,
                   MPI_COMM_WORLD,
                   &tag_.mpiRequest);
        }
      }
      tag_.initialized = true;
      return tag_;
    }
    //==================================
  }
}

#endif
