#include <occa/defines.hpp>

#if OCCA_MPI_ENABLED

#include <occa/core/base.hpp>
#include <occa/experimental/mpi.hpp>
#include <occa/internal/utils/tls.hpp>

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
      if (id_ >= 0) {
        return id_;
      }
      MPI_Comm_rank(MPI_COMM_WORLD,
                    &id_);
      return id_;
    }

    int size() {
      static int processes_ = -1;
      if (processes_ >= 0) {
        return processes_;
      }
      MPI_Comm_size(MPI_COMM_WORLD,
                    &processes_);
      return processes_;
    }

    char* getBuffer() {
      static tls<buffer_t> buffer_;
      return buffer_.value().ptr;
    }

    void barrier() {
      MPI_Barrier(MPI_COMM_WORLD);
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
      return MPI_INT8_T;
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

    bool tag::isInitialized() {
      return initialized;
    }

    void tag::wait() {
      if (!initialized || done) {
        return;
      }
      MPI_Status mpiStatus;
      MPI_Wait(&mpiRequest, &mpiStatus);
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
      std::vector<MPI_Request> mpiRequests;
      std::vector<MPI_Status> mpiStatuses;
      MPI_Status dummyStatus;
      const int size_ = (int) tags_.size();
      int realSize = 0;
      for (int i = 0; i < size_; ++i) {
        tag &tag_= tags_[i];
        if (tag_.initialized) {
          mpiRequests.push_back(tag_.mpiRequest);
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
      const int size_ = (int) tags_.size();
      for (int i = 0; i < size_; ++i) {
        tags_[i].updateStatus();
      }
    }

    tag tags::operator [] (const int index) {
      const int size_ = (int) tags_.size();
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
  }
}

#endif
