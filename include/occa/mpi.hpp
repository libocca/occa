#include <occa/defines.hpp>

#if OCCA_MPI_ENABLED
#  ifndef OCCA_MPI_HEADER
#  define OCCA_MPI_HEADER

#include <mpi.h>
#include <vector>

#include <occa/tools/json.hpp>

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

    void barrier();

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
             const dim_t entries_ = -1,
             const int messageID  = defaultMessageID) {
      tag tag_;
      const dim_t entries = ((entries_ == -1)
                             ? (data.size() / sizeof(TM))
                             : entries_);
      if ((receiverID < 0)            ||
          (mpi::size() <= receiverID) ||
          (entries < 0)) {
        return tag_;
      }
      if (!data
          .getDevice()
          .hasSeparateMemorySpace()) {
        MPI_Isend((void*) data.ptr(),
                  entries,
                  type<TM>(),
                  receiverID,
                  messageID,
                  MPI_COMM_WORLD,
                  &tag_.mpiRequest);
        tag_.initialized = true;
      } else {
        int bufferEntries = buffer_t::size / sizeof(TM);
        void *buffer = getBuffer();

        for (int offset = 0; offset < entries; offset += bufferEntries) {
          int count = offset + bufferEntries;
          if (count >= entries) {
            count = (entries - offset);
          }

          data.copyTo(buffer,
                      count * sizeof(TM),
                      offset);

          MPI_Send(buffer,
                   count,
                   type<TM>(),
                   receiverID,
                   messageID,
                   MPI_COMM_WORLD);
        }
      }
      return tag_;
    }

    template <class TM>
    tag get(const int senderID,
            occa::memory data,
            const dim_t entries_ = -1,
            const int messageID  = defaultMessageID) {
      tag tag_;
      const dim_t entries = ((entries_ == -1)
                             ? (data.size() / sizeof(TM))
                             : entries_);
      if ((senderID < 0)            ||
          (mpi::size() <= senderID) ||
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
        tag_.initialized = true;
      } else {
        int bufferEntries = buffer_t::size / sizeof(TM);
        void *buffer = getBuffer();

        for (int offset = 0; offset < entries; offset += bufferEntries) {
          int count = offset + bufferEntries;
          if (count >= entries) {
            count = (entries - offset);
          }

          MPI_Status mpiStatus;
          MPI_Recv(buffer,
                   count,
                   type<TM>(),
                   senderID,
                   messageID,
                   MPI_COMM_WORLD,
                   &mpiStatus);

          data.copyFrom(buffer,
                        count * sizeof(TM),
                        offset);
        }
      }
      return tag_;
    }
    //==================================
  }
}

#  endif
#endif
