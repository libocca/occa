#include <occa/defines.hpp>

#if OCCA_MPI_ENABLED
#  ifndef OCCA_EXPERIMENTAL_MPI_HEADER
#  define OCCA_EXPERIMENTAL_MPI_HEADER

#include <mpi.h>
#include <vector>

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
    template <class T>
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
    template <class T>
    tag send(const int receiverID,
             const occa::memory &data,
             const dim_t entries_ = -1,
             const int messageID  = defaultMessageID) {
      tag tag_;
      const dim_t entries = ((entries_ == -1)
                             ? (data.size() / sizeof(T))
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
                  type<T>(),
                  receiverID,
                  messageID,
                  MPI_COMM_WORLD,
                  &tag_.mpiRequest);
        tag_.initialized = true;
      } else {
        int bufferEntries = buffer_t::size / sizeof(T);
        void *buffer = getBuffer();

        for (int offset = 0; offset < entries; offset += bufferEntries) {
          int count = offset + bufferEntries;
          if (count >= entries) {
            count = (entries - offset);
          }

          data.copyTo(buffer,
                      count * sizeof(T),
                      offset);

          MPI_Send(buffer,
                   count,
                   type<T>(),
                   receiverID,
                   messageID,
                   MPI_COMM_WORLD);
        }
      }
      return tag_;
    }

    template <class T>
    tag get(const int senderID,
            occa::memory data,
            const dim_t entries_ = -1,
            const int messageID  = defaultMessageID) {
      tag tag_;
      const dim_t entries = ((entries_ == -1)
                             ? (data.size() / sizeof(T))
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
                  type<T>(),
                  senderID,
                  messageID,
                  MPI_COMM_WORLD,
                  &tag_.mpiRequest);
        tag_.initialized = true;
      } else {
        int bufferEntries = buffer_t::size / sizeof(T);
        void *buffer = getBuffer();

        for (int offset = 0; offset < entries; offset += bufferEntries) {
          int count = offset + bufferEntries;
          if (count >= entries) {
            count = (entries - offset);
          }

          MPI_Status mpiStatus;
          MPI_Recv(buffer,
                   count,
                   type<T>(),
                   senderID,
                   messageID,
                   MPI_COMM_WORLD,
                   &mpiStatus);

          data.copyFrom(buffer,
                        count * sizeof(T),
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
