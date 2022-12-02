#ifndef OCCA_CORE_STREAM_HEADER
#define OCCA_CORE_STREAM_HEADER

#include <iostream>

#include <occa/defines.hpp>
#include <occa/types.hpp>

// Unfortunately we need to expose this in include
#include <occa/utils/gc.hpp>

namespace occa {
  class modeStream_t; class stream;
  class modeDevice_t; class device;

  /**
   * @startDoc{stream}
   *
   * Description:
   *   A [[device]] has one active [[stream]] at a time.
   *   If the backend supports it, using multiple streams will achieve better parallelism by having more work queued up.
   *   Work on a stream is considered to be done in order, but can be out of order if work is queued using multiple streams.
   *
   * @endDoc
   */
  class stream : public gc::ringEntry_t {
    friend class occa::modeStream_t;
    friend class occa::device;

   private:
    modeStream_t *modeStream;

   public:
    stream();
    stream(modeStream_t *modeStream_);

    stream(const stream &s);
    stream& operator = (const stream &m);
    ~stream();

   private:
    void setModeStream(modeStream_t *modeStream_);
    void removeStreamRef();

   public:
    void dontUseRefs();

    /**
     * @startDoc{isInitialized}
     *
     * Description:
     *   Check whether the [[stream]] has been intialized.
     *
     * Returns:
     *   Returns `true` if the [[stream]] has been intialized
     *
     * @endDoc
     */
    bool isInitialized() const;

    modeStream_t* getModeStream() const;
    modeDevice_t* getModeDevice() const;

    /**
     * @startDoc{getDevice}
     *
     * Description:
     *   Returns the [[device]] used to build the [[stream]].
     *
     * Returns:
     *   The [[device]] used to build the [[stream]]
     *
     * @endDoc
     */
    occa::device getDevice() const;

    /**
     * @startDoc{mode}
     *
     * Description:
     *   Returns the mode of the [[device]] used to build the [[stream]].
     *
     * Returns:
     *   The `mode` string, such as `"Serial"`, `"CUDA"`, or `"HIP"`.
     *
     * @endDoc
     */
    const std::string& mode() const;

    /**
     * @startDoc{properties}
     *
     * Description:
     *   Get the properties used to build the [[stream]].
     *
     * Description:
     *   Returns the properties used to build the [[stream]].
     *
     * @endDoc
     */
    const occa::json& properties() const;

    /**
     * @startDoc{operator_equals[0]}
     *
     * Description:
     *   Compare if two streams have the same references.
     *
     * Returns:
     *   If the references are the same, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator == (const occa::stream &other) const;

    /**
     * @startDoc{operator_equals[1]}
     *
     * Description:
     *   Compare if two streams have different references.
     *
     * Returns:
     *   If the references are different, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator != (const occa::stream &other) const;

    /**
     * @startDoc{free}
     *
     * Description:
     *   Free the stream object.
     *   Calling [[stream.isInitialized]] will return `false` now.
     *
     * @endDoc
     */
    void free();

    /**
     * @startDoc{finish}
     *
     * Description:
     *   Waits for all asynchronous operations, such as memory allocations
     *   or kernel calls, submitted to this device to complete.
     *
     * @endDoc
     */
    void finish();

    /**
     * @startDoc{unwrap}
     * 
     * Description:
     *   Retreives the mode-specific object associated with this [[memory]].
     *   The lifetime of the returned object is the same as this memory.
     *   Destruction of the returned object during this memory's lifetime results in undefined behavior.   
     *  
     *   > An OCCA application is responsible for correctly converting the returned `void*` pointer to the corresponding mode-specific memory type.
     * 
     * Returns:
     *   A pointer to the mode-specific object associated with this stream.
     * 
     * @endDoc
    */
    void* unwrap();
  };

  std::ostream& operator << (std::ostream &out,
                             const occa::stream &stream);
}

#endif
