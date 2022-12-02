#ifndef OCCA_CORE_STREAMTAG_HEADER
#define OCCA_CORE_STREAMTAG_HEADER

#include <iostream>

#include <occa/defines.hpp>

// Unfortunately we need to expose this in include
#include <occa/utils/gc.hpp>

namespace occa {
  class modeDevice_t; class device;
  class modeStreamTag_t; class streamTag;

  /**
   * @startDoc{streamTag}
   *
   * Description:
   *   The result from calling [[device.tagStream]].
   *
   *   A stream tag can be used to check how much time elapsed between two tags ([[device.timeBetween]]).
   *
   *   A stream tag can also be used to wait all work queued up before the tag ([[streamTag.wait()|streamTag.wait]] or [[device.waitFor()|device.waitFor]]).
   *
   * @endDoc
   */
  class streamTag : public gc::ringEntry_t {
    friend class occa::modeStreamTag_t;
    friend class occa::device;

  private:
    modeStreamTag_t *modeStreamTag;

  public:
    streamTag();
    streamTag(modeStreamTag_t *modeStreamTag_);

    streamTag(const streamTag &s);
    streamTag& operator = (const streamTag &m);
    ~streamTag();

  private:
    void setModeStreamTag(modeStreamTag_t *modeStreamTag_);
    void removeStreamTagRef();

  public:
    void dontUseRefs();

    /**
     * @startDoc{isInitialized}
     *
     * Description:
     *   Check whether the [[streamTag]] has been intialized.
     *
     * Returns:
     *   Returns `true` if the [[streamTag]] has been intialized
     *
     * @endDoc
     */
    bool isInitialized() const;

    modeStreamTag_t* getModeStreamTag() const;
    modeDevice_t* getModeDevice() const;

    /**
     * @startDoc{getDevice}
     *
     * Description:
     *   Returns the [[device]] used to build the [[streamTag]].
     *
     * Returns:
     *   The [[device]] used to build the [[streamTag]]
     *
     * @endDoc
     */
    occa::device getDevice() const;

    /**
     * @startDoc{wait}
     *
     * Description:
     *   Wait for all queued operations on the [[stream]] before this [[streamTag]] was tagged.
     *   This includes [[kernel]] calls and [[memory]] data transfers.
     *
     * @endDoc
     */
    void wait() const;

    /**
     * @startDoc{operator_equals[0]}
     *
     * Description:
     *   Compare if two streamTags have the same references.
     *
     * Returns:
     *   If the references are the same, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator == (const occa::streamTag &other) const;

    /**
     * @startDoc{operator_equals[1]}
     *
     * Description:
     *   Compare if two streamTags have different references.
     *
     * Returns:
     *   If the references are different, this returns `true` otherwise `false`.
     *
     * @endDoc
     */
    bool operator != (const occa::streamTag &other) const;

    /**
     * @startDoc{free}
     *
     * Description:
     *   Free the streamTag object.
     *   Calling [[streamTag.isInitialized]] will return `false` now.
     *
     * @endDoc
     */
    void free();

    /**
     * @startDoc{unwrap}
     * 
     * Description:
     *   Retreives the mode-specific object associated with this [[streamTag]].
     *   The lifetime of the returned object is the same as this streamTag.
     *   Destruction of the returned object during this streamTag's lifetime results in undefined behavior.   
     *  
     *   > An OCCA application is responsible for correctly converting the returned `void*` pointer to the corresponding mode-specific streamTag type.
     * 
     * Returns:
     *   A pointer to the mode-specific object associated with this streamTag
     * 
     * @endDoc
    */
    void* unwrap();
  };
}

#endif
