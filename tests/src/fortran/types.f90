program main
  use occa

  implicit none

  call testOccaType()

  stop 0

contains

  subroutine testOccaType()
    implicit none

    type(occaType) :: t

    ! ---[ Known Types ]--------------------
    t = occaUndefined
    if (.not. occaIsUndefined(t)) stop "*** ERROR ***: types - occaUndefined"
    if (      occaIsDefault(t))   stop "*** ERROR ***: types - occaUndefined"

    t = occaDefault
    if (.not. occaIsDefault(t))   stop "*** ERROR ***: types - occaDefault"
    if (      occaIsUndefined(t)) stop "*** ERROR ***: types - occaDefault"

    t = occaNull
    if (      occaIsDefault(t))   stop "*** ERROR ***: types - occaNull"
    if (      occaIsUndefined(t)) stop "*** ERROR ***: types - occaNull"

    t = occaPtr(C_NULL_ptr)
    if (t%type .ne. OCCA_C_PTR)   stop "*** ERROR ***: types - occaPtr"

    t = occaBool(.true._C_bool)
    if (t%type .ne. OCCA_C_BOOL)  stop "*** ERROR ***: types - occaBool"
    ! ======================================


    ! ---[ Integer Types ]------------------
    t = occaInt(123_C_int8_t)
    if (t%type .ne. OCCA_C_INT8)  stop "*** ERROR ***: types - occaInt8"

    t = occaInt(123_C_int16_t)
    if (t%type .ne. OCCA_C_INT16) stop "*** ERROR ***: types - occaInt16"

    t = occaInt(123_C_int32_t)
    if (t%type .ne. OCCA_C_INT32) stop "*** ERROR ***: types - occaInt32"

    t = occaInt(123_C_int64_t)
    if (t%type .ne. OCCA_C_INT64) stop "*** ERROR ***: types - occaInt64"
    ! ======================================


    ! ---[ Real Types ]---------------------
    t = occaReal(123.45_C_float)
    if (t%type .ne. OCCA_C_FLOAT)  stop "*** ERROR ***: types - occaFloat"

    t = occaReal(123.45_C_double)
    if (t%type .ne. OCCA_C_DOUBLE) stop "*** ERROR ***: types - occaDouble"
    ! ======================================


    ! ---[ Ambiguous Types ]----------------
    t = occaStruct(C_NULL_ptr, 123_occaUDim_t)
    if (t%type .ne. OCCA_C_STRUCT) stop "*** ERROR ***: types - occaStruct"

    t = occaString("123")
    if (t%type .ne. OCCA_C_STRING) stop "*** ERROR ***: types - occaString"
    ! ======================================
  end subroutine
end program
