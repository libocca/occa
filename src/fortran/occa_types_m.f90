module occa_types_m
  ! occa/c/types.h

  use occa_typedefs_m, &
    ! OCCA aliases for derived types:
    occaType => occaType, &

    occaDevice => occaType, &
    occaKernel => occaType, &
    occaKernelBuilder => occaType, &
    occaMemory => occaType, &
    occaStream => occaType, &
    occaStreamTag => occaType, &

    occaDtype => occaType, &
    occaScope => occaType, &
    occaJson => occaType

  implicit none

  interface
    ! -----[ Known Types ]------------------
    ! bool occaIsUndefined(occaType value);
    logical(kind=C_bool) function occaIsUndefined(value) &
                                  bind(C, name="occaIsUndefined")
      import C_bool, occaType
      implicit none
      type(occaType), value :: value
    end function

    ! bool occaIsDefault(occaType value);
    logical(kind=C_bool) function occaIsDefault(value) &
                                  bind(C, name="occaIsDefault")
      import C_bool, occaType
      implicit none
      type(occaType), value :: value
    end function

    ! occaType occaPtr(void *value);
    type(occaType) function occaPtr(value) bind(C, name="occaPtr")
      import occaType, C_ptr
      implicit none
      type(C_ptr), value :: value
    end function

    ! occaType occaBool(bool value);
    ! NOTE: This function is not meant to be used directly. Use `occaBool`
    !       instead.
    type(occaType) function occaBool_C(value) bind(C, name="occaBool")
      import occaType, C_bool
      implicit none
      logical(kind=C_bool), value :: value
    end function
    ! ======================================
  end interface

  interface occaInt
    ! -----[ Integer Types ]----------------
    ! NOTE: This provides a consistent interface for integer types in Fortran,
    !       but slight deviates from the C interface
    ! occaType occaInt8(int8_t value);
    ! occaType occaUInt8(uint8_t value);
    type(occaType) function occaInt8(value) bind(C, name="occaInt8")
      import occaType, C_int8_t
      implicit none
      integer(C_int8_t), value :: value
    end function

    ! occaType occaInt16(int16_t value);
    ! occaType occaUInt16(uint16_t value);
    type(occaType) function occaInt16(value) bind(C, name="occaInt16")
      import occaType, C_int16_t
      implicit none
      integer(C_int16_t), value :: value
    end function

    ! occaType occaInt32(int32_t value);
    ! occaType occaUInt32(uint32_t value);
    type(occaType) function occaInt32(value) bind(C, name="occaInt32")
      import occaType, C_int32_t
      implicit none
      integer(C_int32_t), value :: value
    end function

    ! occaType occaInt64(int64_t value);
    ! occaType occaUInt64(uint64_t value);
    type(occaType) function occaInt64(value) bind(C, name="occaInt64")
      import occaType, C_int64_t
      implicit none
      integer(C_int64_t), value :: value
    end function

    ! NOTE: C_signed_char == C_int8_t
    ! ! occaType occaChar(char value);
    ! ! occaType occaUChar(unsigned char value);
    ! type(occaType) function occaChar(value) bind(C, name="occaChar")
    !   import occaType, C_signed_char
    !   implicit none
    !   integer(C_signed_char), value :: value
    ! end function

    ! NOTE: C_short == C_int16_t
    ! ! occaType occaShort(short value);
    ! ! occaType occaUShort(unsigned short value);
    ! type(occaType) function occaShort(value) bind(C, name="occaShort")
    !   import occaType, C_short
    !   implicit none
    !   integer(C_short), value :: value
    ! end function

    ! NOTE: C_int == C_int32_t
    ! ! occaType occaInt(int value);
    ! ! occaType occaUInt(unsigned int value);
    ! type(occaType) function occaInt(value) bind(C, name="occaInt")
    !   import occaType, C_int
    !   implicit none
    !   integer(C_int), value :: value
    ! end function

    ! NOTE: C_long == C_int64_t
    ! ! occaType occaLong(long value);
    ! ! occaType occaULong(unsigned long value);
    ! type(occaType) function occaLong(value) bind(C, name="occaLong")
    !   import occaType, C_long
    !   implicit none
    !   integer(C_long), value :: value
    ! end function
    ! ======================================
  end interface

  interface occaReal
    ! -----[ Real Types ]-------------------
    ! occaType occaFloat(float value);
    type(occaType) function occaFloat(value) bind(C, name="occaFloat")
      import occaType, C_float
      implicit none
      real(C_float), value :: value
    end function

    ! occaType occaDouble(double value);
    type(occaType) function occaDouble(value) bind(C, name="occaDouble")
      import occaType, C_double
      implicit none
      real(C_double), value :: value
    end function
  end interface

  interface
    ! -----[ Ambiguous Types ]--------------
    ! occaType occaStruct(const void *value, occaUDim_t bytes);
    type(occaType) function occaStruct(value, bytes) bind(C, name="occaStruct")
      import occaType, C_void_ptr, occaUDim_t
      implicit none
      type(C_void_ptr), value, intent(in) :: value
      integer(occaUDim_t), value :: bytes
    end function

    ! occaType occaString(const char *str);
    type(occaType) function occaString(str) bind(C, name="occaString")
      import occaType, C_char
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: str
    end function
    ! ======================================

    ! void occaFree(occaType *value)
    subroutine occaFree(value) bind(C, name="occaFree")
      import occaType
      implicit none
      type(occaType) :: value
    end subroutine
  end interface

  interface occaBool
    module procedure occaBool1
    module procedure occaBool4
    module procedure occaBool8
    !module procedure occaBoolInt1
    !module procedure occaBoolInt4
    !module procedure occaBoolInt8
  end interface

  ! ---[ Type Flags ]---------------------
  integer(C_int), bind(C, name="OCCA_UNDEFINED")     :: OCCA_C_UNDEFINED
  integer(C_int), bind(C, name="OCCA_DEFAULT")       :: OCCA_C_DEFAULT
  integer(C_int), bind(C, name="OCCA_NULL")          :: OCCA_C_NULL

  integer(C_int), bind(C, name="OCCA_PTR")           :: OCCA_C_PTR

  integer(C_int), bind(C, name="OCCA_BOOL")          :: OCCA_C_BOOL
  integer(C_int), bind(C, name="OCCA_INT8")          :: OCCA_C_INT8
  integer(C_int), bind(C, name="OCCA_UINT8")         :: OCCA_C_UINT8
  integer(C_int), bind(C, name="OCCA_INT16")         :: OCCA_C_INT16
  integer(C_int), bind(C, name="OCCA_UINT16")        :: OCCA_C_UINT16
  integer(C_int), bind(C, name="OCCA_INT32")         :: OCCA_C_INT32
  integer(C_int), bind(C, name="OCCA_UINT32")        :: OCCA_C_UINT32
  integer(C_int), bind(C, name="OCCA_INT64")         :: OCCA_C_INT64
  integer(C_int), bind(C, name="OCCA_UINT64")        :: OCCA_C_UINT64
  integer(C_int), bind(C, name="OCCA_FLOAT")         :: OCCA_C_FLOAT
  integer(C_int), bind(C, name="OCCA_DOUBLE")        :: OCCA_C_DOUBLE

  integer(C_int), bind(C, name="OCCA_STRUCT")        :: OCCA_C_STRUCT
  integer(C_int), bind(C, name="OCCA_STRING")        :: OCCA_C_STRING

  integer(C_int), bind(C, name="OCCA_DEVICE")        :: OCCA_C_DEVICE
  integer(C_int), bind(C, name="OCCA_KERNEL")        :: OCCA_C_KERNEL
  integer(C_int), bind(C, name="OCCA_KERNELBUILDER") :: OCCA_C_KERNELBUILDER
  integer(C_int), bind(C, name="OCCA_MEMORY")        :: OCCA_C_MEMORY
  integer(C_int), bind(C, name="OCCA_STREAM")        :: OCCA_C_STREAM
  integer(C_int), bind(C, name="OCCA_STREAMTAG")     :: OCCA_C_STREAMTAG

  integer(C_int), bind(C, name="OCCA_DTYPE")         :: OCCA_C_DTYPE
  integer(C_int), bind(C, name="OCCA_SCOPE")         :: OCCA_C_SCOPE
  integer(C_int), bind(C, name="OCCA_JSON")          :: OCCA_C_JSON
  ! ======================================

  ! ---[ Globals & Flags ]----------------
  type(occaType), bind(C, name="occaNull")          :: occaNull
  type(occaType), bind(C, name="occaDefault")       :: occaDefault
  type(occaType), bind(C, name="occaUndefined")     :: occaUndefined
  type(occaType), bind(C, name="occaTrue")          :: occaTrue
  type(occaType), bind(C, name="occaFalse")         :: occaFalse
  integer(occaUDim_t), bind(C, name="occaAllBytes") :: occaAllBytes
  ! ======================================

  private :: occaBool1, occaBool4, occaBool8
  !private :: occaBool1, occaBool4, occaBool8, &
  !           occaBoolInt1, occaBoolInt4, occaBoolInt8

contains

  type(occaType) function occaBool1(val) result(res)
    implicit none
    logical(kind=1), intent(in) :: val
    res = occaBool_C(logical(val, kind=C_bool))
  end function

  type(occaType) function occaBool4(val) result(res)
    implicit none
    logical(kind=4), intent(in) :: val
    res = occaBool_C(logical(val, kind=C_bool))
  end function

  type(occaType) function occaBool8(val) result(res)
    implicit none
    logical(kind=8), intent(in) :: val
    res = occaBool_C(logical(val, kind=C_bool))
  end function

  ! NOTE: Some compiler support an implict conversion between integer and
  !       logical. This is however a non-standard extension, see e.g.:
  !       https://gcc.gnu.org/onlinedocs/gfortran/Implicitly-convert-LOGICAL-and-INTEGER-values.html
  !type(occaType) function occaBoolInt1(val) result(res)
  !  implicit none
  !  integer(kind=1), intent(in) :: val
  !  logical(kind=C_bool) :: l
  !  l = val
  !  res = occaBool_C(l)
  !end function
  !
  !type(occaType) function occaBoolInt4(val) result(res)
  !  implicit none
  !  integer(kind=4), intent(in) :: val
  !  logical(kind=C_bool) :: l
  !  l = val
  !  res = occaBool_C(l)
  !end function
  !
  !type(occaType) function occaBoolInt8(val) result(res)
  !  implicit none
  !  integer(kind=8), intent(in) :: val
  !  logical(kind=C_bool) :: l
  !  l = val
  !  res = occaBool_C(l)
  !end function

end module occa_types_m
