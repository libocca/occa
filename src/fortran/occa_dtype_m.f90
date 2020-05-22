module occa_dtype_m
  ! occa/c/dtype.h

  use occa_types_m

  implicit none

  interface
    ! -----[ Methods ]----------------------
    ! occaDtype occaCreateDtype(const char *name, const int bytes);
    type(occaDtype) function occaCreateDtype(name, bytes) &
                             bind(C, name="occaCreateDtype")
      import C_char, occaDtype, C_int
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: name
      integer(C_int), value, intent(in) :: bytes
    end function

    ! occaDtype occaCreateDtypeTuple(occaDtype dtype, const int size);
    type(occaDtype) function occaCreateDtypeTuple(dtype, size) &
                             bind(C, name="occaCreateDtypeTuple")
      import occaDtype, C_int
      implicit none
      type(occaDtype), value :: dtype
      integer(C_int), value, intent(in) :: size
    end function

    ! const char* occaDtypeName(occaDtype dtype);
    type(C_char_ptr) function occaDtypeName(dtype) bind(C, name="occaDtypeName")
      import occaDtype, C_char_ptr
      implicit none
      type(occaDtype), value :: dtype
    end function

    ! int occaDtypeBytes(occaDtype dtype);
    integer(C_int) function occaDtypeBytes(dtype) bind(C, name="occaDtypeBytes")
      import C_int, occaDtype
      implicit none
      type(occaDtype), value :: dtype
    end function


    ! void occaDtypeRegisterType(occaDtype dtype);
    subroutine occaDtypeRegisterType(dtype) &
               bind(C, name="occaDtypeRegisterType")
      import occaDtype
      implicit none
      type(occaDtype), value :: dtype
    end subroutine

    ! bool occaDtypeIsRegistered(occaDtype dtype);
    logical(kind=C_bool) function occaDtypeIsRegistered(dtype) &
                                  bind(C, name="occaDtypeIsRegistered")
      import C_bool, occaDtype
      implicit none
      type(occaDtype), value :: dtype
    end function

    ! void occaDtypeAddField(occaDtype dtype,
    !                        const char *field,
    !                        occaDtype fieldType);
    subroutine occaDtypeAddField(dtype, field, fieldType) &
               bind(C, name="occaDtypeAddField")
      import occaDtype, C_char
      implicit none
      type(occaDtype), value :: dtype, fieldType
      character(len=1,kind=C_char), dimension(*), intent(in) :: field
    end subroutine

    ! bool occaDtypesAreEqual(occaDtype a, occaDtype b);
    logical(kind=C_bool) function occaDtypesAreEqual(a, b) &
                                  bind(C, name="occaDtypesAreEqual")
      import C_bool, occaDtype
      implicit none
      type(occaDtype), value :: a, b
    end function

    ! bool occaDtypesMatch(occaDtype a, occaDtype b);
    logical(kind=C_bool) function occaDtypesMatch(a, b) &
                                  bind(C, name="occaDtypesMatch")
      import C_bool, occaDtype
      implicit none
      type(occaDtype), value :: a, b
    end function

    ! occaDtype occaDtypeFromJson(occaJson json);
    type(occaDtype) function occaDtypeFromJson(json) &
                             bind(C, name="occaDtypeFromJson")
      import occaDtype, occaJson
      implicit none
      type(occaJson), value :: json
    end function
    ! occaDtype occaDtypeFromJsonString(const char *str);
    type(occaDtype) function occaDtypeFromJsonString(str) &
                             bind(C, name="occaDtypeFromJsonString")
      import occaDtype, C_char
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: str
    end function

    ! occaJson occaDtypeToJson(occaDtype dtype);
    type(occaJson) function occaDtypeToJson(dtype) &
                            bind(C, name="occaDtypeToJson")
      import occaDtype, occaJson
      implicit none
      type(occaDtype), value :: dtype
    end function
    ! ======================================
  end interface

  ! -----[ Builtins ]---------------------
  type(occaDType), bind(C, name="occaDtypeNone") :: occaDtypeNone

  type(occaDType), bind(C, name="occaDtypeVoid") :: occaDtypeVoid
  type(occaDType), bind(C, name="occaDtypeByte") :: occaDtypeByte

  type(occaDType), bind(C, name="occaDtypeBool") :: occaDtypeBool
  type(occaDType), bind(C, name="occaDtypeChar") :: occaDtypeChar
  type(occaDType), bind(C, name="occaDtypeShort") :: occaDtypeShort
  type(occaDType), bind(C, name="occaDtypeInt") :: occaDtypeInt
  type(occaDType), bind(C, name="occaDtypeLong") :: occaDtypeLong
  type(occaDType), bind(C, name="occaDtypeFloat") :: occaDtypeFloat
  type(occaDType), bind(C, name="occaDtypeDouble") :: occaDtypeDouble

  type(occaDType), bind(C, name="occaDtypeInt8") :: occaDtypeInt8
  type(occaDType), bind(C, name="occaDtypeUint8") :: occaDtypeUint8
  type(occaDType), bind(C, name="occaDtypeInt16") :: occaDtypeInt16
  type(occaDType), bind(C, name="occaDtypeUint16") :: occaDtypeUint16
  type(occaDType), bind(C, name="occaDtypeInt32") :: occaDtypeInt32
  type(occaDType), bind(C, name="occaDtypeUint32") :: occaDtypeUint32
  type(occaDType), bind(C, name="occaDtypeInt64") :: occaDtypeInt64
  type(occaDType), bind(C, name="occaDtypeUint64") :: occaDtypeUint64

  ! OKL Primitives
  type(occaDType), bind(C, name="occaDtypeUchar2") :: occaDtypeUchar2
  type(occaDType), bind(C, name="occaDtypeUchar3") :: occaDtypeUchar3
  type(occaDType), bind(C, name="occaDtypeUchar4") :: occaDtypeUchar4

  type(occaDType), bind(C, name="occaDtypeChar2") :: occaDtypeChar2
  type(occaDType), bind(C, name="occaDtypeChar3") :: occaDtypeChar3
  type(occaDType), bind(C, name="occaDtypeChar4") :: occaDtypeChar4

  type(occaDType), bind(C, name="occaDtypeUshort2") :: occaDtypeUshort2
  type(occaDType), bind(C, name="occaDtypeUshort3") :: occaDtypeUshort3
  type(occaDType), bind(C, name="occaDtypeUshort4") :: occaDtypeUshort4

  type(occaDType), bind(C, name="occaDtypeShort2") :: occaDtypeShort2
  type(occaDType), bind(C, name="occaDtypeShort3") :: occaDtypeShort3
  type(occaDType), bind(C, name="occaDtypeShort4") :: occaDtypeShort4

  type(occaDType), bind(C, name="occaDtypeUint2") :: occaDtypeUint2
  type(occaDType), bind(C, name="occaDtypeUint3") :: occaDtypeUint3
  type(occaDType), bind(C, name="occaDtypeUint4") :: occaDtypeUint4

  type(occaDType), bind(C, name="occaDtypeInt2") :: occaDtypeInt2
  type(occaDType), bind(C, name="occaDtypeInt3") :: occaDtypeInt3
  type(occaDType), bind(C, name="occaDtypeInt4") :: occaDtypeInt4

  type(occaDType), bind(C, name="occaDtypeUlong2") :: occaDtypeUlong2
  type(occaDType), bind(C, name="occaDtypeUlong3") :: occaDtypeUlong3
  type(occaDType), bind(C, name="occaDtypeUlong4") :: occaDtypeUlong4

  type(occaDType), bind(C, name="occaDtypeLong2") :: occaDtypeLong2
  type(occaDType), bind(C, name="occaDtypeLong3") :: occaDtypeLong3
  type(occaDType), bind(C, name="occaDtypeLong4") :: occaDtypeLong4

  type(occaDType), bind(C, name="occaDtypeFloat2") :: occaDtypeFloat2
  type(occaDType), bind(C, name="occaDtypeFloat3") :: occaDtypeFloat3
  type(occaDType), bind(C, name="occaDtypeFloat4") :: occaDtypeFloat4

  type(occaDType), bind(C, name="occaDtypeDouble2") :: occaDtypeDouble2
  type(occaDType), bind(C, name="occaDtypeDouble3") :: occaDtypeDouble3
  type(occaDType), bind(C, name="occaDtypeDouble4") :: occaDtypeDouble4
  ! ======================================

end module occa_dtype_m
