module occa_json_m
  ! occa/c/json.h

  use occa_types_m

  implicit none

  interface
    ! occaJson occaCreateJson();
    type(occaJson) function occaCreateJson() bind(C, name="occaCreateJson")
      import occaJson
    end function

    ! ---[ Global methods ]-----------------
    ! occaJson occaJsonParse(const char *c);
    type(occaJson) function occaJsonParse(c) bind(C, name="occaJsonParse")
      import occaJson, C_char
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: c
    end function

    ! occaJson occaJsonRead(const char *filename);
    type(occaJson) function occaJsonRead(filename) bind(C, name="occaJsonRead")
      import occaJson, C_char
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: filename
    end function

    ! void occaJsonWrite(occaJson j, const char *filename);
    subroutine occaJsonWrite(j, filename) bind(C, name="occaJsonWrite")
      import occaJson, C_char
      implicit none
      type(occaJson), value :: j
      character(len=1,kind=C_char), dimension(*), intent(in) :: filename
    end subroutine

    ! const char* occaJsonDump(occaJson j, const int indent);
    type(C_char_ptr) function occaJsonDump(j, indent) &
                              bind(C, name="occaJsonDump")
      import occaJson, C_char_ptr, C_int
      implicit none
      type(occaJson), value :: j
      integer(C_int), value, intent(in) :: indent
    end function
    ! ======================================


    ! ---[ Type checks ]--------------------
    ! bool occaJsonIsBoolean(occaJson j);
    logical(kind=C_bool) function occaJsonIsBoolean(j) &
                                  bind(C, name="occaJsonIsBoolean")
      import occaJson, C_bool
      implicit none
      type(occaJson), value :: j
    end function
    ! bool occaJsonIsNumber(occaJson j);
    logical(kind=C_bool) function occaJsonIsNumber(j) &
                                  bind(C, name="occaJsonIsNumber")
      import occaJson, C_bool
      implicit none
      type(occaJson), value :: j
    end function
    ! bool occaJsonIsString(occaJson j);
    logical(kind=C_bool) function occaJsonIsString(j) &
                                  bind(C, name="occaJsonIsString")
      import occaJson, C_bool
      implicit none
      type(occaJson), value :: j
    end function
    ! bool occaJsonIsArray(occaJson j);
    logical(kind=C_bool) function occaJsonIsArray(j) &
                                  bind(C, name="occaJsonIsArray")
      import occaJson, C_bool
      implicit none
      type(occaJson), value :: j
    end function
    ! bool occaJsonIsObject(occaJson j);
    logical(kind=C_bool) function occaJsonIsObject(j) &
                                  bind(C, name="occaJsonIsObject")
      import occaJson, C_bool
      implicit none
      type(occaJson), value :: j
    end function
    ! ======================================


    ! ---[ Casters ]------------------------
    ! void occaJsonCastToBoolean(occaJson j);
    subroutine occaJsonCastToBoolean(j) bind(C, name="occaJsonCastToBoolean")
      import occaJson
      implicit none
      type(occaJson), value :: j
    end subroutine
    ! void occaJsonCastToNumber(occaJson j);
    subroutine occaJsonCastToNumber(j) bind(C, name="occaJsonCastToNumber")
      import occaJson
      implicit none
      type(occaJson), value :: j
    end subroutine
    ! void occaJsonCastToString(occaJson j);
    subroutine occaJsonCastToString(j) bind(C, name="occaJsonCastToString")
      import occaJson
      implicit none
      type(occaJson), value :: j
    end subroutine
    ! void occaJsonCastToArray(occaJson j);
    subroutine occaJsonCastToArray(j) bind(C, name="occaJsonCastToArray")
      import occaJson
      implicit none
      type(occaJson), value :: j
    end subroutine
    ! void occaJsonCastToObject(occaJson j);
    subroutine occaJsonCastToObject(j) bind(C, name="occaJsonCastToObject")
      import occaJson
      implicit none
      type(occaJson), value :: j
    end subroutine
    ! ======================================


    ! ---[ Getters ]------------------------
    ! bool occaJsonGetBoolean(occaJson j);
    logical(kind=C_bool) function occaJsonGetBoolean(j) &
                                  bind(C, name="occaJsonGetBoolean")
      import occaJson, C_bool
      implicit none
      type(occaJson), value :: j
    end function
    ! occaType occaJsonGetNumber(occaJson j, const int type);
    type(occaType) function occaJsonGetNumber(j, type) &
                            bind(C, name="occaJsonGetNumber")
      import occaJson, occaType, C_int
      implicit none
      type(occaJson), value :: j
      integer(C_int), value, intent(in) :: type
    end function
    ! const char* occaJsonGetString(occaJson j);
    type(C_char_ptr) function occaJsonGetString(j) &
                              bind(C, name="occaJsonGetString")
      import occaJson, C_char_ptr
      implicit none
      type(occaJson), value :: j
    end function
    ! ======================================


    ! ---[ Object methods ]-----------------
    ! occaType occaJsonObjectGet(occaJson j,
    !                            const char *key,
    !                            occaType defaultValue);
    type(occaType) function occaJsonObjectGet(j, key, defaultValue) &
                            bind(C, name="occaJsonObjectGet")
      import occaType, occaJson, C_char
      implicit none
      type(occaJson), value :: j
      character(len=1,kind=C_char), dimension(*), intent(in) :: key
      type(occaType), value :: defaultValue
    end function

    ! void occaJsonObjectSet(occaJson j, const char *key, occaType value);
    subroutine occaJsonObjectSet(j, key, value) &
               bind(C, name="occaJsonObjectSet")
      import occaType, occaJson, C_char
      implicit none
      type(occaJson), value :: j
      character(len=1,kind=C_char), dimension(*), intent(in) :: key
      type(occaType), value :: value
    end subroutine

    ! bool occaJsonObjectHas(occaJson j, const char *key);
    logical(kind=C_bool) function occaJsonObjectHas(j, key) &
                                  bind(C, name="occaJsonObjectHas")
      import occaJson, C_bool, C_char
      implicit none
      type(occaJson), value :: j
      character(len=1,kind=C_char), dimension(*), intent(in) :: key
    end function
    ! ======================================


    ! ---[ Array methods ]------------------
    ! int occaJsonArraySize(occaJson j);
    integer(C_int) function occaJsonArraySize(j) &
                            bind(C, name="occaJsonArraySize")
      import occaJson, C_int
      implicit none
      type(occaJson), value :: j
    end function

    ! occaType occaJsonArrayGet(occaJson j, const int index);
    type(occaType) function occaJsonArrayGet(j, index) &
                            bind(C, name="occaJsonArrayGet")
      import occaType, occaJson, C_int
      implicit none
      type(occaJson), value :: j
      integer(C_int), value, intent(in) :: index
    end function

    ! void occaJsonArrayPush(occaJson j, occaType value);
    subroutine occaJsonArrayPush(j, value) &
               bind(C, name="occaJsonArrayPush")
      import occaType, occaJson
      implicit none
      type(occaJson), value :: j
      type(occaType), value :: value
    end subroutine

    ! void occaJsonArrayPop(occaJson j);
    subroutine occaJsonArrayPop(j) bind(C, name="occaJsonArrayPop")
      import occaJson
      implicit none
      type(occaJson), value :: j
    end subroutine

    ! void occaJsonArrayInsert(occaJson j, const int index, occaType value);
    subroutine occaJsonArrayInsert(j, index, value) &
               bind(C, name="occaJsonArrayInsert")
      import occaType, occaJson, C_int
      implicit none
      type(occaJson), value :: j
      integer(C_int), value, intent(in) :: index
      type(occaType), value :: value
    end subroutine

    ! void occaJsonArrayClear(occaJson j);
    subroutine occaJsonArrayClear(j) bind(C, name="occaJsonArrayClear")
      import occaJson
      implicit none
      type(occaJson), value :: j
    end subroutine
    ! ======================================
  end interface

end module occa_json_m
