module occa_properties_m
  ! occa/c/properties.h

  use occa_types_m

  implicit none

  interface
    ! occaProperties occaCreateProperties();
    type(occaProperties) function occaCreateProperties() &
                                  bind(C, name="occaCreateProperties")
      import occaProperties
      implicit none
    end function

    ! occaProperties occaCreatePropertiesFromString(const char *c);
    type(occaProperties) function occaCreatePropertiesFromString(c) &
                                  bind(C, name="occaCreatePropertiesFromString")
      import C_char, occaProperties
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: c
    end function

    ! occaType occaPropertiesGet(occaProperties props,
    !                            const char *key,
    !                            occaType defaultValue);
    type(occaType) function occaPropertiesGet(props, key, defaultValue) &
                            bind(C, name="occaPropertiesGet")
      import C_char, occaProperties, occaType
      implicit none
      type(occaProperties), value :: props
      character(len=1,kind=C_char), dimension(*), intent(in) :: key
      type(occaType), value, intent(in) :: defaultValue
    end function

    ! void occaPropertiesSet(occaProperties props,
    !                        const char *key,
    !                        occaType value);
    subroutine occaPropertiesSet(props, key, value) &
               bind(C, name="occaPropertiesSet")
      import C_char, occaProperties, occaType
      implicit none
      type(occaProperties), value :: props
      character(len=1,kind=C_char), dimension(*), intent(in) :: key
      type(occaType), value :: value
    end subroutine

    ! bool occaPropertiesHas(occaProperties props, const char *key);
    logical(kind=C_bool) function occaPropertiesHas(props, key) &
                                  bind(C, name="occaPropertiesHas")
      import C_bool, C_char, occaProperties
      implicit none
      type(occaProperties), value, intent(in) :: props
      character(len=1,kind=C_char), dimension(*), intent(in) :: key
    end function
  end interface

end module occa_properties_m
