module occa_scope_m
  ! occa/c/scope.h

  use occa_types_m

  implicit none

  interface
    ! occaScope occaCreateScope(occaJson props);
    type(occaScope) function occaCreateScope(props) &
                             bind(C, name="occaCreateScope")
      import occaScope, occaJson
      implicit none
      type(occaJson), value :: props
    end function

    ! void occaScopeAdd(occaScope scope, const char *name, occaType value);
    subroutine occaScopeAdd(scope, name, value) bind(C, name="occaScopeAdd")
      import C_char, occaScope, occaType
      implicit none
      type(occaScope), value :: scope
      character(len=1,kind=C_char), dimension(*), intent(in) :: name
      type(occaType), value :: value
    end subroutine

    ! void occaScopeAddConst(occaScope scope, const char *name, occaType value);
    subroutine occaScopeAddConst(scope, name, value) &
               bind(C, name="occaScopeAddConst")
      import C_char, occaScope, occaType
      implicit none
      type(occaScope), value :: scope
      character(len=1,kind=C_char), dimension(*), intent(in) :: name
      type(occaType), value :: value
    end subroutine
  end interface

end module occa_scope_m
