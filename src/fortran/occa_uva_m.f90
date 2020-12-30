module occa_uva_m
  ! occa/c/uva.h

  use occa_types_m

  implicit none

  interface
    ! bool occaIsManaged(void *ptr);
    logical(kind=C_bool) function occaIsManaged(ptr) &
                                  bind(C, name="occaIsManaged")
      import C_void_ptr, C_bool
      implicit none
      type(C_void_ptr), value :: ptr
    end function
    ! void occaStartManaging(void *ptr);
    subroutine occaStartManaging(ptr) bind(C, name="occaStartManaging")
      import C_void_ptr
      implicit none
      type(C_void_ptr), value :: ptr
    end subroutine
    ! void occaStopManaging(void *ptr);
    subroutine occaStopManaging(ptr) bind(C, name="occaStopManaging")
      import C_void_ptr
      implicit none
      type(C_void_ptr), value :: ptr
    end subroutine

    ! void occaSyncToDevice(void *ptr, const occaUDim_t bytes);
    subroutine occaSyncToDevice(ptr, bytes) bind(C, name="occaSyncToDevice")
      import C_void_ptr, occaUDim_t
      implicit none
      type(C_void_ptr), value :: ptr
      integer(occaUDim_t), value :: bytes
    end subroutine
    ! void occaSyncToHost(void *ptr, const occaUDim_t bytes);
    subroutine occaSyncToHost(ptr, bytes) bind(C, name="occaSyncToHost")
      import C_void_ptr, occaUDim_t
      implicit none
      type(C_void_ptr), value :: ptr
      integer(occaUDim_t), value :: bytes
    end subroutine

    ! bool occaNeedsSync(void *ptr);
    logical(kind=C_bool) function occaNeedsSync(ptr) &
                                  bind(C, name="occaNeedsSync")
      import C_void_ptr, C_bool
      implicit none
      type(C_void_ptr), value :: ptr
    end function
    ! void occaSync(void *ptr);
    subroutine occaSync(ptr) bind(C, name="occaSync")
      import C_void_ptr
      implicit none
      type(C_void_ptr), value :: ptr
    end subroutine
    ! void occaDontSync(void *ptr);
    subroutine occaDontSync(ptr) bind(C, name="occaDontSync")
      import C_void_ptr
      implicit none
      type(C_void_ptr), value :: ptr
    end subroutine

    ! void occaFreeUvaPtr(void *ptr);
    subroutine occaFreeUvaPtr(ptr) bind(C, name="occaFreeUvaPtr")
      import C_void_ptr
      implicit none
      type(C_void_ptr), value :: ptr
    end subroutine
  end interface

end module occa_uva_m
