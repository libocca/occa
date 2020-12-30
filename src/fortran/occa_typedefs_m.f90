module occa_typedefs_m

  use, intrinsic :: iso_c_binding, &
  ! C type aliases for pointer derived types:
      C_ptr => C_ptr, &
      C_char_ptr => C_ptr, &
      C_void_ptr => C_ptr, &
      C_int64_t => C_int64_t, &
      occaDim_t => C_int64_t, &
      occaUDim_t => C_int64_t

  implicit none

  !-----------------------------------------------------------------------------
  ! Type definitions mapping the `struct` in the C layer to a defined type in
  ! Fortran. For a basic example see:
  !   https://gcc.gnu.org/onlinedocs/gfortran/Derived-Types-and-struct.html
  !-----------------------------------------------------------------------------
  type, bind(C) :: occaDim
    integer(occaUDim_t) :: x, y, z
  end type occaDim

  type, bind(C) :: occaUnion
    private
    ! Use the largest data member in the C `union`.
    integer(C_int64_t) :: data
  end type occaUnion

  type, bind(C) :: occaType
    integer(C_int) :: magicHeader
    integer(C_int) :: type
    integer(C_int64_t) :: bytes
    logical(kind=C_bool) needsFree

    ! In the C layer, this is a `union`, which does not have a Fortran
    ! counterpart. Thus use a defined type, corresponding to a `struct` in C,
    ! with the largest data member in the C `union`.
    type(occaUnion) :: value
  end type occaType

end module occa_typedefs_m
