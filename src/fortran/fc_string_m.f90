module fc_string_m
  use, intrinsic :: iso_c_binding, &
  ! C type aliases for pointer derived types:
      C_ptr => C_ptr, &
      C_char_ptr => C_ptr

  implicit none

  private

  interface
    !---------------------------------------------------------------------------
    ! <string.h>
    !---------------------------------------------------------------------------
    ! extern size_t strlen (const char *s)
    integer(C_size_t) pure function C_strlen(s) bind(C, name="strlen")
      import C_char_ptr, C_size_t
      implicit none
      type(C_char_ptr), value, intent(in) :: s
    end function C_strlen
  end interface

  public :: C_F_str, F_C_str

contains

  ! HACK: For some reason, C_associated was not defined as pure.
  pure function C_associated_pure(ptr) result(a)
    type(C_ptr), intent(in) :: ptr
    integer(C_intptr_t) :: iptr
    logical a
    iptr = transfer(ptr,iptr)
    a = (iptr /= 0)
  end function C_associated_pure

  pure function C_strlen_safe(s) result(length)
    integer(C_size_t) :: length
    type(C_char_ptr), value, intent(in) :: s
    if (.not. C_associated_pure(s)) then
      length = 0
    else
      length = C_strlen(s)
    end if
  end function C_strlen_safe

  function C_F_str(C_string) result(F_string)
    type(C_char_ptr), intent(in) :: C_string
    character(len=C_strlen_safe(C_string)) :: F_string
    character(len=1,kind=C_char), dimension(:), pointer :: p_chars
    integer :: i, length
    length = len(F_string)
    if (length/=0) then
      call C_F_pointer(C_string,p_chars,[length])
      do i=1,length
        F_string(i:i) = p_chars(i)
      end do
    end if
  end function

  pure function F_C_str(F_string) result(C_string)
    character(len=*,kind=C_char), intent(in) :: F_string
    character(len=:,kind=C_char), allocatable :: C_string
    integer :: length
    length = len_trim(F_string)+1
    allocate(character(len=length) :: C_string)
    C_string = trim(F_string)//C_NULL_char
  end function

end module fc_string_m
