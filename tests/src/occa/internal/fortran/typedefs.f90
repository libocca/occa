program main
  use occa

  implicit none

  call occaPrintModeInfo()
  call testOccaTypedef(verbose=1)

  stop 0

contains

  subroutine testOccaTypedef(verbose)
    implicit none

    interface
      ! typedefs_helper.cpp
      logical(kind=C_bool) pure function occaTypeHasCorrectSize(s, &
                                                                magicHeader_s, &
                                                                type_s, &
                                                                bytes_s, &
                                                                needsFree_s, &
                                                                value_s, &
                                                                verbose)  &
                                         bind(C, name="occaTypeHasCorrectSize")
        use, intrinsic :: iso_c_binding, only: C_size_t, C_bool
        implicit none
        integer(C_size_t), value, intent(in) :: s, magicHeader_s, type_s, &
                                                bytes_s, needsFree_s, value_s
        logical(kind=C_bool), value, intent(in) :: verbose
      end function

      pure subroutine printOccaType(value, verbose) &
                      bind(C, name="printOccaType")
        use, intrinsic :: iso_c_binding, only: C_bool
        use occa, only: occaType
        implicit none
        type(occaType), intent(in) :: value
        logical(kind=C_bool), value, intent(in) :: verbose
      end subroutine
    end interface

    integer, intent(in), optional :: verbose
    logical(kind=C_bool) :: ok, vrb
    type(occaType) :: t

    vrb = .false.
    if (present(verbose)) then
      if (verbose > 0) vrb = .true.
    end if

    ! Check occaType size (useful to verify the F <-> C type mapping)
    ok = .false.
    ok = occaTypeHasCorrectSize(C_sizeof(t), &
                                C_sizeof(t%magicHeader), &
                                C_sizeof(t%type), &
                                C_sizeof(t%bytes), &
                                C_sizeof(t%needsFree), &
                                C_sizeof(t%value), &
                                vrb)
    if (.not. ok) then
      stop "*** ERROR ***: `occaType` C and Fortran implementations are &
           &inconsistent!"
    end if

    ! Dump occaDefault data (useful for debugging F <-> C mapping)
    t = occaDefault
    ok = .false.
    ok = occaIsDefault(t)
    call printOccaType(t, verbose=vrb)
    if (.not. ok) then
      stop "*** ERROR ***: Incorrect `occaDefault` type!"
    end if
  end subroutine
end program
