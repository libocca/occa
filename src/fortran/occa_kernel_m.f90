! --------------[ DO NOT EDIT ]--------------
!  THIS IS AN AUTOMATICALLY GENERATED FILE
!  EDIT:
!    scripts/codegen/create_fortran_kernel_interface.py
!    scripts/codegen/occa_kernel_m.f90.in
! ===========================================

module occa_kernel_m
  ! occa/c/kernel.h

  use occa_types_m

  implicit none

  interface
    ! bool occaKernelIsInitialized(occaKernel kernel);
    logical(kind=C_bool) function occaKernelIsInitialized(kernel) &
                                  bind(C, name="occaKernelIsInitialized")
      import occaKernel, C_bool
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! occaJson occaKernelGetProperties(occaKernel kernel);
    type(occaJson) function occaKernelGetProperties(kernel) &
         bind(C, name="occaKernelGetProperties")
      import occaKernel, occaJson
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! occaDevice occaKernelGetDevice(occaKernel kernel);
    type(occaDevice) function occaKernelGetDevice(kernel) &
                              bind(C, name="occaKernelGetDevice")
      import occaKernel, occaDevice
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! const char* occaKernelName(occaKernel kernel);
    type(C_char_ptr) function occaKernelName(kernel) &
                              bind(C, name="occaKernelName")
      import occaKernel, C_char_ptr
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! const char* occaKernelSourceFilename(occaKernel kernel);
    type(C_char_ptr) function occaKernelSourceFilename(kernel) &
                              bind(C, name="occaKernelSourceFilename")
      import occaKernel, C_char_ptr
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! const char* occaKernelBinaryFilename(occaKernel kernel);
    type(C_char_ptr) function occaKernelBinaryFilename(kernel) &
                              bind(C, name="occaKernelBinaryFilename")
      import occaKernel, C_char_ptr
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! const char* occaKernelHash(occaKernel kernel);
    type(C_char_ptr) function occaKernelHash(kernel) &
                              bind(C, name="occaKernelHash")
      import occaKernel, C_char_ptr
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! const char* occaKernelFullHash(occaKernel kernel);
    type(C_char_ptr) function occaKernelFullHash(kernel) &
                              bind(C, name="occaKernelFullHash")
      import occaKernel, C_char_ptr
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! bool occaKernelMaxDims(occaKernel kernel);
    logical(kind=C_bool) function occaKernelMaxDims(kernel) &
                                  bind(C, name="occaKernelMaxDims")
      import occaKernel, C_bool
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! occaDim occaKernelMaxOuterDims(occaKernel kernel);
    type(occaDim) function occaKernelMaxOuterDims(kernel) &
                           bind(C, name="occaKernelMaxOuterDims")
      import occaKernel, occaDim
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! occaDim occaKernelMaxInnerDims(occaKernel kernel);
    type(occaDim) function occaKernelMaxInnerDims(kernel) &
                           bind(C, name="occaKernelMaxInnerDims")
      import occaKernel, occaDim
      implicit none
      type(occaKernel), value :: kernel
    end function

    ! void occaKernelSetRunDims(occaKernel kernel,
    !                           occaDim outerDims,
    !                           occaDim innerDims);
    subroutine occaKernelSetRunDims(kernel, outerDims, innerDims) &
               bind(C, name="occaKernelSetRunDims")
      import occaKernel, occaDim
      implicit none
      type(occaKernel), value :: kernel
      type(occaDim), value :: outerDims, innerDims
    end subroutine

    ! void occaKernelPushArg(occaKernel kernel, occaType arg);
    subroutine occaKernelPushArg(kernel, arg) bind(C, name="occaKernelPushArg")
      import occaKernel, occaType
      implicit none
      type(occaKernel), value :: kernel
      type(occaType), value :: arg
    end subroutine

    ! void occaKernelClearArgs(occaKernel kernel);
    subroutine occaKernelClearArgs(kernel) bind(C, name="occaKernelClearArgs")
      import occaKernel, occaType
      implicit none
      type(occaKernel), value :: kernel
    end subroutine

    ! void occaKernelRunFromArgs(occaKernel kernel);
    subroutine occaKernelRunFromArgs(kernel) &
               bind(C, name="occaKernelRunFromArgs")
      import occaKernel
      implicit none
      type(occaKernel), value :: kernel
    end subroutine

    ! void occaKernelVaRun(occaKernel kernel, const int argc, va_list args);
    ! NOTE: There is no clean way to implement this in Fortran as there is no
    !       clean way to map va_list (https://en.wikipedia.org/wiki/Stdarg.h)
  end interface

  interface occaKernelRunN
    ! `occaKernelRun` is a variadic macro in C:
    ! void occaKernelRunN(occaKernel kernel, const int argc, ...);
    !
    subroutine occaKernelRunN01(kernel, argc, arg01) &
               bind(C, name="occaKernelRunF01")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01
    end subroutine

    subroutine occaKernelRunN02(kernel, argc, arg01, arg02) &
               bind(C, name="occaKernelRunF02")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02
    end subroutine

    subroutine occaKernelRunN03(kernel, argc, arg01, arg02, arg03) &
               bind(C, name="occaKernelRunF03")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03
    end subroutine

    subroutine occaKernelRunN04(kernel, argc, arg01, arg02, arg03, arg04) &
               bind(C, name="occaKernelRunF04")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04
    end subroutine

    subroutine occaKernelRunN05(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05) &
               bind(C, name="occaKernelRunF05")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05
    end subroutine

    subroutine occaKernelRunN06(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06) &
               bind(C, name="occaKernelRunF06")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06
    end subroutine

    subroutine occaKernelRunN07(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07) &
               bind(C, name="occaKernelRunF07")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07
    end subroutine

    subroutine occaKernelRunN08(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08) &
               bind(C, name="occaKernelRunF08")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08
    end subroutine

    subroutine occaKernelRunN09(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09) &
               bind(C, name="occaKernelRunF09")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09
    end subroutine

    subroutine occaKernelRunN10(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10) &
               bind(C, name="occaKernelRunF10")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10
    end subroutine

    subroutine occaKernelRunN11(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10, arg11) &
               bind(C, name="occaKernelRunF11")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10, arg11
    end subroutine

    subroutine occaKernelRunN12(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10, arg11, arg12) &
               bind(C, name="occaKernelRunF12")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10, arg11, arg12
    end subroutine

    subroutine occaKernelRunN13(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10, arg11, arg12, &
                                              arg13) &
               bind(C, name="occaKernelRunF13")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10, arg11, arg12, &
                               arg13
    end subroutine

    subroutine occaKernelRunN14(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10, arg11, arg12, &
                                              arg13, arg14) &
               bind(C, name="occaKernelRunF14")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10, arg11, arg12, &
                               arg13, arg14
    end subroutine

    subroutine occaKernelRunN15(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10, arg11, arg12, &
                                              arg13, arg14, arg15) &
               bind(C, name="occaKernelRunF15")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10, arg11, arg12, &
                               arg13, arg14, arg15
    end subroutine

    subroutine occaKernelRunN16(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10, arg11, arg12, &
                                              arg13, arg14, arg15, arg16) &
               bind(C, name="occaKernelRunF16")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10, arg11, arg12, &
                               arg13, arg14, arg15, arg16
    end subroutine

    subroutine occaKernelRunN17(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10, arg11, arg12, &
                                              arg13, arg14, arg15, arg16, &
                                              arg17) &
               bind(C, name="occaKernelRunF17")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10, arg11, arg12, &
                               arg13, arg14, arg15, arg16, arg17
    end subroutine

    subroutine occaKernelRunN18(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10, arg11, arg12, &
                                              arg13, arg14, arg15, arg16, &
                                              arg17, arg18) &
               bind(C, name="occaKernelRunF18")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10, arg11, arg12, &
                               arg13, arg14, arg15, arg16, arg17, arg18
    end subroutine

    subroutine occaKernelRunN19(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10, arg11, arg12, &
                                              arg13, arg14, arg15, arg16, &
                                              arg17, arg18, arg19) &
               bind(C, name="occaKernelRunF19")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10, arg11, arg12, &
                               arg13, arg14, arg15, arg16, arg17, arg18, &
                               arg19
    end subroutine

    subroutine occaKernelRunN20(kernel, argc, arg01, arg02, arg03, arg04, &
                                              arg05, arg06, arg07, arg08, &
                                              arg09, arg10, arg11, arg12, &
                                              arg13, arg14, arg15, arg16, &
                                              arg17, arg18, arg19, arg20) &
               bind(C, name="occaKernelRunF20")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, &
                               arg07, arg08, arg09, arg10, arg11, arg12, &
                               arg13, arg14, arg15, arg16, arg17, arg18, &
                               arg19, arg20
    end subroutine
  end interface

  interface occaKernelRun
    module procedure occaKernelRun01
    module procedure occaKernelRun02
    module procedure occaKernelRun03
    module procedure occaKernelRun04
    module procedure occaKernelRun05
    module procedure occaKernelRun06
    module procedure occaKernelRun07
    module procedure occaKernelRun08
    module procedure occaKernelRun09
    module procedure occaKernelRun10
    module procedure occaKernelRun11
    module procedure occaKernelRun12
    module procedure occaKernelRun13
    module procedure occaKernelRun14
    module procedure occaKernelRun15
    module procedure occaKernelRun16
    module procedure occaKernelRun17
    module procedure occaKernelRun18
    module procedure occaKernelRun19
    module procedure occaKernelRun20
  end interface

contains

  subroutine occaKernelRun01(kernel, arg01)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01
    call occaKernelRunN01(kernel,  1, arg01)
  end subroutine

  subroutine occaKernelRun02(kernel, arg01, arg02)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02
    call occaKernelRunN02(kernel,  2, arg01, arg02)
  end subroutine

  subroutine occaKernelRun03(kernel, arg01, arg02, arg03)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03
    call occaKernelRunN03(kernel,  3, arg01, arg02, arg03)
  end subroutine

  subroutine occaKernelRun04(kernel, arg01, arg02, arg03, arg04)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04
    call occaKernelRunN04(kernel,  4, arg01, arg02, arg03, arg04)
  end subroutine

  subroutine occaKernelRun05(kernel, arg01, arg02, arg03, arg04, arg05)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05
    call occaKernelRunN05(kernel,  5, arg01, arg02, arg03, arg04, arg05)
  end subroutine

  subroutine occaKernelRun06(kernel, arg01, arg02, arg03, arg04, arg05, arg06)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06
    call occaKernelRunN06(kernel,  6, arg01, arg02, arg03, arg04, arg05, &
                                      arg06)
  end subroutine

  subroutine occaKernelRun07(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07
    call occaKernelRunN07(kernel,  7, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07)
  end subroutine

  subroutine occaKernelRun08(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08
    call occaKernelRunN08(kernel,  8, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08)
  end subroutine

  subroutine occaKernelRun09(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09
    call occaKernelRunN09(kernel,  9, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09)
  end subroutine

  subroutine occaKernelRun10(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10
    call occaKernelRunN10(kernel, 10, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10)
  end subroutine

  subroutine occaKernelRun11(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10, arg11)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10, arg11
    call occaKernelRunN11(kernel, 11, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10, &
                                      arg11)
  end subroutine

  subroutine occaKernelRun12(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10, arg11, arg12)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10, arg11, arg12
    call occaKernelRunN12(kernel, 12, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10, &
                                      arg11, arg12)
  end subroutine

  subroutine occaKernelRun13(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10, arg11, arg12, &
                                     arg13)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10, arg11, arg12, arg13
    call occaKernelRunN13(kernel, 13, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10, &
                                      arg11, arg12, arg13)
  end subroutine

  subroutine occaKernelRun14(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10, arg11, arg12, &
                                     arg13, arg14)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10, arg11, arg12, arg13, arg14
    call occaKernelRunN14(kernel, 14, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10, &
                                      arg11, arg12, arg13, arg14)
  end subroutine

  subroutine occaKernelRun15(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10, arg11, arg12, &
                                     arg13, arg14, arg15)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10, arg11, arg12, arg13, arg14, &
                             arg15
    call occaKernelRunN15(kernel, 15, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10, &
                                      arg11, arg12, arg13, arg14, arg15)
  end subroutine

  subroutine occaKernelRun16(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10, arg11, arg12, &
                                     arg13, arg14, arg15, arg16)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10, arg11, arg12, arg13, arg14, &
                             arg15, arg16
    call occaKernelRunN16(kernel, 16, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10, &
                                      arg11, arg12, arg13, arg14, arg15, &
                                      arg16)
  end subroutine

  subroutine occaKernelRun17(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10, arg11, arg12, &
                                     arg13, arg14, arg15, arg16, arg17)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10, arg11, arg12, arg13, arg14, &
                             arg15, arg16, arg17
    call occaKernelRunN17(kernel, 17, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10, &
                                      arg11, arg12, arg13, arg14, arg15, &
                                      arg16, arg17)
  end subroutine

  subroutine occaKernelRun18(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10, arg11, arg12, &
                                     arg13, arg14, arg15, arg16, arg17, arg18)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10, arg11, arg12, arg13, arg14, &
                             arg15, arg16, arg17, arg18
    call occaKernelRunN18(kernel, 18, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10, &
                                      arg11, arg12, arg13, arg14, arg15, &
                                      arg16, arg17, arg18)
  end subroutine

  subroutine occaKernelRun19(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10, arg11, arg12, &
                                     arg13, arg14, arg15, arg16, arg17, arg18, &
                                     arg19)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10, arg11, arg12, arg13, arg14, &
                             arg15, arg16, arg17, arg18, arg19
    call occaKernelRunN19(kernel, 19, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10, &
                                      arg11, arg12, arg13, arg14, arg15, &
                                      arg16, arg17, arg18, arg19)
  end subroutine

  subroutine occaKernelRun20(kernel, arg01, arg02, arg03, arg04, arg05, arg06, &
                                     arg07, arg08, arg09, arg10, arg11, arg12, &
                                     arg13, arg14, arg15, arg16, arg17, arg18, &
                                     arg19, arg20)
    type(occaKernel), value :: kernel
    type(occaType), value :: arg01, arg02, arg03, arg04, arg05, arg06, arg07, &
                             arg08, arg09, arg10, arg11, arg12, arg13, arg14, &
                             arg15, arg16, arg17, arg18, arg19, arg20
    call occaKernelRunN20(kernel, 20, arg01, arg02, arg03, arg04, arg05, &
                                      arg06, arg07, arg08, arg09, arg10, &
                                      arg11, arg12, arg13, arg14, arg15, &
                                      arg16, arg17, arg18, arg19, arg20)
  end subroutine

end module occa_kernel_m
