program main
  use, intrinsic :: iso_c_binding, &
      C_void_ptr => C_ptr
  use occa

  implicit none

  integer :: i
  integer(occaUDim_t) :: iu
  integer(occaUDim_t) :: entries = 5
  character(len=1024) :: arg, info

  ! OCCA device, kernel, memory and property objects
  type(occaKernel)     :: addVectors
  type(C_void_ptr)     :: a, b, ab
  real(C_float), pointer :: a_ptr(:), b_ptr(:), ab_ptr(:)

  ! Set default OCCA device info
  info = "{mode: 'Serial'}"
  !info = "{mode: 'OpenMP', schedule: 'compact', chunk: 10}"
  !info = "{mode: 'CUDA'  , device_id: 0}"
  !info = "{mode: 'OpenCL', platform_id: 0, device_id: 0}"

  ! Parse command arguments
  i = 1
  do while (i .le. command_argument_count())
    call get_command_argument(i, arg)

    select case (arg)
      case ("-v", "--verbose")
        call occaJsonObjectSet(occaSettings(), "kernel/verbose", occaTrue)
      case ("-d", "--device")
        i = i+1
        call get_command_argument(i, info)
      case ("-h", "--help")
        call print_help()
        stop
      case default
        write(*,'(2a, /)') "Unrecognised command-line option: ", arg
        stop
    end select
    i = i+1
  end do

  ! Print device infos
  call occaPrintModeInfo()

  ! Create OCCA device
  call occaSetDeviceFromString(F_C_str(info))

  ! umalloc: [U]nified [M]emory [Alloc]ation
  ! Allocate host memory that auto-syncs with the device between before kernel
  ! calls and occaFinish() if needed.
  a  = occaTypedUMalloc(entries, occaDtypeFloat, C_NULL_ptr, occaDefault)
  b  = occaTypedUMalloc(entries, occaDtypeFloat, C_NULL_ptr, occaDefault)
  ab = occaTypedUMalloc(entries, occaDtypeFloat, C_NULL_ptr, occaDefault)

  ! Assign Fortran pointers to the (host) memory
  if (C_associated(a)) then
    call C_F_pointer(a,a_ptr,[entries])
  else
    a_ptr => null()
  end if
  if (C_associated(b)) then
    call C_F_pointer(b,b_ptr,[entries])
  else
    b_ptr => null()
  end if
  if (C_associated(ab)) then
    call C_F_pointer(ab,ab_ptr,[entries])
  else
    ab_ptr => null()
  end if

  ! Initialise host arrays
  do iu=1,entries
    a_ptr(iu) = real(iu)-1
    b_ptr(iu) = 2-real(iu)
  end do
  ab_ptr = 0

  ! Compile the kernel at run-time
  addVectors = occaBuildKernel(F_C_str("addVectors.okl"), &
                               F_C_str("addVectors"), &
                               occaDefault)

  ! Launch device kernel
  ! Arrays a, b, and ab are now resident on the device
  call occaKernelRun(addVectors, occaInt(entries), occaPtr(a), occaPtr(b), occaPtr(ab))

  ! a and b are const in the kernel, so we can use `dontSync` to manually force
  ! a and b to not sync
  call occaDontSync(a)
  call occaDontSync(b)

  ! Finish work queued up on the device, synchronizing a, b, and ab and making
  ! it safe to use them again
  call occaFinish()

  ! Assert values
  do iu=1,entries
    write(*,'(a,i2,a,f3.1)') "ab(", iu, ") = ", ab_ptr(iu)
  end do
  do iu=1,entries
    if (abs(ab_ptr(iu) - (a_ptr(iu) + b_ptr(iu))) > 1.0e-8) stop "*** Wrong result ***"
  end do

  ! Free device memory and OCCA objects
  call occaFree(addVectors)
  call occaFreeUvaPtr(a)
  call occaFreeUvaPtr(b)
  call occaFreeUvaPtr(ab)

contains
  subroutine print_help()
    write(*,'(a, /)') "Example showing how to use background devices, allowing passing of the device implicitly"
    write(*,'(a, /)') "command-line options:"
    write(*,'(a)')    "  -v, --verbose     Compile kernels in verbose mode"
    write(*,'(a)')    "  -d, --device      Device properties (default: ""{mode: 'Serial'}"")"
    write(*,'(a)')    "  -h, --help        Print this information and exit"
  end subroutine print_help
end program main
