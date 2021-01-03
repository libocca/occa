program main
  use occa

  implicit none

  integer :: alloc_err, i
  integer(occaUDim_t) :: iu
  integer(occaUDim_t) :: entries = 5
  character(len=1024) :: arg

  real(C_float), allocatable, target :: a(:), b(:), ab(:)

  ! OCCA streams, kernel, memory and property objects
  type(occaStream)     :: streamA, streamB
  type(occaKernel)     :: addVectors
  type(occaMemory)     :: o_a, o_b, o_ab

  ! Parse command arguments
  i = 1
  do while (i .le. command_argument_count())
    call get_command_argument(i, arg)

    select case (arg)
      case ("-v", "--verbose")
        call occaJsonObjectSet(occaSettings(), "kernel/verbose", occaTrue)
      case ("-h", "--help")
        call print_help()
        stop
      case default
        write(*,'(2a, /)') "Unrecognised command-line option: ", arg
        stop
    end select
    i = i+1
  end do

  ! Allocate host memory
  allocate(a(1:entries), b(1:entries), ab(1:entries), stat=alloc_err)
  if (alloc_err /= 0) stop "*** Not enough memory ***"

  ! Initialise host arrays
  do iu=1,entries
    a(iu) = real(iu)-1
    b(iu) = 2-real(iu)
  end do
  ab = 0

  ! Print device infos
  call occaPrintModeInfo()

  ! Get/create OCCA streams
  streamA = occaGetStream()
  streamB = occaCreateStream(occaDefault)

  ! Allocate memory on the device
  o_a  = occaTypedMalloc(entries, occaDtypeFloat, C_NULL_ptr, occaDefault)
  o_b  = occaTypedMalloc(entries, occaDtypeFloat, C_NULL_ptr, occaDefault)
  o_ab = occaTypedMalloc(entries, occaDtypeFloat, C_NULL_ptr, occaDefault)

  addVectors = occaBuildKernel(F_C_str("addVectors.okl"), &
                               F_C_str("addVectors"), &
                               occaDefault)

  ! Copy memory to the device
  call occaCopyPtrToMem(o_a, C_loc(a), entries*C_float, 0_occaUDim_t, occaDefault)
  call occaCopyPtrToMem(o_b, C_loc(b), occaAllBytes   , 0_occaUDim_t, occaDefault)

  ! Set stream and launch device kernel
  call occaSetStream(streamA)
  call occaKernelRun(addVectors, occaInt(entries), o_a, o_b, o_ab)

  call occaSetStream(streamB)
  call occaKernelRun(addVectors, occaInt(entries), o_a, o_b, o_ab)

  ! Copy result to the host
  call occaCopyMemToPtr(C_loc(ab), o_ab, occaAllBytes, 0_occaUDim_t, occaDefault)

  ! Assert values
  do iu=1,entries
    write(*,'(a,i2,a,f3.1)') "ab(", iu, ") = ", ab(iu)
  end do
  do iu=1,entries
    if (abs(ab(iu) - (a(iu) + b(iu))) > 1.0e-8) stop "*** Wrong result ***"
  end do

  ! Free host memory
  deallocate(a, b, ab, stat=alloc_err)
  if (alloc_err /= 0) stop "*** Deallocation not successful ***"

  ! Free device memory and OCCA objects
  call occaFree(addVectors)
  call occaFree(o_a)
  call occaFree(o_b)
  call occaFree(o_ab)

contains
  subroutine print_help()
    write(*,'(a, /)') "Example showing the use of multiple device streams"
    write(*,'(a, /)') "command-line options:"
    write(*,'(a)')    "  -v, --verbose     Compile kernels in verbose mode"
    write(*,'(a)')    "  -h, --help        Print this information and exit"
  end subroutine print_help
end program main
