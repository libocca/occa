program main
  use occa
  use iso_c_binding

  implicit none

  integer :: alloc_err, i
  integer(occaUDim_t) :: iu
  integer(occaUDim_t) :: entries = 5
  character(len=1024) :: arg, info

  type(C_ptr) :: cptr

  real(C_float), allocatable, target :: a(:), b(:), ab(:)

  ! OCCA device, kernel, memory and property objects
  type(occaDevice)     :: device
  type(occaKernel)     :: addVectors
  type(occaMemory)     :: o_a, o_b, o_ab
  type(occaJson) :: props

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

  ! Create OCCA device
  device = occaCreateDeviceFromString(F_C_str(info))

  ! Print device mode
  ! Use an intermediate variable to avoid an ICE with the 2019 Intel compilers
  cptr = occaDeviceMode(device)
  write(*,'(a,a)') "occaDeviceMode: ", C_F_str(cptr)

  ! Allocate memory on the device
  o_a  = occaDeviceTypedMalloc(device, entries, occaDtypeFloat, C_NULL_ptr, occaDefault)
  o_b  = occaDeviceTypedMalloc(device, entries, occaDtypeFloat, C_NULL_ptr, occaDefault)

  ! We can also allocate memory without a dtype
  ! WARNING: This will disable runtime type checking
  o_ab = occaDeviceMalloc(device, entries*C_float, C_NULL_ptr, occaDefault)

  ! Setup properties that can be passed to the kernel
  props = occaCreateJson()
  call occaJsonObjectSet(props, F_C_str("defines/TILE_SIZE"), occaInt(10))

  ! Compile the kernel at run-time
  addVectors = occaDeviceBuildKernel(device, &
                                     F_C_str("addVectors.okl"), &
                                     F_C_str("addVectors"), &
                                     props)

  ! Copy memory to the device
  call occaCopyPtrToMem(o_a, C_loc(a), entries*C_float, 0_occaUDim_t, occaDefault)
  call occaCopyPtrToMem(o_b, C_loc(b), occaAllBytes   , 0_occaUDim_t, occaDefault)

  ! Launch device kernel
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
  call occaFree(props)
  call occaFree(addVectors)
  call occaFree(o_a)
  call occaFree(o_b)
  call occaFree(o_ab)
  call occaFree(device)

contains
  subroutine print_help()
    write(*,'(a, /)') "Example showing how to statically compile a program"
    write(*,'(a, /)') "command-line options:"
    write(*,'(a)')    "  -v, --verbose     Compile kernels in verbose mode"
    write(*,'(a)')    "  -d, --device      Device properties (default: ""{mode: 'Serial'}"")"
    write(*,'(a)')    "  -h, --help        Print this information and exit"
  end subroutine print_help
end program main
