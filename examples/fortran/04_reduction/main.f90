program main
  use occa

  implicit none

  integer :: alloc_err, i
  integer(occaUDim_t), parameter :: entries = 10000
  integer(occaUDim_t), parameter :: blk = 256
  integer(occaUDim_t), parameter :: blks = (entries + blk - 1)/blk
  character(len=1024) :: arg, info

  real(C_float), allocatable, target :: vec(:), blockSum(:)
  real(C_float) :: sig

  ! OCCA device, kernel, memory and property objects
  type(occaKernel) :: reduction
  type(occaMemory) :: o_vec, o_blockSum
  type(occaJson)   :: reductionProps

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
  allocate(vec(1:entries), blockSum(1:blks), stat=alloc_err)
  if (alloc_err /= 0) stop "*** Not enough memory ***"

  ! Initialize host memory
  vec = 1
  blockSum = 0
  sig = sum(vec)

  ! Print device infos
  call occaPrintModeInfo()

  ! Create OCCA device
  call occaSetDeviceFromString(F_C_str(info))

  ! Allocate memory on the device
  o_vec      = occaTypedMalloc(entries, occaDtypeFloat, C_NULL_ptr, occaDefault)
  o_blockSum = occaTypedMalloc(blks,    occaDtypeFloat, C_NULL_ptr, occaDefault)

  ! Pass value of 'block' at kernel compile-time
  reductionProps = occaCreateJson()
  call occaJsonObjectSet(reductionProps, F_C_str("defines/block"), occaInt(blk))

  reduction = occaBuildKernel(F_C_str("reduction.okl"), &
                              F_C_str("reduction"), &
                              reductionProps)

  ! Host -> Device
  call occaCopyPtrToMem(o_vec, C_loc(vec), occaAllBytes, 0_occaUDim_t, occaDefault)

  call occaKernelRun(reduction, occaInt(entries), o_vec, o_blockSum)

  ! Host <- Device
  call occaCopyMemToPtr(C_loc(blockSum), o_blockSum, occaAllBytes, 0_occaUDim_t, occaDefault)

  blockSum(1) = sum(blockSum)

  ! Validate
  if (abs(blockSum(1) - sig) > 1.0e-8) then
    write(*,*) "sum      = ", sig
    write(*,*) "blockSum = ", blockSum(1)
    stop "*** Reduction failed ***"
  else
    write(*,*) "Reduction = ", blockSum(1)
  end if

  ! Free host memory
  deallocate(vec, blockSum, stat=alloc_err)
  if (alloc_err /= 0) stop "*** Deallocation not successful ***"

  ! Free device memory and OCCA objects
  call occaFree(reductionProps)
  call occaFree(reduction)
  call occaFree(o_vec)
  call occaFree(o_blockSum)

contains
  subroutine print_help()
    write(*,'(a, /)') "Example of a reduction kernel which sums a vector in parallel"
    write(*,'(a, /)') "command-line options:"
    write(*,'(a)')    "  -v, --verbose     Compile kernels in verbose mode"
    write(*,'(a)')    "  -d, --device      Device properties (default: ""{mode: 'Serial'}"")"
    write(*,'(a)')    "  -h, --help        Print this information and exit"
  end subroutine print_help
end program main
