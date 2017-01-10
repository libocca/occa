program main
  use occa
  implicit none

  integer :: alloc_err

  integer(4) :: i, entries = 5

  integer(4) :: platformID = 0, deviceID = 0, dims

  character(len=1024) :: Serial_Info  : "mode: 'Serial'"
  character(len=1024) :: OpenMP_Info  : "mode: 'OpenMP' , schedule: 'compact', chunk: 10"
  character(len=1024) :: OpenCL_Info  : "mode: 'OpenCL' , platformID: 0, deviceID: 0"
  character(len=1024) :: CUDA_Info    : "mode: 'CUDA'   , deviceID: 0"
  character(len=1024) :: Pthreads_Info: "mode: 'Threads', threadCount: 4, schedule: 'compact', pinnedCores: [0, 0, 1, 1]"

  real(4), allocatable :: a(:), b(:), ab(:)

  type(occaDevice) :: device
  type(occaKernel) :: addVectors
  type(occaMemory) :: o_a, o_b, o_ab

  call occaPrintAvailableDevices

  allocate(a(1:entries), b(1:entries), ab(1:entries), stat = alloc_err)
  if (alloc_err /= 0) stop "*** Not enough memory ***"

  do i=1,entries
    a(i) = i-1
    b(i) = 2-i
    ab(i) = 0
  end do

  device = occaCreateDevice(Serial_Info)

  o_a  = occaDeviceMalloc(device, int(entries,8)*4_8)
  o_b  = occaDeviceMalloc(device, int(entries,8)*4_8)
  o_ab = occaDeviceMalloc(device, int(entries,8)*4_8)

  addVectors = occaDeviceBuildKernel(device, "addVectors.okl", "addVectors")

  call occaCopyPtrToMem(o_a, a(1), int(entries,8)*4_8, 0_8);
  call occaCopyPtrToMem(o_b, b(1));

  call occaKernelRun(addVectors, occaTypeMem_t(entries), o_a, o_b, o_ab)

  call occaCopyMemToPtr(ab(1), o_ab);

  print *,"a = ", a(:)
  print *,"b = ", b(:)
  print *,"ab = ", ab(:)

  deallocate(a, b, ab, stat = alloc_err)
  if (alloc_err /= 0) stop "*** deallocation not successful ***"

  call occaKernelFree(addVectors)
  call occaMemoryFree(o_a)
  call occaMemoryFree(o_b)
  call occaMemoryFree(o_ab)
  call occaDeviceFree(device)
end program main
