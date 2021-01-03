program main
  use mpi
  use occa
  use, intrinsic :: iso_fortran_env, only : stdout=>output_unit, &
                                            stderr=>error_unit

  implicit none

  integer :: ierr, id
  integer :: myid, npes, gcomm, tag ! MPI variables
  integer, dimension(2) :: request
  integer, dimension(MPI_STATUS_SIZE) :: status
  integer :: otherID, offset
  integer(occaUDim_t) :: iu
  integer(occaUDim_t) :: entries = 8
  character(len=1024), dimension(0:1) :: info
  real(C_float), dimension(0:1) :: ab_sum
  real(C_float) :: ab_gather

  real(C_float), allocatable, target :: a(:), b(:), ab(:)
  real(C_float), pointer :: ab_ptr(:)

  ! OCCA kernel, memory and property objects
  type(occaDevice)     :: device
  type(occaKernel)     :: addVectors
  type(occaMemory)     :: o_a, o_b, o_ab

  ! Initialise MPI
  gcomm = MPI_COMM_WORLD
  call MPI_Init(ierr)
  if (ierr /= MPI_SUCCESS) call stop_mpi("*** MPI_Init error ***")
  call MPI_Comm_rank(gcomm, myid,  ierr)
  call MPI_Comm_size(gcomm, npes,  ierr)
  if (npes /= 2) then
    call stop_mpi("*** Example expects to run with 2 processes ***", myid)
  end if

  ! Set OCCA device info
  !info = "{mode: 'Serial'}"
  info = "{mode: 'OpenMP', schedule: 'compact', chunk: 2}"

  ! Print device infos
  if(myid == 0) then
    call occaPrintModeInfo()
  endif
  call MPI_Barrier(gcomm, ierr)

  ! Allocate host memory
  allocate(a(1:entries), b(1:entries), ab(1:entries), stat=ierr)
  if (ierr /= 0) call stop_mpi("*** Not enough memory ***", myid)

  ! Initialise host arrays
  do iu=1,entries
    a(iu) = real(iu)-1
    b(iu) = myid-real(iu)
  end do
  ab = 0.0

  ! Create OCCA device
  device = occaCreateDeviceFromString(F_C_str(info(myid)))

  ! Allocate memory on the device
  o_a  = occaTypedMalloc(entries, occaDtypeFloat, C_NULL_ptr, occaDefault)
  o_b  = occaTypedMalloc(entries, occaDtypeFloat, C_NULL_ptr, occaDefault)
  o_ab = occaTypedMalloc(entries, occaDtypeFloat, C_NULL_ptr, occaDefault)

  ! Compile the kernel at run-time
  addVectors = occaBuildKernel(F_C_str("addVectors.okl"), &
                               F_C_str("addVectors"), &
                               occaDefault)

  ! Copy memory to the device
  call occaCopyPtrToMem(o_a, C_loc(a), entries*C_float, 0_occaUDim_t, occaDefault)
  call occaCopyPtrToMem(o_b, C_loc(b), occaAllBytes   , 0_occaUDim_t, occaDefault)

  ! Launch device kernel
  call occaKernelRun(addVectors, occaInt(entries), o_a, o_b, o_ab)

  ! Get a pointer to the device memory
  ! This only works if the device and host share a global address space, and
  ! thus only the 'Serial' and 'OpenMP' modes can be used. For an alternative
  ! implementation which works for other modes as well see the
  ! '11_mpi_unified_memory' example.
  if (occaDeviceHasSeparateMemorySpace(device)) then
    call stop_mpi("*** Example only works in 'Serial' and 'OpenMP' mode ***", myid)
  end if
  if (C_associated(occaMemoryPtr(o_ab, occaDefault))) then
    call C_F_pointer(occaMemoryPtr(o_ab, occaDefault),ab_ptr,[entries])
  else
    ab_ptr => null()
  end if

  ! Send/receive the result array
  otherID = mod(myid + 1, 2)
  offset  = int(entries/2)
  tag     = 123
  request = MPI_REQUEST_NULL
  call MPI_IRecv(ab_ptr(otherID*offset+1), &
                 offset, &
                 MPI_FLOAT, &
                 otherID, &
                 tag, &
                 gcomm, &
                 request(1), &
                 ierr)
  call MPI_ISend(ab_ptr(myid*offset+1), &
                 offset, &
                 MPI_FLOAT, &
                 otherID, &
                 tag, &
                 gcomm, &
                 request(2), &
                 ierr)
  call MPI_Wait(request(1), status, ierr)
  call MPI_Wait(request(2), status, ierr)

  ! Copy result to the host
  call occaCopyMemToPtr(C_loc(ab), o_ab, occaAllBytes, 0_occaUDim_t, occaDefault)

  ! Assert values
  call flush(stdout)
  ab_sum = myid
  ab_gather = sum(ab_ptr)
  call MPI_Gather(ab_gather, 1, MPI_FLOAT, ab_sum, 1, MPI_FLOAT, 0, gcomm, ierr)
  if (myid == 0) then
    if (abs(ab_sum(myid) - ab_sum(otherID)) > 1.0e-8) stop "*** Wrong result ***"
  end if

  ! Print values
  call flush(stdout)
  call MPI_Barrier(gcomm, ierr)
  do id=0,npes-1
    if (id == myid) then
      call flush(stdout)
      do iu=1,entries
      write(stdout,'(a,i1,a,i2,a,f5.1)') "#", id, ": ab(", iu, ") = ", ab_ptr(iu)
      call flush(stdout)
      end do
    end if
    call MPI_Barrier(gcomm, ierr)
  end do

  ! Free host memory
  deallocate(a, b, ab, stat=ierr)
  if (ierr /= 0) call stop_mpi("*** Deallocation not successful ***", myid)

  ! Free device memory and OCCA objects
  call occaFree(addVectors)
  call occaFree(o_a)
  call occaFree(o_b)
  call occaFree(o_ab)
  call occaFree(device)

  ! Cleanup MPI
  call MPI_Finalize(ierr)
  if (ierr /= MPI_SUCCESS) call stop_mpi("*** MPI_Finalize error ***", myid)

contains
  subroutine stop_mpi(error, myid, error_code)
    implicit none

    integer, intent(in), optional :: myid, error_code
    character(len=*), intent(in), optional :: error
    integer :: ierr, ec

    call flush(stdout)
    call MPI_Barrier(gcomm, ierr)

    if(present(error)) then
      if(present(myid)) then
        if(myid == 0) then
          write(stdout,'(a)') ''
          write(stdout,'(a)') error
          write(stdout,'(a)') ''
        end if
      else
        write(stdout,'(a)') ''
        write(stdout,'(a)') error
        write(stdout,'(a)') ''
      end if
    end if

    call flush(stdout)
    call MPI_Barrier(gcomm, ierr)

    if(present(error_code)) then
      ec = error_code
    else
      ec = -1
    end if

    call MPI_Abort(gcomm, ec, ierr)
  end subroutine
end program main
