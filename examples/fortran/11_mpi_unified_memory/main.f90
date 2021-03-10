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

  ! OCCA kernel, memory and property objects
  type(occaKernel)     :: addVectors
  type(C_void_ptr)     :: a, b, ab
  real(C_float), pointer :: a_ptr(:), b_ptr(:), ab_ptr(:)

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
  !info = "{mode: 'OpenMP', schedule: 'compact', chunk: 2}"
  !info(0) = "{mode: 'OpenMP', schedule: 'compact', chunk: 2}"
  !info(1) = "{mode: 'CUDA'  , device_id: 0}"
  info(0) = "{mode: 'OpenMP', schedule: 'compact', chunk: 2}"
  info(1) = "{mode: 'OpenCL', platform_id: 0, device_id: 0}"

  ! Print device infos
  if(myid == 0) then
    call occaPrintModeInfo()
  endif
  call MPI_Barrier(gcomm, ierr)

  ! Create OCCA device
  call occaSetDeviceFromString(F_C_str(info(myid)))

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
    b_ptr(iu) = myid-real(iu)
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

  ! Free device memory and OCCA objects
  call occaFree(addVectors)
  call occaFreeUvaPtr(a)
  call occaFreeUvaPtr(b)
  call occaFreeUvaPtr(ab)

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
