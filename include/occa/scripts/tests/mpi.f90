program mpi_fortran

  use mpi
  implicit none
  integer :: ierr

  call MPI_Init(ierr)
  if (ierr /= 0) stop "*** MPI_Init error ***"

  call MPI_Finalize(ierr)
  if (ierr /= 0) stop "*** MPI_Finalize error ***"

  stop 0

end program
