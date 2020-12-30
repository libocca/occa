program compiler_fortran

  implicit none

#if   defined(__INTEL_COMPILER)
  write(*,'(a)') 'INTEL'
  stop
#elif defined(__PGI)
  write(*,'(a)') 'PGI'
  stop
#elif defined(_CRAYFTN)
  write(*,'(a)') 'CRAY'
  stop
#elif defined(__GFORTRAN__)
  write(*,'(a)') 'GCC'
  stop
#elif defined(__ibmxl__)
  write(*,'(a)') 'IBM'
  stop
#elif defined(__PATHSCALE__)
  write(*,'(a)') 'PATHSCALE'
  stop
#endif

  write(*,'(a)') 'N/A'
  stop

end program
