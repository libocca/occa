program parallel_hello_world

  use omp_lib
  implicit none
  integer :: i

  i = -1
  i = omp_get_thread_num()
  if (i == -1) stop "*** omp_get_thread_num error ***"

  stop 0

end
