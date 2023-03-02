module occa_stream_m
! occa/c/device.h

use occa_types_m
implicit none

interface
  ! void occaStreamFinish(occaStream stream);
  subroutine occaStreamFinish(stream) bind(C, name="occaStreamFinish")
    import occaStream
    implicit none
    type(occaStream), value :: stream
  end subroutine
end interface

end module occa_stream_m