program main
  use occa
  implicit none

  integer(4) :: platformID = 0, deviceID = 0

  type(occaDevice) :: device

  device = occaGetDevice('OpenCL', platformID, deviceID)

  call occaDeviceFree(device)
end program main
