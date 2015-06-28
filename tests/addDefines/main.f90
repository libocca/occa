program main
  use occa
  implicit none

  type(occaKernelInfo) :: kInfo

  kInfo = occaCreateKernelInfo();

  call occaKernelInfoAddDefine(kInfo, "A", 0);
  call occaKernelInfoAddDefine(kInfo, "B", 0.0);
  call occaKernelInfoAddDefine(kInfo, "D", "This is a string");

end program main
