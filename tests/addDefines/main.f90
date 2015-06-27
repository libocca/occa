program main
  use occa
  implicit none

  type(occaKernelInfo) :: kInfo

  kInfo = occaCreateKernelInfo();

  call occaKernelInfoAddDefine(kInfo, "A", 0);
  call occaKernelInfoAddDefine(kInfo, "B", 0.0);
  call occaKernelInfoAddDefine(kInfo, "C", 'C');
  call occaKernelInfoAddDefine(kInfo, "D", "D");

end program main
