module occaFTypes_m
  implicit none

  type :: occaDevice
     private
     integer, pointer :: p
  end type occaDevice

  type :: occaDim
     private
     integer, pointer :: p
  end type occaDim

  type :: occaKernel
     private
     integer, pointer :: p
  end type occaKernel

  type :: occaMemory
     private
     integer, pointer :: p
  end type occaMemory

  type :: occaType
     private
     integer, pointer :: p
  end type occaType

  type :: occaArgumentList
     private
     integer, pointer :: p
  end type occaArgumentList

  type :: occaStream
     private
     integer, pointer :: p
  end type occaStream

  type :: occaTag
     private
     integer, pointer :: p
  end type occaTag

  type :: occaKernelInfo
     private
     integer, pointer :: p
  end type occaKernelInfo

end module occaFTypes_m
