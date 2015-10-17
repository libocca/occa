module occa
  use occaFTypes_m
  use ISO_C_BINDING

  implicit none

  integer(C_INT), bind(C, name="occaUsingOKL")    :: occaUsingOKL
  integer(C_INT), bind(C, name="occaUsingOFL")    :: occaUsingOFL
  integer(C_INT), bind(C, name="occaUsingNative") :: occaUsingNative

  ! ---[ TypesCasting ]------------------
  interface occaTypeMem_t
     module procedure occaTypeMem_int4_c
     ! module procedure occaType_int8_c
     module procedure occaTypeMem_real4_c
     module procedure occaTypeMem_real8_c
     module procedure occaTypeMem_str_c
  end interface occaTypeMem_t

  ! ---[ Globals ]----------------------
  interface occaSetVerboseCompilation
     subroutine occaSetVerboseCompilation_fc(value)
       use occaFTypes_m

       implicit none
       logical(1), intent(in) :: value
     end subroutine occaSetVerboseCompilation_fc
  end interface occaSetVerboseCompilation

  ! ---[ Device ]-----------------------
  interface occaPrintAvailableDevices
     subroutine occaPrintAvailableDevices_fc()
       use occaFTypes_m

       implicit none
     end subroutine occaPrintAvailableDevices_fc
  end interface occaPrintAvailableDevices

  interface occaDeviceSetCompiler
     subroutine occaDeviceSetCompiler_fc(device, compiler)
       use occaFTypes_m

       implicit none
       type(occaDevice), intent(inout) :: device
       character(len=*), intent(in)    :: compiler
     end subroutine occaDeviceSetCompiler_fc
  end interface occaDeviceSetCompiler

  interface occaDeviceSetCompilerFlags
     subroutine occaDeviceSetCompilerFlags_fc(device, compilerFlags)
       use occaFTypes_m

       implicit none
       type(occaDevice), intent(inout) :: device
       character(len=*), intent(in)    :: compilerFlags
     end subroutine occaDeviceSetCompilerFlags_fc
  end interface occaDeviceSetCompilerFlags

  interface occaCreateDevice
     module procedure occaCreateDevice_func
     module procedure occaCreateDeviceFromArgs_func
     module procedure occaCreateDeviceFromInfo_func
  end interface occaCreateDevice

  interface occaDeviceBuildKernel
     module procedure occaDeviceBuildKernelNoKernelInfo_func
     module procedure occaDeviceBuildKernel_func
  end interface occaDeviceBuildKernel

  interface occaDeviceBuildKernelFromSource
     module procedure occaDeviceBuildKernelFromSourceNoKernelInfo_func
     module procedure occaDeviceBuildKernelFromSource_func
  end interface occaDeviceBuildKernelFromSource

  interface occaDeviceBuildKernelFromString
     module procedure occaDeviceBuildKernelFromString_func
     module procedure occaDeviceBuildKernelFromStringNoArgs_func
     module procedure occaDeviceBuildKernelFromStringNoKernelInfo_func
  end interface occaDeviceBuildKernelFromString

  interface occaDeviceBuildKernelFromBinary
     module procedure occaDeviceBuildKernelFromBinary_func
  end interface occaDeviceBuildKernelFromBinary

  interface occaDeviceMalloc
     module procedure occaDeviceMalloc_null
     module procedure occaDeviceMalloc_int4
     module procedure occaDeviceMalloc_int8
     module procedure occaDeviceMalloc_real4
     module procedure occaDeviceMalloc_real8
     module procedure occaDeviceMalloc_char
  end interface occaDeviceMalloc

  !  interface occaDeviceManagedAlloc
  !    module procedure occaDeviceManagedAlloc_null
  !    module procedure occaDeviceManagedAlloc_int4
  !    module procedure occaDeviceManagedAlloc_int8
  !    module procedure occaDeviceManagedAlloc_real4
  !    module procedure occaDeviceManagedAlloc_real8
  !    module procedure occaDeviceManagedAlloc_char
  ! end interface occaDeviceManagedAlloc

  interface occaDeviceTextureAlloc
     module procedure occaDeviceTextureAlloc_func
  end interface occaDeviceTextureAlloc

  ! interface occaDeviceManagedTextureAlloc
  !    module procedure occaDeviceManagedTextureAlloc_func
  ! end interface occaDeviceManagedTextureAlloc

  interface occaDeviceMappedAlloc
     module procedure occaDeviceMappedAlloc_null
     module procedure occaDeviceMappedAlloc_int4
     module procedure occaDeviceMappedAlloc_int8
     module procedure occaDeviceMappedAlloc_real4
     module procedure occaDeviceMappedAlloc_real8
     module procedure occaDeviceMappedAlloc_char
  end interface occaDeviceMappedAlloc

  ! interface occaDeviceManagedMappedAlloc
  !    module procedure occaDeviceManagedMappedAlloc_null
  !    module procedure occaDeviceManagedMappedAlloc_int4
  !    module procedure occaDeviceManagedMappedAlloc_int8
  !    module procedure occaDeviceManagedMappedAlloc_real4
  !    module procedure occaDeviceManagedMappedAlloc_real8
  !    module procedure occaDeviceManagedMappedAlloc_char
  ! end interface occaDeviceManagedMappedAlloc

  interface occaDeviceFlush
     subroutine occaDeviceFlush_fc(device, compilerFlags)
       use occaFTypes_m

       implicit none
       type(occaDevice), intent(inout) :: device
       character(len=*), intent(in)    :: compilerFlags
     end subroutine occaDeviceFlush_fc
  end interface occaDeviceFlush

  interface occaDeviceFinish
     subroutine occaDeviceFinish_fc(device)
       use occaFTypes_m

       implicit none
       type(occaDevice), intent(inout) :: device
     end subroutine occaDeviceFinish_fc
  end interface occaDeviceFinish

  interface occaDeviceCreateStream
     module procedure occaDeviceCreateStream_func
  end interface occaDeviceCreateStream

  interface occaDeviceGetStream
     module procedure occaDeviceGetStream_func
  end interface occaDeviceGetStream

  interface occaDeviceSetStream
     subroutine occaDeviceSetStream_fc(device, stream)
       use occaFTypes_m

       implicit none
       type(occaDevice), intent(inout) :: device
       type(occaStream), intent(inout) :: stream
     end subroutine occaDeviceSetStream_fc
  end interface occaDeviceSetStream

  interface occaDeviceTimeBetweenTags
     module procedure occaDeviceTimeBetweenTags_func
  end interface occaDeviceTimeBetweenTags

  interface occaDeviceTagStream
     module procedure occaDeviceTagStream_func
  end interface occaDeviceTagStream

  interface occaStreamFree
     subroutine occaStreamFree_fc(stream)
       use occaFTypes_m

       implicit none
       type(occaStream), intent(inout) :: stream
     end subroutine occaStreamFree_fc
  end interface occaStreamFree

  interface occaDeviceFree
     subroutine occaDeviceFree_fc(device)
       use occaFTypes_m

       implicit none
       type(occaDevice), intent(inout) :: device
     end subroutine occaDeviceFree_fc
  end interface occaDeviceFree

  ! ---[ Kernel ]-----------------------

  interface occaKernelPreferredDimSize
     module procedure occaKernelPreferredDimSize_func
  end interface occaKernelPreferredDimSize

  interface occaKernelSetAllWorkingDims
     subroutine occaKernelSetAllWorkingDims_fc(kernel, dims, itemsX, itemsY, itemsZ, groupsX, groupsY, groupsZ)
       use occaFTypes_m

       implicit none
       type(occaKernel), intent(inout) :: kernel
       integer(4),       intent(in)    :: dims
       integer(8),       intent(in)    :: itemsX, itemsY, itemsZ
       integer(8),       intent(in)    :: groupsX, groupsY, groupsZ
     end subroutine occaKernelSetAllWorkingDims_fc
  end interface occaKernelSetAllWorkingDims

  interface occaCreateArgumentList
     module procedure occaCreateArgumentList_func
  end interface occaCreateArgumentList

  interface occaArgumentListClear
     subroutine occaArgumentListClear_fc(list)
       use occaFTypes_m
       implicit none
       type(occaArgumentList), intent(inout) :: list
     end subroutine occaArgumentListClear_fc
  end interface occaArgumentListClear

  interface occaArgumentListFree
     subroutine occaArgumentListFree_fc(list)
       use occaFTypes_m
       implicit none
       type(occaArgumentList), intent(inout) :: list
     end subroutine occaArgumentListFree_fc
  end interface occaArgumentListFree

  interface occaArgumentListAddArg
     subroutine occaArgumentListAddArgMem_fc(list, idx, arg)
       use occaFTypes_m
       implicit none
       type(occaArgumentList), intent(inout) :: list
       integer(4),             intent(in)    :: idx
       type(occaMemory),       intent(in)    :: arg
     end subroutine occaArgumentListAddArgMem_fc

     ! subroutine occaArgumentListAddArgType_fc(list, idx, arg)
     !   use occaFTypes_m
     !   implicit none
     !   type(occaArgumentList), intent(inout) :: list
     !   integer(4),             intent(in)    :: idx
     !   type(occaType),         intent(in)    :: arg
     ! end subroutine occaArgumentListAddArgType_fc

     subroutine occaArgumentListAddArgInt4_fc(list, idx, arg)
       use occaFTypes_m
       implicit none
       type(occaArgumentList), intent(inout) :: list
       integer(4),             intent(in)    :: idx
       integer(4),             intent(in)    :: arg
     end subroutine occaArgumentListAddArgInt4_fc

     subroutine occaArgumentListAddArgReal4_fc(list, idx, arg)
       use occaFTypes_m
       implicit none
       type(occaArgumentList), intent(inout) :: list
       integer(4),             intent(in)    :: idx
       real(4),                intent(in)    :: arg
     end subroutine occaArgumentListAddArgReal4_fc

     subroutine occaArgumentListAddArgReal8_fc(list, idx, arg)
       use occaFTypes_m
       implicit none
       type(occaArgumentList), intent(inout) :: list
       integer(4),             intent(in)    :: idx
       real(8),                intent(in)    :: arg
     end subroutine occaArgumentListAddArgReal8_fc

     subroutine occaArgumentListAddArgChar_fc(list, idx, arg)
       use occaFTypes_m
       implicit none
       type(occaArgumentList), intent(inout) :: list
       integer(4),             intent(in)    :: idx
       character,              intent(in)    :: arg
     end subroutine occaArgumentListAddArgChar_fc
  end interface occaArgumentListAddArg

  interface occaKernelRun
     subroutine occaKernelRun01_fc(kernel, arg01)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01
     end subroutine occaKernelRun01_fc

     subroutine occaKernelRun02_fc(kernel, arg01, arg02)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02
     end subroutine occaKernelRun02_fc

     subroutine occaKernelRun03_fc(kernel, arg01, arg02, arg03)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03
     end subroutine occaKernelRun03_fc

     subroutine occaKernelRun04_fc(kernel, arg01, arg02, arg03, arg04)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04
     end subroutine occaKernelRun04_fc

     subroutine occaKernelRun05_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05
     end subroutine occaKernelRun05_fc

     subroutine occaKernelRun06_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06
     end subroutine occaKernelRun06_fc

     subroutine occaKernelRun07_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07
     end subroutine occaKernelRun07_fc

     subroutine occaKernelRun08_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08
     end subroutine occaKernelRun08_fc

     subroutine occaKernelRun09_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09
     end subroutine occaKernelRun09_fc

     subroutine occaKernelRun10_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10
     end subroutine occaKernelRun10_fc

     subroutine occaKernelRun11_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11
     end subroutine occaKernelRun11_fc

     subroutine occaKernelRun12_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12
     end subroutine occaKernelRun12_fc

     subroutine occaKernelRun13_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13
     end subroutine occaKernelRun13_fc

     subroutine occaKernelRun14_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14
     end subroutine occaKernelRun14_fc

     subroutine occaKernelRun15_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14, arg15)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14, arg15
     end subroutine occaKernelRun15_fc

     subroutine occaKernelRun16_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14, arg15, arg16)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14, arg15, arg16
     end subroutine occaKernelRun16_fc

     subroutine occaKernelRun17_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14, arg15, arg16, &
          arg17)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14, arg15, arg16, &
            arg17
     end subroutine occaKernelRun17_fc

     subroutine occaKernelRun18_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14, arg15, arg16, &
          arg17, arg18)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14, arg15, arg16, &
            arg17, arg18
     end subroutine occaKernelRun18_fc

     subroutine occaKernelRun19_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14, arg15, arg16, &
          arg17, arg18, arg19)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14, arg15, arg16, &
            arg17, arg18, arg19
     end subroutine occaKernelRun19_fc

     subroutine occaKernelRun20_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14, arg15, arg16, &
          arg17, arg18, arg19, arg20)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14, arg15, arg16, &
            arg17, arg18, arg19, arg20
     end subroutine occaKernelRun20_fc

     subroutine occaKernelRun21_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14, arg15, arg16, &
          arg17, arg18, arg19, arg20, &
          arg21)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14, arg15, arg16, &
            arg17, arg18, arg19, arg20, &
            arg21
     end subroutine occaKernelRun21_fc

     subroutine occaKernelRun22_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14, arg15, arg16, &
          arg17, arg18, arg19, arg20, &
          arg21, arg22)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14, arg15, arg16, &
            arg17, arg18, arg19, arg20, &
            arg21, arg22
     end subroutine occaKernelRun22_fc

     subroutine occaKernelRun23_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14, arg15, arg16, &
          arg17, arg18, arg19, arg20, &
          arg21, arg22, arg23)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14, arg15, arg16, &
            arg17, arg18, arg19, arg20, &
            arg21, arg22, arg23
     end subroutine occaKernelRun23_fc

     subroutine occaKernelRun24_fc(kernel, arg01, arg02, arg03, arg04, &
          arg05, arg06, arg07, arg08, &
          arg09, arg10, arg11, arg12, &
          arg13, arg14, arg15, arg16, &
          arg17, arg18, arg19, arg20, &
          arg21, arg22, arg23, arg24)
       use occaFTypes_m
       implicit none
       type(occaKernel), intent(inout) :: kernel
       type(occaMemory), intent(in)    :: arg01, arg02, arg03, arg04, &
            arg05, arg06, arg07, arg08, &
            arg09, arg10, arg11, arg12, &
            arg13, arg14, arg15, arg16, &
            arg17, arg18, arg19, arg20, &
            arg21, arg22, arg23, arg24
     end subroutine occaKernelRun24_fc
  end interface occaKernelRun

  interface occaKernelRun_
     subroutine occaKernelRun__fc(kernel, list)
       use occaFTypes_m
       implicit none
       type(occaKernel),       intent(inout) :: kernel
       type(occaArgumentList), intent(in)    :: list
     end subroutine occaKernelRun__fc
  end interface occaKernelRun_

  interface occaKernelFree
     subroutine occaKernelFree_fc(kernel)
       use occaFTypes_m
       implicit none
       type(occaKernel),       intent(inout) :: kernel
     end subroutine occaKernelFree_fc
  end interface occaKernelFree

  interface occaCreateDeviceInfo
     module procedure occaCreateDeviceInfo_func
  end interface occaCreateDeviceInfo

  interface occaDeviceInfoAppend
     subroutine occaDeviceInfoAppend_fc(info, key, value_)
       use occaFTypes_m
       implicit none
       type(occaDeviceInfo), intent(inout) :: info
       character(len=*),     intent(in)    :: key
       character(len=*),     intent(in)    :: value_
     end subroutine occaDeviceInfoAppend_fc
  end interface occaDeviceInfoAppend

  interface occaDeviceInfoFree
     subroutine occaDeviceInfoFree_fc(info)
       use occaFTypes_m
       implicit none
       type(occaDeviceInfo), intent(inout) :: info
     end subroutine occaDeviceInfoFree_fc
  end interface occaDeviceInfoFree

  interface occaCreateKernelInfo
     module procedure occaCreateKernelInfo_func
  end interface occaCreateKernelInfo

  interface occaKernelInfoAddDefine
     subroutine occaKernelInfoAddDefineInt4_fc(info, macro, val)
       use occaFTypes_m
       implicit none
       type(occaKernelInfo), intent(inout) :: info
       character(len=*),     intent(in)    :: macro
       integer(4),           intent(in)    :: val
     end subroutine occaKernelInfoAddDefineInt4_fc

     subroutine occaKernelInfoAddDefineReal4_fc(info, macro, val)
       use occaFTypes_m
       implicit none
       type(occaKernelInfo), intent(inout) :: info
       character(len=*),     intent(in)    :: macro
       real(4),           intent(in)    :: val
     end subroutine occaKernelInfoAddDefineReal4_fc

     subroutine occaKernelInfoAddDefineReal8_fc(info, macro, val)
       use occaFTypes_m
       implicit none
       type(occaKernelInfo), intent(inout) :: info
       character(len=*),     intent(in)    :: macro
       real(8),              intent(in)    :: val
     end subroutine occaKernelInfoAddDefineReal8_fc

     subroutine occaKernelInfoAddDefineString_fc(info, macro, val)
       use occaFTypes_m
       implicit none
       type(occaKernelInfo), intent(inout) :: info
       character(len=*),     intent(in)    :: macro
       character(len=*),     intent(in)    :: val
     end subroutine occaKernelInfoAddDefineString_fc
  end interface occaKernelInfoAddDefine

  interface occaKernelInfoAddInclude
     subroutine occaKernelInfoAddInclude_fc(info, filename)
       use occaFTypes_m
       implicit none
       type(occaKernelInfo), intent(inout) :: info
       character(len=*),     intent(in)    :: filename
     end subroutine occaKernelInfoAddInclude_fc
  end interface occaKernelInfoAddInclude

  interface occaKernelInfoFree
     subroutine occaKernelInfoFree_fc(info)
       use occaFTypes_m
       implicit none
       type(occaKernelInfo), intent(inout) :: info
     end subroutine occaKernelInfoFree_fc
  end interface occaKernelInfoFree

  interface occaDeviceWrapMemory
     module procedure occaDeviceWrapMemory_func
  end interface occaDeviceWrapMemory

  interface occaDeviceWrapStream
     module procedure occaDeviceWrapStream_func
  end interface occaDeviceWrapStream

  ! ---[ Memory ]-----------------------

  interface occaMemoryGetMappedPointer
     function occaMemoryGetMappedPointer_fc(mem) result(ptr)
       use occaFTypes_m
       implicit none
       type(occaMemory), intent(in)   :: mem
       real(4), dimension(:), pointer :: ptr
     end function occaMemoryGetMappedPointer_fc
  end interface occaMemoryGetMappedPointer

  interface occaCopyMemToMem
     subroutine occaCopyMemToMem_fc(dest, src, bytes, destOffset, srcOffset)
       use occaFTypes_m
       implicit none
       type(occaMemory), intent(out) :: dest
       type(occaMemory), intent(in)  :: src
       integer(8),       intent(in)  :: bytes
       integer(8),       intent(in)  :: destOffset
       integer(8),       intent(in)  :: srcOffset
     end subroutine occaCopyMemToMem_fc
  end interface occaCopyMemToMem

  interface occaCopyPtrToMem
     module procedure occaCopyPtrToMem_int4
     module procedure occaCopyPtrToMem_int8
     module procedure occaCopyPtrToMem_real4
     module procedure occaCopyPtrToMem_real8
     module procedure occaCopyPtrToMem_char
     module procedure occaCopyPtrToMemAuto_int4
     module procedure occaCopyPtrToMemAuto_int8
     module procedure occaCopyPtrToMemAuto_real4
     module procedure occaCopyPtrToMemAuto_real8
     module procedure occaCopyPtrToMemAuto_char
  end interface occaCopyPtrToMem

  interface occaCopyMemToPtr
     module procedure occaCopyMemToPtr_int4
     module procedure occaCopyMemToPtr_int8
     module procedure occaCopyMemToPtr_real4
     module procedure occaCopyMemToPtr_real8
     module procedure occaCopyMemToPtr_char
     module procedure occaCopyMemToPtrAuto_int4
     module procedure occaCopyMemToPtrAuto_int8
     module procedure occaCopyMemToPtrAuto_real4
     module procedure occaCopyMemToPtrAuto_real8
     module procedure occaCopyMemToPtrAuto_char
  end interface occaCopyMemToPtr

  interface occaAsyncCopyMemToMem
     subroutine occaAsyncCopyMemToMem_fc(dest, src, bytes, destOffset, srcOffset)
       use occaFTypes_m
       implicit none
       type(occaMemory), intent(out) :: dest
       type(occaMemory), intent(in)  :: src
       integer(8),       intent(in)  :: bytes
       integer(8),       intent(in)  :: destOffset
       integer(8),       intent(in)  :: srcOffset
     end subroutine occaAsyncCopyMemToMem_fc
  end interface occaAsyncCopyMemToMem

  interface occaAsyncCopyPtrToMem
     module procedure occaAsyncCopyPtrToMem_int4
     module procedure occaAsyncCopyPtrToMem_int8
     module procedure occaAsyncCopyPtrToMem_real4
     module procedure occaAsyncCopyPtrToMem_real8
     module procedure occaAsyncCopyPtrToMem_char
     module procedure occaAsyncCopyPtrToMemAuto_int4
     module procedure occaAsyncCopyPtrToMemAuto_int8
     module procedure occaAsyncCopyPtrToMemAuto_real4
     module procedure occaAsyncCopyPtrToMemAuto_real8
     module procedure occaAsyncCopyPtrToMemAuto_char
  end interface occaAsyncCopyPtrToMem

  interface occaAsyncCopyMemToPtr
     module procedure occaAsyncCopyMemToPtr_int4
     module procedure occaAsyncCopyMemToPtr_int8
     module procedure occaAsyncCopyMemToPtr_real4
     module procedure occaAsyncCopyMemToPtr_real8
     module procedure occaAsyncCopyMemToPtr_char
     module procedure occaAsyncCopyMemToPtrAuto_int4
     module procedure occaAsyncCopyMemToPtrAuto_int8
     module procedure occaAsyncCopyMemToPtrAuto_real4
     module procedure occaAsyncCopyMemToPtrAuto_real8
     module procedure occaAsyncCopyMemToPtrAuto_char
  end interface occaAsyncCopyMemToPtr

  interface occaMemoryFree
     subroutine occaMemoryFree_fc(memory)
       use occaFTypes_m
       implicit none
       type(occaMemory), intent(inout) :: memory
     end subroutine occaMemoryFree_fc
  end interface occaMemoryFree

  !---[ Helper Functions ]--------------

  interface occaSysCall
     module procedure occaSysCall_
     ! module procedure occaSysCall_output
  end interface occaSysCall

contains !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! ---[ TypesCasting ]-----------------
  type(occaMemory) function occaTypeMem_int4_c(v) result(t)
    integer(4), intent(in) :: v

    interface
       subroutine occaInt32_fc(t, v)
         use occaFTypes_m
         implicit none
         type(occaMemory), intent(out) :: t
         integer(4),       intent(in)  :: v
       end subroutine occaInt32_fc
    end interface

    call occaInt32_fc(t, v)
  end function occaTypeMem_int4_c

  ! type(occaMemory) function occaTypeMem_int8_c(v) result(t)
  !   integer(8), intent(in) :: v

  !   interface
  !     subroutine occaInt64_fc(t, v)
  !       use occaFTypes_m
  !       implicit none
  !       type(occaMemory), intent(out) :: t
  !       integer(8),     intent(in)  :: v
  !     end subroutine occaInt64_fc
  !   end interface

  !   call occaInt64_fc(t, v)
  ! end function occaTypeMem_int8_c

  type(occaMemory) function occaTypeMem_real4_c(v) result(t)
    real(4), intent(in) :: v

    interface
       subroutine occaFloat_fc(t, v)
         use occaFTypes_m
         implicit none
         type(occaMemory), intent(out) :: t
         real(4),          intent(in)  :: v
       end subroutine occaFloat_fc
    end interface

    call occaFloat_fc(t, v)
  end function occaTypeMem_real4_c

  type(occaMemory) function occaTypeMem_real8_c(v) result(t)
    real(8), intent(in) :: v

    interface
       subroutine occaDouble_fc(t, v)
         use occaFTypes_m
         implicit none
         type(occaMemory), intent(out) :: t
         real(8),          intent(in)  :: v
       end subroutine occaDouble_fc
    end interface

    call occaDouble_fc(t, v)
  end function occaTypeMem_real8_c

  type(occaMemory) function occaTypeMem_str_c(v) result(t)
    character(len=*), intent(in) :: v

    interface
       subroutine occaString_fc(t, v)
         use occaFTypes_m
         implicit none
         type(occaMemory), intent(out) :: t
         character(len=*), intent(in)  :: v
       end subroutine occaString_fc
    end interface

    call occaString_fc(t, v)
  end function occaTypeMem_str_c

  ! ---[ Device ]-----------------------
  !  interface occaDeviceMode
  !     const char* occaDeviceMode_fc(occaDevice device)
  !  end interface occaDeviceMode


  type(occaDevice) function occaCreateDevice_func(infos) result(device)
    character(len=*), intent(in) :: infos

    interface
       subroutine occaCreateDevice_fc(device, infos)
         use occaFTypes_m
         implicit none
         type(occaDevice), intent(out) :: device
         character(len=*), intent(in)  :: infos
       end subroutine occaCreateDevice_fc
    end interface

    call occaCreateDevice_fc(device, infos)
  end function occaCreateDevice_func

  type(occaDevice) function occaCreateDeviceFromInfo_func(dInfo) result(device)
    type(occaDeviceInfo), intent(in) :: dInfo

    interface
       subroutine occaCreateDeviceFromInfo_fc(device, dInfo)
         use occaFTypes_m
         implicit none
         type(occaDevice), intent(out) :: device
         type(occaDeviceInfo), intent(in) :: dInfo
       end subroutine occaCreateDeviceFromInfo_fc
    end interface

    call occaCreateDeviceFromInfo_fc(device, dInfo)
  end function occaCreateDeviceFromInfo_func

  type(occaDevice) function occaCreateDeviceFromArgs_func(mode, arg1, arg2) result(device)
    character(len=*), intent(in)  :: mode
    integer(4),       intent(in)  :: arg1
    integer(4),       intent(in)  :: arg2

    interface
       subroutine occaCreateDeviceFromArgs_fc(device, mode, arg1, arg2)
         use occaFTypes_m
         implicit none
         type(occaDevice), intent(out) :: device
         character(len=*), intent(in)  :: mode
         integer(4),       intent(in)  :: arg1
         integer(4),       intent(in)  :: arg2
       end subroutine occaCreateDeviceFromArgs_fc
    end interface

    call occaCreateDeviceFromArgs_fc(device, mode, arg1, arg2)
  end function occaCreateDeviceFromArgs_func

  integer(8) function occaDeviceMemorySize(device) result (bytes)
    type(occaDevice), intent(inout) :: device

    interface
       subroutine occaDeviceMemorySize_fc(device, bytes)
         use occaFTypes_m

         implicit none
         type(occaDevice), intent(inout) :: device
         integer(8),       intent(out)   :: bytes
       end subroutine occaDeviceMemorySize_fc
    end interface

    call occaDeviceMemorySize_fc(device, bytes)
  end function occaDeviceMemorySize

  integer(8) function occaDeviceMemoryAllocated(device) result (bytes)
    type(occaDevice), intent(inout) :: device

    interface
       subroutine occaDeviceMemoryAllocated_fc(device, bytes)
         use occaFTypes_m

         implicit none
         type(occaDevice), intent(inout) :: device
         integer(8),       intent(out)   :: bytes
       end subroutine occaDeviceMemoryAllocated_fc
    end interface

    call occaDeviceMemoryAllocated_fc(device, bytes)
  end function occaDeviceMemoryAllocated

  integer(8) function occaDeviceBytesAllocated(device) result (bytes)
    type(occaDevice), intent(inout) :: device

    interface
       subroutine occaDeviceBytesAllocated_fc(device, bytes)
         use occaFTypes_m

         implicit none
         type(occaDevice), intent(inout) :: device
         integer(8),       intent(out)   :: bytes
       end subroutine occaDeviceBytesAllocated_fc
    end interface

    call occaDeviceBytesAllocated_fc(device, bytes)
  end function occaDeviceBytesAllocated

  type(occaKernel) function occaDeviceBuildKernel_func(device, str, functionName, info) result(kernel)
    type(occaDevice),     intent(in)  :: device
    character(len=*),     intent(in)  :: str
    character(len=*),     intent(in)  :: functionName
    type(occaKernelInfo), intent(in)  :: info

    interface
       subroutine occaDeviceBuildKernel_fc(kernel, device, str, functionName, info)
         use occaFTypes_m
         implicit none
         type(occaKernel),     intent(out) :: kernel
         type(occaDevice),     intent(in)  :: device
         character(len=*),     intent(in)  :: str
         character(len=*),     intent(in)  :: functionName
         type(occaKernelInfo), intent(in)  :: info
       end subroutine occaDeviceBuildKernel_fc
    end interface

    call occaDeviceBuildKernel_fc(kernel, device, str, functionName, info)
  end function occaDeviceBuildKernel_func

  type(occaKernel) function occaDeviceBuildKernelNoKernelInfo_func(device, str, functionName) result(kernel)
    type(occaDevice),     intent(in)  :: device
    character(len=*),     intent(in)  :: str
    character(len=*),     intent(in)  :: functionName

    interface
       subroutine occaDeviceBuildKernelNoKernelInfo_fc(kernel, device, str, functionName)
         use occaFTypes_m
         implicit none
         type(occaKernel),     intent(out) :: kernel
         type(occaDevice),     intent(in)  :: device
         character(len=*),     intent(in)  :: str
         character(len=*),     intent(in)  :: functionName
       end subroutine occaDeviceBuildKernelNoKernelInfo_fc
    end interface

    call occaDeviceBuildKernelNoKernelInfo_fc(kernel, device, str, functionName)
  end function occaDeviceBuildKernelNoKernelInfo_func

  type(occaKernel) function occaDeviceBuildKernelFromSource_func(device, filename, functionName, info) result(kernel)
    type(occaDevice),     intent(in)  :: device
    character(len=*),     intent(in)  :: filename
    character(len=*),     intent(in)  :: functionName
    type(occaKernelInfo), intent(in)  :: info

    interface
       subroutine occaDeviceBuildKernelFromSource_fc(kernel, device, filename, functionName, info)
         use occaFTypes_m
         implicit none
         type(occaKernel),     intent(out) :: kernel
         type(occaDevice),     intent(in)  :: device
         character(len=*),     intent(in)  :: filename
         character(len=*),     intent(in)  :: functionName
         type(occaKernelInfo), intent(in)  :: info
       end subroutine occaDeviceBuildKernelFromSource_fc
    end interface

    call occaDeviceBuildKernelFromSource_fc(kernel, device, filename, functionName, info)
  end function occaDeviceBuildKernelFromSource_func

  type(occaKernel) function occaDeviceBuildKernelFromSourceNoKernelInfo_func(device, filename, functionName) result(kernel)
    type(occaDevice),     intent(in)  :: device
    character(len=*),     intent(in)  :: filename
    character(len=*),     intent(in)  :: functionName

    interface
       subroutine occaDeviceBuildKernelFromSourceNoKernelInfo_fc(kernel, device, filename, functionName)
         use occaFTypes_m
         implicit none
         type(occaKernel),     intent(out) :: kernel
         type(occaDevice),     intent(in)  :: device
         character(len=*),     intent(in)  :: filename
         character(len=*),     intent(in)  :: functionName
       end subroutine occaDeviceBuildKernelFromSourceNoKernelInfo_fc
    end interface

    call occaDeviceBuildKernelFromSourceNoKernelInfo_fc(kernel, device, filename, functionName)
  end function occaDeviceBuildKernelFromSourceNoKernelInfo_func

  type(occaKernel) function occaDeviceBuildKernelFromString_func(device, str, functionName, info, language) result(kernel)
    type(occaDevice),     intent(in)  :: device
    character(len=*),     intent(in)  :: str
    character(len=*),     intent(in)  :: functionName
    type(occaKernelInfo), intent(in)  :: info
    integer             , intent(in)  :: language

    interface
       subroutine occaDeviceBuildKernelFromString_fc(kernel, device, str, functionName, info, language)
         use occaFTypes_m
         implicit none
         type(occaKernel),     intent(out) :: kernel
         type(occaDevice),     intent(in)  :: device
         character(len=*),     intent(in)  :: str
         character(len=*),     intent(in)  :: functionName
         type(occaKernelInfo), intent(in)  :: info
         integer             , intent(in)  :: language
       end subroutine occaDeviceBuildKernelFromString_fc
    end interface

    call occaDeviceBuildKernelFromString_fc(kernel, device, str, functionName, info, language)
  end function occaDeviceBuildKernelFromString_func

  type(occaKernel) function occaDeviceBuildKernelFromStringNoKernelInfo_func(device, str, functionName, language) result(kernel)
    type(occaDevice),     intent(in)  :: device
    character(len=*),     intent(in)  :: str
    character(len=*),     intent(in)  :: functionName
    integer         ,     intent(in)  :: language

    interface
       subroutine occaDeviceBuildKernelFromStringNoKernelInfo_fc(kernel, device, str, functionName, language)
         use occaFTypes_m
         implicit none
         type(occaKernel),     intent(out) :: kernel
         type(occaDevice),     intent(in)  :: device
         character(len=*),     intent(in)  :: str
         character(len=*),     intent(in)  :: functionName
         integer         ,     intent(in)  :: language
       end subroutine occaDeviceBuildKernelFromStringNoKernelInfo_fc
    end interface

    call occaDeviceBuildKernelFromStringNoKernelInfo_fc(kernel, device, str, functionName, language)
  end function occaDeviceBuildKernelFromStringNoKernelInfo_func

  type(occaKernel) function occaDeviceBuildKernelFromStringNoArgs_func(device, str, functionName) result(kernel)
    type(occaDevice),     intent(in)  :: device
    character(len=*),     intent(in)  :: str
    character(len=*),     intent(in)  :: functionName

    interface
       subroutine occaDeviceBuildKernelFromStringNoArgs_fc(kernel, device, str, functionName)
         use occaFTypes_m
         implicit none
         type(occaKernel),     intent(out) :: kernel
         type(occaDevice),     intent(in)  :: device
         character(len=*),     intent(in)  :: str
         character(len=*),     intent(in)  :: functionName
       end subroutine occaDeviceBuildKernelFromStringNoArgs_fc
    end interface

    call occaDeviceBuildKernelFromStringNoArgs_fc(kernel, device, str, functionName)
  end function occaDeviceBuildKernelFromStringNoArgs_func

  type(occaKernel) function occaDeviceBuildKernelFromBinary_func(device, filename, functionName) result(kernel)
    type(occaDevice),     intent(in)  :: device
    character(len=*),     intent(in)  :: filename
    character(len=*),     intent(in)  :: functionName

    interface
       subroutine occaDeviceBuildKernelFromBinary_fc(kernel, device, filename, functionName)
         use occaFTypes_m
         implicit none
         type(occaKernel),     intent(out) :: kernel
         type(occaDevice),     intent(in)  :: device
         character(len=*),     intent(in)  :: filename
         character(len=*),     intent(in)  :: functionName
       end subroutine occaDeviceBuildKernelFromBinary_fc
    end interface

    call occaDeviceBuildKernelFromBinary_fc(kernel, device, filename, functionName)
  end function occaDeviceBuildKernelFromBinary_func

  !---[ Malloc ]------------------------

  type(occaMemory) function occaDeviceMalloc_null(device, sz) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz

    call occaDeviceMallocNULL_fc(mem, device, sz)
  end function occaDeviceMalloc_null

  type(occaMemory) function occaDeviceMalloc_int4(device, sz, buf) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz
    integer(4),        intent(in)    :: buf

    call occaDeviceMalloc_fc(mem, device, sz, buf)
  end function occaDeviceMalloc_int4
  type(occaMemory) function occaDeviceMalloc_int8(device, sz, buf) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz
    integer(8),        intent(in)    :: buf

    call occaDeviceMalloc_fc(mem, device, sz, buf)
  end function occaDeviceMalloc_int8
  type(occaMemory) function occaDeviceMalloc_real4(device, sz, buf) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz
    real(4),           intent(in)    :: buf

    call occaDeviceMalloc_fc(mem, device, sz, buf)
  end function occaDeviceMalloc_real4
  type(occaMemory) function occaDeviceMalloc_real8(device, sz, buf) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz
    real(8),           intent(in)    :: buf

    call occaDeviceMalloc_fc(mem, device, sz, buf)
  end function occaDeviceMalloc_real8
  type(occaMemory) function occaDeviceMalloc_char(device, sz, buf) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz
    character,         intent(in)    :: buf

    call occaDeviceMalloc_fc(mem, device, sz, buf)
  end function occaDeviceMalloc_char

  !---[ Managed ]--------------

  ! type(occaStream) function occaDeviceWrapStream_func(device, handle) result(stream)
  !   type(occaDevice), intent(in)  :: device
  !   character       , intent(in)  :: handle

  !   interface
  !      subroutine occaDeviceWrapStream_fc(stream, device, handle)
  !        use occaFTypes_m
  !        implicit none
  !        type(occaStream), intent(out) :: stream
  !        type(occaDevice), intent(in)  :: device
  !        character       , intent(in)  :: handle
  !      end subroutine occaDeviceWrapStream_fc
  !   end interface

  !   call occaDeviceWrapStream_fc(stream, device, handle)
  ! end function occaDeviceWrapStream_func

  ! type(occaMemory) function occaDeviceManagedAlloc_null(device, sz) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz

  !   call occaDeviceManagedAllocNULL_fc(mem, device, sz)
  ! end function occaDeviceManagedAlloc_null

  ! type(occaMemory) function occaDeviceManagedAlloc_int4(device, sz, buf) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz
  !   integer(4),        intent(in)    :: buf

  !   call occaDeviceManagedAlloc_fc(mem, device, sz, buf)
  ! end function occaDeviceManagedAlloc_int4
  ! type(occaMemory) function occaDeviceManagedAlloc_int8(device, sz, buf) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz
  !   integer(8),        intent(in)    :: buf

  !   call occaDeviceManagedAlloc_fc(mem, device, sz, buf)
  ! end function occaDeviceManagedAlloc_int8
  ! type(occaMemory) function occaDeviceManagedAlloc_real4(device, sz, buf) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz
  !   real(4),           intent(in)    :: buf

  !   call occaDeviceManagedAlloc_fc(mem, device, sz, buf)
  ! end function occaDeviceManagedAlloc_real4
  ! type(occaMemory) function occaDeviceManagedAlloc_real8(device, sz, buf) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz
  !   real(8),           intent(in)    :: buf

  !   call occaDeviceManagedAlloc_fc(mem, device, sz, buf)
  ! end function occaDeviceManagedAlloc_real8
  ! type(occaMemory) function occaDeviceManagedAlloc_char(device, sz, buf) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz
  !   character,         intent(in)    :: buf

  !   call occaDeviceManagedAlloc_fc(mem, device, sz, buf)
  ! end function occaDeviceManagedAlloc_char

  !---[ Texture Alloc ]-----------------

  type(occaMemory) function occaDeviceTextureAlloc_func(dim, dimX, dimY, dimZ, source, type, permissions) result(mem)
    integer(4), intent(in) :: dim
    integer(8), intent(in) :: dimX, dimY, dimZ
    integer(8), intent(in) :: source
    integer(8), intent(in) :: type
    integer(4), intent(in) :: permissions

    interface
       subroutine occaDeviceTextureAlloc_fc(mem, dim, dimX, dimY, dimZ, source, type, permissions)
         use occaFTypes_m
         implicit none
         type(occaMemory), intent(out) :: mem
         integer(4)      , intent(in)  :: dim
         integer(8)      , intent(in)  :: dimX, dimY, dimZ
         integer(8)      , intent(in)  :: source
         integer(8)      , intent(in)  :: type
         integer(4)      , intent(in)  :: permissions
       end subroutine occaDeviceTextureAlloc_fc
    end interface

    call occaDeviceTextureAlloc_fc(mem, dim, dimX, dimY, dimZ, source, type, permissions)
  end function occaDeviceTextureAlloc_func

  !---[ Managed ]--------------

  ! type(occaMemory) function occaDeviceManagedTextureAlloc_func(dim, dimX, dimY, dimZ, source, type, permissions) result(mem)
  !   integer(4), intent(in) :: dim
  !   integer(8), intent(in) :: dimX, dimY, dimZ
  !   integer(8), intent(in) :: source
  !   integer(8), intent(in) :: type
  !   integer(4), intent(in) :: permissions

  !   interface
  !      subroutine occaDeviceManagedTextureAlloc_fc(mem, dim, dimX, dimY, dimZ, source, type, permissions)
  !        use occaFTypes_m
  !        implicit none
  !        type(occaMemory), intent(out) :: mem
  !        integer(4)      , intent(in)  :: dim
  !        integer(8)      , intent(in)  :: dimX, dimY, dimZ
  !        integer(8)      , intent(in)  :: source
  !        integer(8)      , intent(in)  :: type
  !        integer(4)      , intent(in)  :: permissions
  !      end subroutine occaDeviceManagedTextureAlloc_fc
  !   end interface

  !    call occaDeviceManagedTextureAlloc_fc(mem, dim, dimX, dimY, dimZ, source, type, permissions)
  !  end function occaDeviceManagedTextureAlloc_func

  !---[ Mapped Alloc ]------------------

  type(occaMemory) function occaDeviceMappedAlloc_null(device, sz) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz

    call occaDeviceMappedAllocNULL_fc(mem, device, sz)
  end function occaDeviceMappedAlloc_null

  type(occaMemory) function occaDeviceMappedAlloc_int4(device, sz, buf) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz
    integer(4),        intent(in)    :: buf

    call occaDeviceMappedAlloc_fc(mem, device, sz, buf)
  end function occaDeviceMappedAlloc_int4

  type(occaMemory) function occaDeviceMappedAlloc_int8(device, sz, buf) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz
    integer(8),        intent(in)    :: buf

    call occaDeviceMappedAlloc_fc(mem, device, sz, buf)
  end function occaDeviceMappedAlloc_int8

  type(occaMemory) function occaDeviceMappedAlloc_real4(device, sz, buf) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz
    real(4),           intent(in)    :: buf

    call occaDeviceMappedAlloc_fc(mem, device, sz, buf)
  end function occaDeviceMappedAlloc_real4

  type(occaMemory) function occaDeviceMappedAlloc_real8(device, sz, buf) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz
    real(8),           intent(in)    :: buf

    call occaDeviceMappedAlloc_fc(mem, device, sz, buf)
  end function occaDeviceMappedAlloc_real8

  type(occaMemory) function occaDeviceMappedAlloc_char(device, sz, buf) result(mem)
    type(occaDevice),  intent(inout) :: device
    integer(8),        intent(in)    :: sz
    character,         intent(in)    :: buf

    call occaDeviceMappedAlloc_fc(mem, device, sz, buf)
  end function occaDeviceMappedAlloc_char

  !---[ Managed ]--------------

  ! type(occaMemory) function occaDeviceManagedMappedAlloc_null(device, sz) result(ptr)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz

  !   call occaDeviceManagedMappedAllocNULL_fc(mem, device, sz)
  ! end function occaDeviceManagedMappedAlloc_null

  ! type(occaMemory) function occaDeviceManagedMappedAlloc_int4(device, sz, buf) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz
  !   integer(4),        intent(in)    :: buf

  !   call occaDeviceManagedMappedAlloc_fc(mem, device, sz, buf)
  ! end function occaDeviceManagedMappedAlloc_int4
  ! type(occaMemory) function occaDeviceManagedMappedAlloc_int8(device, sz, buf) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz
  !   integer(8),        intent(in)    :: buf

  !   call occaDeviceManagedMappedAlloc_fc(mem, device, sz, buf)
  ! end function occaDeviceManagedMappedAlloc_int8
  ! type(occaMemory) function occaDeviceManagedMappedAlloc_real4(device, sz, buf) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz
  !   real(4),           intent(in)    :: buf

  !   call occaDeviceManagedMappedAlloc_fc(mem, device, sz, buf)
  ! end function occaDeviceManagedMappedAlloc_real4
  ! type(occaMemory) function occaDeviceManagedMappedAlloc_real8(device, sz, buf) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz
  !   real(8),           intent(in)    :: buf

  !   call occaDeviceManagedMappedAlloc_fc(mem, device, sz, buf)
  ! end function occaDeviceManagedMappedAlloc_real8
  ! type(occaMemory) function occaDeviceManagedMappedAlloc_char(device, sz, buf) result(mem)
  !   type(occaDevice),  intent(inout) :: device
  !   integer(8),        intent(in)    :: sz
  !   character,         intent(in)    :: buf

  !   call occaDeviceManagedMappedAlloc_fc(mem, device, sz, buf)
  ! end function occaDeviceManagedMappedAlloc_char

  !=====================================

  type(occaStream) function occaDeviceCreateStream_func(device) result(stream)
    type(occaDevice),     intent(inout)  :: device

    interface
       subroutine occaDeviceCreateStream_fc(stream, device)
         use occaFTypes_m
         implicit none
         type(occaStream),  intent(out)   :: stream
         type(occaDevice),  intent(inout) :: device
       end subroutine occaDeviceCreateStream_fc
    end interface

    call occaDeviceCreateStream_fc(stream, device)
  end function occaDeviceCreateStream_func

  type(occaStream) function occaDeviceGetStream_func(device) result(stream)
    type(occaDevice),     intent(inout)  :: device

    interface
       subroutine occaDeviceGetStream_fc(stream, device)
         use occaFTypes_m
         implicit none
         type(occaStream),  intent(out)   :: stream
         type(occaDevice),  intent(inout) :: device
       end subroutine occaDeviceGetStream_fc
    end interface

    call occaDeviceGetStream_fc(stream, device)
  end function occaDeviceGetStream_func


  type(occaStreamTag) function occaDeviceTagStream_func(device) result(tag)
    type(occaDevice), intent(inout)  :: device

    interface
       subroutine occaDeviceTagStream_fc(tag, device)
         use occaFTypes_m
         implicit none
         type(occaStreamTag), intent(out)   :: tag
         type(occaDevice),    intent(inout) :: device
       end subroutine occaDeviceTagStream_fc
    end interface

    call occaDeviceTagStream_fc(tag, device)
  end function occaDeviceTagStream_func

  real(8) function occaDeviceTimeBetweenTags_func(device, startTag, endTag) result(time)
    type(occaDevice),    intent(inout) :: device
    type(occaStreamTag), intent(in)    :: startTag
    type(occaStreamTag), intent(in)    :: endTag

    interface
       subroutine occaDeviceTimeBetweenTags_fc(time, device, startTag, endTag)
         use occaFTypes_m
         implicit none
         real(8),             intent(out)   :: time
         type(occaDevice),    intent(inout) :: device
         type(occaStreamTag), intent(in)    :: startTag
         type(occaStreamTag), intent(in)    :: endTag
       end subroutine occaDeviceTimeBetweenTags_fc
    end interface

    call occaDeviceTimeBetweenTags_fc(time, device, startTag, endTag)
  end function occaDeviceTimeBetweenTags_func


  ! ---[ Kernel ]-----------------------
  !  interface occaKernelMode
  !     const char* occaKernelMode_fc(occaKernel kernel)
  !  end interface occaKernelMode

  integer(4) function occaKernelPreferredDimSize_func(kernel) result(sz)
    type(occaKernel),     intent(in) :: kernel

    interface
       subroutine occaKernelPreferredDimSize_fc(sz, kernel)
         use occaFTypes_m
         implicit none
         integer(4),       intent(out) :: sz
         type(occaKernel), intent(in)  :: kernel
       end subroutine occaKernelPreferredDimSize_fc
    end interface

    call occaKernelPreferredDimSize_fc(sz, kernel)
  end function occaKernelPreferredDimSize_func


  type(occaArgumentList) function occaCreateArgumentList_func() result(args)

    interface
       subroutine occaCreateArgumentList_fc(args)
         use occaFTypes_m
         implicit none
         type(occaArgumentList), intent(out) :: args
       end subroutine occaCreateArgumentList_fc
    end interface

    call occaCreateArgumentList_fc(args)
  end function occaCreateArgumentList_func

  type(occaDeviceInfo) function occaCreateDeviceInfo_func() result(info)

    interface
       subroutine occaCreateDeviceInfo_fc(info)
         use occaFTypes_m
         implicit none
         type(occaDeviceInfo), intent(out) :: info
       end subroutine occaCreateDeviceInfo_fc
    end interface

    call occaCreateDeviceInfo_fc(info)
  end function occaCreateDeviceInfo_func

  type(occaKernelInfo) function occaCreateKernelInfo_func() result(info)

    interface
       subroutine occaCreateKernelInfo_fc(info)
         use occaFTypes_m
         implicit none
         type(occaKernelInfo), intent(out) :: info
       end subroutine occaCreateKernelInfo_fc
    end interface

    call occaCreateKernelInfo_fc(info)
  end function occaCreateKernelInfo_func

  type(occaMemory) function occaDeviceWrapMemory_func(device, handle, bytes) result(mem)
    type(occaDevice), intent(in)  :: device
    character       , intent(in)  :: handle
    integer(8)      , intent(in)  :: bytes

    interface
       subroutine occaDeviceWrapMemory_fc(mem, device, handle, bytes)
         use occaFTypes_m
         implicit none
         type(occaMemory), intent(out) :: mem
         type(occaDevice), intent(in)  :: device
         character       , intent(in)  :: handle
         integer(8)      , intent(in)  :: bytes
       end subroutine occaDeviceWrapMemory_fc
    end interface

    call occaDeviceWrapMemory_fc(mem, device, handle, bytes)
  end function occaDeviceWrapMemory_func

  type(occaStream) function occaDeviceWrapStream_func(device, handle) result(stream)
    type(occaDevice), intent(in)  :: device
    character       , intent(in)  :: handle

    interface
       subroutine occaDeviceWrapStream_fc(stream, device, handle)
         use occaFTypes_m
         implicit none
         type(occaStream), intent(out) :: stream
         type(occaDevice), intent(in)  :: device
         character       , intent(in)  :: handle
       end subroutine occaDeviceWrapStream_fc
    end interface

    call occaDeviceWrapStream_fc(stream, device, handle)
  end function occaDeviceWrapStream_func

  ! ---[ Memory ]-----------------------
  !  interface occaMemoryMode
  !     const char* occaMemoryMode_fc(occaMemory memory)
  !  end interface occaMemoryMode

  subroutine occaCopyPtrToMem_int4(dest, src, bytes, offset)
    type(occaMemory), intent(out) :: dest
    integer(4),       intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaCopyPtrToMem_fc(dest, src, bytes, offset)
  end subroutine occaCopyPtrToMem_int4

  subroutine occaCopyPtrToMem_int8(dest, src, bytes, offset)
    type(occaMemory), intent(out) :: dest
    integer(8),       intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaCopyPtrToMem_fc(dest, src, bytes, offset)
  end subroutine occaCopyPtrToMem_int8

  subroutine occaCopyPtrToMem_real4(dest, src, bytes, offset)
    type(occaMemory), intent(out) :: dest
    real(4),          intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaCopyPtrToMem_fc(dest, src, bytes, offset)
  end subroutine occaCopyPtrToMem_real4

  subroutine occaCopyPtrToMem_real8(dest, src, bytes, offset)
    type(occaMemory), intent(out) :: dest
    real(8),          intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaCopyPtrToMem_fc(dest, src, bytes, offset)
  end subroutine occaCopyPtrToMem_real8

  subroutine occaCopyPtrToMem_char(dest, src, bytes, offset)
    type(occaMemory), intent(out) :: dest
    character,        intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaCopyPtrToMem_fc(dest, src, bytes, offset)
  end subroutine occaCopyPtrToMem_char

  subroutine occaCopyMemToPtr_int4(dest, src, bytes, offset)
    integer(4),       intent(out) :: dest
    type(occaMemory), intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaCopyMemToPtr_fc(dest, src, bytes, offset)
  end subroutine occaCopyMemToPtr_int4

  subroutine occaCopyMemToPtr_int8(dest, src, bytes, offset)
    integer(8),       intent(out) :: dest
    type(occaMemory), intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaCopyMemToPtr_fc(dest, src, bytes, offset)
  end subroutine occaCopyMemToPtr_int8

  subroutine occaCopyMemToPtr_real4(dest, src, bytes, offset)
    real(4),          intent(out) :: dest
    type(occaMemory), intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaCopyMemToPtr_fc(dest, src, bytes, offset)
  end subroutine occaCopyMemToPtr_real4

  subroutine occaCopyMemToPtr_real8(dest, src, bytes, offset)
    real(8),          intent(out) :: dest
    type(occaMemory), intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaCopyMemToPtr_fc(dest, src, bytes, offset)
  end subroutine occaCopyMemToPtr_real8

  subroutine occaCopyMemToPtr_char(dest, src, bytes, offset)
    character,        intent(out) :: dest
    type(occaMemory), intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaCopyMemToPtr_fc(dest, src, bytes, offset)
  end subroutine occaCopyMemToPtr_char

  subroutine occaAsyncCopyPtrToMem_int4(dest, src, bytes, offset)
    type(occaMemory), intent(out) :: dest
    integer(4),       intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaAsyncCopyPtrToMem_fc(dest, src, bytes, offset)
  end subroutine occaAsyncCopyPtrToMem_int4

  subroutine occaAsyncCopyPtrToMem_int8(dest, src, bytes, offset)
    type(occaMemory), intent(out) :: dest
    integer(8),       intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaAsyncCopyPtrToMem_fc(dest, src, bytes, offset)
  end subroutine occaAsyncCopyPtrToMem_int8

  subroutine occaAsyncCopyPtrToMem_real4(dest, src, bytes, offset)
    type(occaMemory), intent(out) :: dest
    real(4),          intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaAsyncCopyPtrToMem_fc(dest, src, bytes, offset)
  end subroutine occaAsyncCopyPtrToMem_real4

  subroutine occaAsyncCopyPtrToMem_real8(dest, src, bytes, offset)
    type(occaMemory), intent(out) :: dest
    real(8),          intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaAsyncCopyPtrToMem_fc(dest, src, bytes, offset)
  end subroutine occaAsyncCopyPtrToMem_real8

  subroutine occaAsyncCopyPtrToMem_char(dest, src, bytes, offset)
    type(occaMemory), intent(out) :: dest
    character,        intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaAsyncCopyPtrToMem_fc(dest, src, bytes, offset)
  end subroutine occaAsyncCopyPtrToMem_char

  subroutine occaAsyncCopyMemToPtr_int4(dest, src, bytes, offset)
    integer(4),       intent(out) :: dest
    type(occaMemory), intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaAsyncCopyMemToPtr_fc(dest, src, bytes, offset)
  end subroutine occaAsyncCopyMemToPtr_int4

  subroutine occaAsyncCopyMemToPtr_int8(dest, src, bytes, offset)
    integer(8),       intent(out) :: dest
    type(occaMemory), intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaAsyncCopyMemToPtr_fc(dest, src, bytes, offset)
  end subroutine occaAsyncCopyMemToPtr_int8

  subroutine occaAsyncCopyMemToPtr_real4(dest, src, bytes, offset)
    real(4),          intent(out) :: dest
    type(occaMemory), intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaAsyncCopyMemToPtr_fc(dest, src, bytes, offset)
  end subroutine occaAsyncCopyMemToPtr_real4

  subroutine occaAsyncCopyMemToPtr_real8(dest, src, bytes, offset)
    real(8),          intent(out) :: dest
    type(occaMemory), intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaAsyncCopyMemToPtr_fc(dest, src, bytes, offset)
  end subroutine occaAsyncCopyMemToPtr_real8

  subroutine occaAsyncCopyMemToPtr_char(dest, src, bytes, offset)
    character,        intent(out) :: dest
    type(occaMemory), intent(in)  :: src
    integer(8),       intent(in)  :: bytes
    integer(8),       intent(in)  :: offset

    call occaAsyncCopyMemToPtr_fc(dest, src, bytes, offset)
  end subroutine occaAsyncCopyMemToPtr_char

  subroutine occaCopyPtrToMemAuto_int4(dest, src)
    type(occaMemory), intent(out) :: dest
    integer(4),       intent(in)  :: src

    call occaCopyPtrToMemAuto_fc(dest, src)
  end subroutine occaCopyPtrToMemAuto_int4

  subroutine occaCopyPtrToMemAuto_int8(dest, src)
    type(occaMemory), intent(out) :: dest
    integer(8),       intent(in)  :: src

    call occaCopyPtrToMemAuto_fc(dest, src)
  end subroutine occaCopyPtrToMemAuto_int8

  subroutine occaCopyPtrToMemAuto_real4(dest, src)
    type(occaMemory), intent(out) :: dest
    real(4),          intent(in)  :: src

    call occaCopyPtrToMemAuto_fc(dest, src)
  end subroutine occaCopyPtrToMemAuto_real4

  subroutine occaCopyPtrToMemAuto_real8(dest, src)
    type(occaMemory), intent(out) :: dest
    real(8),          intent(in)  :: src

    call occaCopyPtrToMemAuto_fc(dest, src)
  end subroutine occaCopyPtrToMemAuto_real8

  subroutine occaCopyPtrToMemAuto_char(dest, src)
    type(occaMemory), intent(out) :: dest
    character,        intent(in)  :: src

    call occaCopyPtrToMemAuto_fc(dest, src)
  end subroutine occaCopyPtrToMemAuto_char

  subroutine occaCopyMemToPtrAuto_int4(dest, src)
    integer(4),       intent(out) :: dest
    type(occaMemory), intent(in)  :: src

    call occaCopyMemToPtrAuto_fc(dest, src)
  end subroutine occaCopyMemToPtrAuto_int4

  subroutine occaCopyMemToPtrAuto_int8(dest, src)
    integer(8),       intent(out) :: dest
    type(occaMemory), intent(in)  :: src

    call occaCopyMemToPtrAuto_fc(dest, src)
  end subroutine occaCopyMemToPtrAuto_int8

  subroutine occaCopyMemToPtrAuto_real4(dest, src)
    real(4),          intent(out) :: dest
    type(occaMemory), intent(in)  :: src

    call occaCopyMemToPtrAuto_fc(dest, src)
  end subroutine occaCopyMemToPtrAuto_real4

  subroutine occaCopyMemToPtrAuto_real8(dest, src)
    real(8),          intent(out) :: dest
    type(occaMemory), intent(in)  :: src

    call occaCopyMemToPtrAuto_fc(dest, src)
  end subroutine occaCopyMemToPtrAuto_real8

  subroutine occaCopyMemToPtrAuto_char(dest, src)
    character,        intent(out) :: dest
    type(occaMemory), intent(in)  :: src

    call occaCopyMemToPtrAuto_fc(dest, src)
  end subroutine occaCopyMemToPtrAuto_char

  subroutine occaAsyncCopyPtrToMemAuto_int4(dest, src)
    type(occaMemory), intent(out) :: dest
    integer(4),       intent(in)  :: src

    call occaAsyncCopyPtrToMemAuto_fc(dest, src)
  end subroutine occaAsyncCopyPtrToMemAuto_int4

  subroutine occaAsyncCopyPtrToMemAuto_int8(dest, src)
    type(occaMemory), intent(out) :: dest
    integer(8),       intent(in)  :: src

    call occaAsyncCopyPtrToMemAuto_fc(dest, src)
  end subroutine occaAsyncCopyPtrToMemAuto_int8

  subroutine occaAsyncCopyPtrToMemAuto_real4(dest, src)
    type(occaMemory), intent(out) :: dest
    real(4),          intent(in)  :: src

    call occaAsyncCopyPtrToMemAuto_fc(dest, src)
  end subroutine occaAsyncCopyPtrToMemAuto_real4

  subroutine occaAsyncCopyPtrToMemAuto_real8(dest, src)
    type(occaMemory), intent(out) :: dest
    real(8),          intent(in)  :: src

    call occaAsyncCopyPtrToMemAuto_fc(dest, src)
  end subroutine occaAsyncCopyPtrToMemAuto_real8

  subroutine occaAsyncCopyPtrToMemAuto_char(dest, src)
    type(occaMemory), intent(out) :: dest
    character,        intent(in)  :: src

    call occaAsyncCopyPtrToMemAuto_fc(dest, src)
  end subroutine occaAsyncCopyPtrToMemAuto_char

  subroutine occaAsyncCopyMemToPtrAuto_int4(dest, src)
    integer(4),       intent(out) :: dest
    type(occaMemory), intent(in)  :: src

    call occaAsyncCopyMemToPtrAuto_fc(dest, src)
  end subroutine occaAsyncCopyMemToPtrAuto_int4

  subroutine occaAsyncCopyMemToPtrAuto_int8(dest, src)
    integer(8),       intent(out) :: dest
    type(occaMemory), intent(in)  :: src

    call occaAsyncCopyMemToPtrAuto_fc(dest, src)
  end subroutine occaAsyncCopyMemToPtrAuto_int8

  subroutine occaAsyncCopyMemToPtrAuto_real4(dest, src)
    real(4),          intent(out) :: dest
    type(occaMemory), intent(in)  :: src

    call occaAsyncCopyMemToPtrAuto_fc(dest, src)
  end subroutine occaAsyncCopyMemToPtrAuto_real4

  subroutine occaAsyncCopyMemToPtrAuto_real8(dest, src)
    real(8),          intent(out) :: dest
    type(occaMemory), intent(in)  :: src

    call occaAsyncCopyMemToPtrAuto_fc(dest, src)
  end subroutine occaAsyncCopyMemToPtrAuto_real8

  subroutine occaAsyncCopyMemToPtrAuto_char(dest, src)
    character,        intent(out) :: dest
    type(occaMemory), intent(in)  :: src

    call occaAsyncCopyMemToPtrAuto_fc(dest, src)
  end subroutine occaAsyncCopyMemToPtrAuto_char

  subroutine occaSysCall_(stat, cmdline)
    integer(4),       intent(out) :: stat
    character(len=*), intent(in)  :: cmdline

    call occaSysCall_fc(stat, 0, cmdline)
  end subroutine occaSysCall_

  ! subroutine occaSysCall_output(stat, output, cmdline)
  !   integer(4),       intent(out) :: stat
  !   character(len=*), intent(out) :: output
  !   character(len=*), intent(in)  :: cmdline

  !   call occaSysCall_fc(stat, output, cmdline)
  ! end subroutine occaSysCall_output

end module occa
