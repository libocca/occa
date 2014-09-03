module occa
  implicit none

  public

c ---[ TypeCasting ]------------------
  type :: occaDevice
     private
     integer, pointer :: p
  end type occaDevice

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

  type :: occaKernelInfo
     private
     integer, pointer :: p
  end type occaKernelInfo

  type :: occaDevice
     private
     integer, pointer :: p
  end type occaDevice

  interface occaType_t(value_)
     function occaType_int4_c(value_)
       implicit none
       integer(4), intent(in) :: value_

       call occaInt_fc(value_)
     end function occaType_int4_c

     function occaType_int8_c(value_)
       implicit none
       integer(8), intent(in) :: value_

       call occaLong_fc(value_)
     end function occaType_int8_c

     function occaType_real4_c(value_)
       implicit none
       real(4), intent(in) :: value_

       call occaFloat_fc(value_)
     end function occaType_real4_c

     function occaType_real8_c(value_)
       implicit none
       real(8), intent(in) :: value_

       call occaDouble_fc(value_)
     end function occaType_real8_c

     function occaType_str_c(value_)
       implicit none
       character(len=*), intent(in) :: value_

       call occaString_fc(value_)
     end function occaType_str_c
  end interface occaString
c ====================================


c ---[ Device ]-----------------------
  interface occaDeviceMode
     const char* occaDeviceMode_fc(occaDevice device)
  end interface occaDeviceMode

  interface occaDeviceSetCompiler
     void occaDeviceSetCompiler_fc(occaDevice device, const char *compiler)
  end interface occaDeviceSetCompiler

  interface occaDeviceSetCompilerFlags
     void occaDeviceSetCompilerFlags_fc(occaDevice device, const char *compilerFlags)
  end interface occaDeviceSetCompilerFlags

  interface occaGetDevice
     occaDevice occaGetDevice_fc(const char *mode, int arg1, int arg2)
  end interface occaGetDevice

  interface occaBuildKernelFromSource
     occaKernel occaBuildKernelFromSource_fc(occaDevice device, const char *filename, const char *functionName, occaKernelInfo info)
  end interface occaBuildKernelFromSource

  interface occaBuildKernelFromBinary
     occaKernel occaBuildKernelFromBinary_fc(occaDevice device, const char *filename, const char *functionName)
  end interface occaBuildKernelFromBinary

  interface occaBuildKernelFromLoopy
     occaKernel occaBuildKernelFromLoopy_fc(occaDevice device, const char *filename, const char *functionName, const char *pythonCode)
  end interface occaBuildKernelFromLoopy

  interface occaDeviceMalloc
     occaMemory occaDeviceMalloc_fc(occaDevice device, uintptr_t bytes, void *source)
  end interface occaDeviceMalloc

  interface occaDeviceFlush
     void occaDeviceFlush_fc(occaDevice device)
  end interface occaDeviceFlush
  interface occaDeviceFinish
     void occaDeviceFinish_fc(occaDevice device)
  end interface occaDeviceFinish

  interface occaDeviceGenStream
     occaStream occaDeviceGenStream_fc(occaDevice device)
  end interface occaDeviceGenStream
  interface occaDeviceGetStream
     occaStream occaDeviceGetStream_fc(occaDevice device)
  end interface occaDeviceGetStream
  interface occaDeviceSetStream
     void occaDeviceSetStream_fc(occaDevice device, occaStream stream)
  end interface occaDeviceSetStream

  interface occaDeviceTagStream
     occaTag occaDeviceTagStream_fc(occaDevice device)
  end interface occaDeviceTagStream

  double occaDeviceTimeBetweenTags_fc(occaDevice device, occaTag startTag, occaTag endTag)

  interface occaDeviceStreamFree
     void occaDeviceStreamFree_fc(occaDevice device, occaStream stream)
  end interface occaDeviceStreamFree

  interface occaDeviceFree
     void occaDeviceFree_fc(occaDevice device)
  end interface occaDeviceFree
c ====================================


c ---[ Kernel ]-----------------------
  interface occaKernelMode
     const char* occaKernelMode_fc(occaKernel kernel)
  end interface occaKernelMode

  interface occaKernelPreferredDimSize
     int occaKernelPreferredDimSize_fc(occaKernel kernel)
  end interface occaKernelPreferredDimSize

  interface occaKernelSetWorkingDims
     void occaKernelSetWorkingDims_fc(occaKernel kernel, int dims, occaDim items, occaDim groups)
  end interface occaKernelSetWorkingDims

  interface occaKernelSetAllWorkingDims
     void occaKernelSetAllWorkingDims_fc(occaKernel kernel, int dims, uintptr_t itemsX, uintptr_t itemsY, uintptr_t itemsZ, uintptr_t groupsX, uintptr_t groupsY, uintptr_t groupsZ)
  end interface occaKernelSetAllWorkingDims

  interface occaKernelTimeTaken
     double occaKernelTimeTaken_fc(occaKernel kernel)
  end interface occaKernelTimeTaken

  interface occaGenArgumentList
     occaArgumentList occaGenArgumentList_fc()
  end interface occaGenArgumentList

  interface occaArgumentListClear
     void occaArgumentListClear_fc(occaArgumentList list)
  end interface occaArgumentListClear

  interface occaArgumentListFree
     void occaArgumentListFree_fc(occaArgumentList list)
  end interface occaArgumentListFree

  interface occaArgumentListAddArg
     void occaArgumentListAddArg_fc(occaArgumentList list, int argPos, void *type)
  end interface occaArgumentListAddArg

  interface occaKernelRun_
     void occaKernelRun__fc(occaKernel kernel, occaArgumentList list)
  end interface occaKernelRun_

  interface occaKernelFree
     void occaKernelFree_fc(occaKernel kernel)
  end interface occaKernelFree

  interface occaGenKernelInfo
     occaKernelInfo occaGenKernelInfo_fc()
  end interface occaGenKernelInfo

  interface occaKernelInfoAddDefine
     void occaKernelInfoAddDefine_fc(occaKernelInfo info, const char *macro, occaType value_)
  end interface occaKernelInfoAddDefine

  interface occaKernelInfoFree
     void occaKernelInfoFree_fc(occaKernelInfo info)
  end interface occaKernelInfoFree
c ====================================


c ---[ Memory ]-----------------------
  interface occaMemoryMode
     const char* occaMemoryMode_fc(occaMemory memory)
  end interface occaMemoryMode

  interface occaCopyMemToMem
     void occaCopyMemToMem_fc(occaMemory dest, occaMemory src, const uintptr_t bytes, const uintptr_t destOffset, const uintptr_t srcOffset)
  end interface occaCopyMemToMem

  interface occaCopyPtrToMem
     void occaCopyPtrToMem_fc(occaMemory dest, const void *src, const uintptr_t bytes, const uintptr_t offset)
  end interface occaCopyPtrToMem

  interface occaCopyMemToPtr
     void occaCopyMemToPtr_fc(void *dest, occaMemory src, const uintptr_t bytes, const uintptr_t offset)
  end interface occaCopyMemToPtr

  interface occaAsyncCopyMemToMem
     void occaAsyncCopyMemToMem_fc(occaMemory dest, occaMemory src, const uintptr_t bytes, const uintptr_t destOffset, const uintptr_t srcOffset)
  end interface occaAsyncCopyMemToMem

  interface occaAsyncCopyPtrToMem
     void occaAsyncCopyPtrToMem_fc(occaMemory dest, const void *src, const uintptr_t bytes, const uintptr_t offset)
  end interface occaAsyncCopyPtrToMem

  interface occaAsyncCopyMemToPtr
     void occaAsyncCopyMemToPtr_fc(void *dest, occaMemory src, const uintptr_t bytes, const uintptr_t offset)
  end interface occaAsyncCopyMemToPtr

  interface occaMemorySwap
     void occaMemorySwap_fc(occaMemory memoryA, occaMemory memoryB)
  end interface occaMemorySwap

  interface occaMemoryFree
     void occaMemoryFree_fc(occaMemory memory)
  end interface occaMemoryFree
c ====================================


end module occa
