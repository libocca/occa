module occa_device_m
  ! occa/c/device.h

  use occa_types_m

  implicit none

  interface
    ! occaDevice occaCreateDevice(occaType info);
    type(occaDevice) function occaCreateDevice(info) &
                              bind(C, name="occaCreateDevice")
      import occaType, occaDevice
      implicit none
      type(occaType) :: info
    end function

    ! occaDevice occaCreateDeviceFromString(const char *info);
    type(occaDevice) function occaCreateDeviceFromString(info) &
                              bind(C, name="occaCreateDeviceFromString")
      import C_char, occaDevice
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: info
    end function

    ! bool occaDeviceIsInitialized(occaDevice device);
    logical(kind=C_bool) function occaDeviceIsInitialized(device) &
                                  bind(C, name="occaDeviceIsInitialized")
      import occaDevice, C_bool
      implicit none
      type(occaDevice), value :: device
    end function

    ! const char* occaDeviceMode(occaDevice device);
    type(C_char_ptr) function occaDeviceMode(device) &
                              bind(C, name="occaDeviceMode")
      import occaDevice, C_char_ptr
      implicit none
      type(occaDevice), value :: device
    end function

    ! occaJson occaDeviceGetProperties(occaDevice device);
    type(occaJson) function occaDeviceGetProperties(device) &
                                  bind(C, name="occaDeviceGetProperties")
      import occaJson, occaDevice
      implicit none
      type(occaDevice), value :: device
    end function

    ! occaJson occaDeviceGetKernelProperties(occaDevice device);
    type(occaJson) function occaDeviceGetKernelProperties(device) &
                                  bind(C, name="occaDeviceGetKernelProperties")
      import occaJson, occaDevice
      implicit none
      type(occaDevice), value :: device
    end function

    ! occaJson occaDeviceGetMemoryProperties(occaDevice device);
    type(occaJson) function occaDeviceGetMemoryProperties(device) &
                                  bind(C, name="occaDeviceGetMemoryProperties")
      import occaJson, occaDevice
      implicit none
      type(occaDevice), value :: device
    end function

    ! occaJson occaDeviceGetStreamProperties(occaDevice device);
    type(occaJson) function occaDeviceGetStreamProperties(device) &
                                  bind(C, name="occaDeviceGetStreamProperties")
      import occaJson, occaDevice
      implicit none
      type(occaDevice), value :: device
    end function

    ! occaUDim_t occaDeviceMemorySize(occaDevice device);
    integer(occaUDim_t) function occaDeviceMemorySize(device) &
                                 bind(C, name="occaDeviceMemorySize")
      import occaDevice, occaUDim_t
      implicit none
      type(occaDevice), value :: device
    end function

    ! occaUDim_t occaDeviceMemoryAllocated(occaDevice device);
    integer(occaUDim_t) function occaDeviceMemoryAllocated(device) &
                                 bind(C, name="occaDeviceMemoryAllocated")
      import occaDevice, occaUDim_t
      implicit none
      type(occaDevice), value :: device
    end function

    ! void occaDeviceFinish(occaDevice device);
    subroutine occaDeviceFinish(device) bind(C, name="occaDeviceFinish")
      import occaDevice
      implicit none
      type(occaDevice), value :: device
    end subroutine

    ! bool occaDeviceHasSeparateMemorySpace(occaDevice device);
    logical(kind=C_bool) function occaDeviceHasSeparateMemorySpace(device) &
                                  bind(C, name="occaDeviceHasSeparateMemorySpace")
      import occaDevice, C_bool
      implicit none
      type(occaDevice), value :: device
    end function

    ! ---[ Stream ]-------------------------
    ! occaStream occaDeviceCreateStream(occaDevice device,
    !                                   occaJson props);
    type(occaStream) function occaDeviceCreateStream(device, props) &
                              bind(C, name="occaDeviceCreateStream")
      import occaJson, occaDevice, occaStream
      implicit none
      type(occaDevice), value :: device
      type(occaJson), value :: props
    end function

    ! occaStream occaDeviceGetStream(occaDevice device);
    type(occaStream) function occaDeviceGetStream(device) &
                              bind(C, name="occaDeviceGetStream")
      import occaDevice, occaStream
      implicit none
      type(occaDevice), value :: device
    end function

    ! void occaDeviceSetStream(occaDevice device, occaStream stream);
    subroutine occaDeviceSetStream(device, stream) &
               bind(C, name="occaDeviceSetStream")
      import occaDevice, occaStream
      implicit none
      type(occaDevice), value :: device
      type(occaStream), value :: stream
    end subroutine

    ! occaStreamTag occaDeviceTagStream(occaDevice device);
    type(occaStreamTag) function occaDeviceTagStream(device) &
                                 bind(C, name="occaDeviceTagStream")
      import occaDevice, occaStreamTag
      implicit none
      type(occaDevice), value :: device
    end function

    ! void occaDeviceWaitForTag(occaDevice device, occaStreamTag tag);
    subroutine occaDeviceWaitForTag(device, tag) &
               bind(C, name="occaDeviceWaitForTag")
      import occaDevice, occaStreamTag
      implicit none
      type(occaDevice), value :: device
      type(occaStreamTag), value :: tag
    end subroutine

    ! double occaDeviceTimeBetweenTags(occaDevice device,
    !                                  occaStreamTag startTag,
    !                                  occaStreamTag endTag);
    real(C_double) function occaDeviceTimeBetweenTags(device, startTag, endTag) &
                            bind(C, name="occaDeviceTimeBetweenTags")
      import occaDevice, occaStreamTag, C_double
      implicit none
      type(occaDevice), value :: device
      type(occaStreamTag), value :: startTag, endTag
    end function
    ! ======================================

    ! ---[ Kernel ]-------------------------
    ! occaKernel occaDeviceBuildKernel(occaDevice device,
    !                                  const char *filename,
    !                                  const char *kernelName,
    !                                  const occaJson props);
    type(occaKernel) function occaDeviceBuildKernel(device, &
                                                    filename, &
                                                    kernelName, &
                                                    props) &
                              bind(C, name="occaDeviceBuildKernel")
      import C_char, occaKernel, occaDevice, occaJson
      implicit none
      type(occaDevice), value :: device
      character(len=1,kind=C_char), dimension(*), intent(in) :: filename, &
                                                                kernelName
      type(occaJson), value, intent(in) :: props
    end function

    ! occaKernel occaDeviceBuildKernelFromString(occaDevice device,
    !                                            const char *str,
    !                                            const char *kernelName,
    !                                            const occaJson props);
    type(occaKernel) function occaDeviceBuildKernelFromString(device, &
                                                              str, &
                                                              kernelName, &
                                                              props) &
                              bind(C, name="occaDeviceBuildKernelFromString")
      import C_char, occaKernel, occaDevice, occaJson
      implicit none
      type(occaDevice), value :: device
      character(len=1,kind=C_char), dimension(*), intent(in) :: str, &
                                                                kernelName
      type(occaJson), value, intent(in) :: props
    end function

    ! occaKernel occaDeviceBuildKernelFromBinary(occaDevice device,
    !                                            const char *filename,
    !                                            const char *kernelName,
    !                                            const occaJson props);
    type(occaKernel) function occaDeviceBuildKernelFromBinary(device, &
                                                              filename, &
                                                              kernelName, &
                                                              props) &
                              bind(C, name="occaDeviceBuildKernelFromBinary")
      import C_char, occaKernel, occaDevice, occaJson
      implicit none
      type(occaDevice), value :: device
      character(len=1,kind=C_char), dimension(*), intent(in) :: filename, &
                                                                kernelName
      type(occaJson), value, intent(in) :: props
    end function
    ! ======================================

    ! ---[ Memory ]-------------------------
    ! occaMemory occaDeviceMalloc(occaDevice device,
    !                             const occaUDim_t bytes,
    !                             const void *src,
    !                             occaJson props);
    type(occaMemory) function occaDeviceMalloc(device, bytes, src, props) &
                              bind(C, name="occaDeviceMalloc")
      import C_void_ptr, occaMemory, occaDevice, occaJson, occaUDim_t
      implicit none
      type(occaDevice), value :: device
      integer(occaUDim_t), value, intent(in) :: bytes
      type(C_void_ptr), value, intent(in) :: src
      type(occaJson), value :: props
    end function

    ! occaMemory occaDeviceTypedMalloc(occaDevice device,
    !                                  const occaUDim_t entries,
    !                                  const occaDtype dtype,
    !                                  const void *src,
    !                                  occaJson props);
    type(occaMemory) function occaDeviceTypedMalloc(device, &
                                                    entries, &
                                                    dtype, &
                                                    src, &
                                                    props) &
                              bind(C, name="occaDeviceTypedMalloc")
      import C_void_ptr, occaMemory, occaDevice, occaJson, occaUDim_t, &
             occaDtype
      implicit none
      type(occaDevice), value :: device
      integer(occaUDim_t), value, intent(in) :: entries
      type(occaDtype), value, intent(in) :: dtype
      type(C_void_ptr), value, intent(in) :: src
      type(occaJson), value :: props
    end function

    ! void* occaDeviceUMalloc(occaDevice device,
    !                         const occaUDim_t bytes,
    !                         const void *src,
    !                         occaJson props);
    type(C_void_ptr) function occaDeviceUMalloc(device, bytes, src, props) &
                              bind(C, name="occaDeviceUMalloc")
      import C_void_ptr, occaDevice, occaJson, occaUDim_t
      implicit none
      type(occaDevice), value :: device
      integer(occaUDim_t), value, intent(in) :: bytes
      type(C_void_ptr), value, intent(in) :: src
      type(occaJson), value :: props
    end function

    ! void* occaDeviceTypedUMalloc(occaDevice device,
    !                              const occaUDim_t entries,
    !                              const occaDtype type,
    !                              const void *src,
    !                              occaJson props);
    type(C_void_ptr) function occaDeviceTypedUMalloc(device, &
                                                     entries, &
                                                     dtype, &
                                                     src, &
                                                     props) &
                              bind(C, name="occaDeviceTypedUMalloc")
      import C_void_ptr, occaDevice, occaJson, occaUDim_t, occaDtype
      implicit none
      type(occaDevice), value :: device
      integer(occaUDim_t), value, intent(in) :: entries
      type(occaDtype), value, intent(in) :: dtype
      type(C_void_ptr), value, intent(in) :: src
      type(occaJson), value :: props
    end function
    ! ======================================
  end interface

end module occa_device_m
