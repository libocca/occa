module occa_base_m
  ! occa/c/base.h

  use occa_types_m

  implicit none

  interface
    ! ---[ Globals & Flags ]----------------
    ! occaJson occaSettings();
    type(occaJson) function occaSettings() bind(C, name="occaSettings")
      import occaJson
    end function

    ! void occaPrintModeInfo();
    pure subroutine occaPrintModeInfo() bind(C, name="occaPrintModeInfo")
    end subroutine
    ! ======================================

    ! ---[ Device ]-------------------------
    ! occaDevice occaHost();
    type(occaDevice) function occaHost() bind(C, name="occaHost")
      import occaDevice
    end function

    ! occaDevice occaGetDevice();
    type(occaDevice) function occaGetDevice() bind(C, name="occaGetDevice")
      import occaDevice
    end function

    ! void occaSetDevice(occaDevice device);
    subroutine occaSetDevice(device) bind(C, name="occaSetDevice")
      import occaDevice
      implicit none
      type(occaDevice), value :: device
    end subroutine

    ! void occaSetDeviceFromString(const char *info);
    subroutine occaSetDeviceFromString(info) &
               bind(C, name="occaSetDeviceFromString")
      import C_char
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: info
    end subroutine

    ! occaJson occaDeviceProperties();
    type(occaJson) function occaDeviceProperties() &
                                  bind(C, name="occaDeviceProperties")
      import occaJson
    end function

    ! void occaFinish();
    subroutine occaFinish() bind(C, name="occaFinish")
    end subroutine

    ! occaStream occaCreateStream(occaJson props);
    type(occaStream) function occaCreateStream(props) &
                              bind(C, name="occaCreateStream")
      import occaStream, occaJson
      implicit none
      type(occaJson), value :: props
    end function

    ! occaStream occaGetStream();
    type(occaStream) function occaGetStream() bind(C, name="occaGetStream")
      import occaStream
    end function

    ! void occaSetStream(occaStream stream);
    subroutine occaSetStream(stream) bind(C, name="occaSetStream")
      import occaStream
      implicit none
      type(occaStream), value :: stream
    end subroutine

    ! occaStreamTag occaTagStream();
    type(occaStreamTag) function occaTagStream() bind(C, name="occaTagStream")
      import occaStreamTag
    end function

    ! void occaWaitForTag(occaStreamTag tag);
    subroutine occaWaitForTag(tag) bind(C, name="occaWaitForTag")
      import occaStreamTag
      implicit none
      type(occaStreamTag), value :: tag
    end subroutine

    ! double occaTimeBetweenTags(occaStreamTag startTag, occaStreamTag endTag);
    real(C_double) function occaTimeBetweenTags(startTag, endTag) &
                            bind(C, name="occaTimeBetweenTags")
      import occaStreamTag, C_double
      implicit none
      type(occaStreamTag), value :: startTag, endTag
    end function
    ! ======================================

    ! ---[ Kernel ]-------------------------
    ! occaKernel occaBuildKernel(const char *filename,
    !                            const char *kernelName,
    !                            const occaJson props);
    type(occaKernel) function occaBuildKernel(filename, &
                                              kernelName, &
                                              props) &
                              bind(C, name="occaBuildKernel")
      import C_char, occaKernel, occaJson
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: filename, &
                                                                kernelName
      type(occaJson), value, intent(in) :: props
    end function

    ! occaKernel occaBuildKernelFromString(const char *source,
    !                                      const char *kernelName,
    !                                      const occaJson props);
    type(occaKernel) function occaBuildKernelFromString(str, &
                                                        kernelName, &
                                                        props) &
                              bind(C, name="occaBuildKernelFromString")
      import C_char, occaKernel, occaJson
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: str, &
                                                                kernelName
      type(occaJson), value, intent(in) :: props
    end function

    ! occaKernel occaBuildKernelFromBinary(const char *filename,
    !                                      const char *kernelName,
    !                                      const occaJson props);
    type(occaKernel) function occaBuildKernelFromBinary(filename, &
                                                        kernelName, &
                                                        props) &
                              bind(C, name="occaBuildKernelFromBinary")
      import C_char, occaKernel, occaJson
      implicit none
      character(len=1,kind=C_char), dimension(*), intent(in) :: filename, &
                                                                kernelName
      type(occaJson), value, intent(in) :: props
    end function
    ! ======================================

    ! ---[ Memory ]-------------------------
    ! occaMemory occaMalloc(const occaUDim_t bytes,
    !                       const void *src,
    !                       occaJson props);
    type(occaMemory) function occaMalloc(bytes, src, props) &
                              bind(C, name="occaMalloc")
      import occaMemory, occaUDim_t, C_void_ptr, occaJson
      implicit none
      integer(occaUDim_t), value, intent(in) :: bytes
      type(C_void_ptr), value, intent(in) :: src
      type(occaJson), value :: props
    end function

    ! occaMemory occaTypedMalloc(const occaUDim_t entries,
    !                            const occaDtype type,
    !                            const void *src,
    !                            occaJson props);
    type(occaMemory) function occaTypedMalloc(entries, type, src, props) &
                              bind(C, name="occaTypedMalloc")
      import occaMemory, occaUDim_t, occaDtype, C_void_ptr, occaJson
      implicit none
      integer(occaUDim_t), value, intent(in) :: entries
      type(occaDtype), value, intent(in) :: type
      type(C_void_ptr), value, intent(in) :: src
      type(occaJson), value :: props
    end function

    ! void* occaUMalloc(const occaUDim_t bytes,
    !                   const void *src,
    !                   occaJson props);
    type(C_void_ptr) function occaUMalloc(bytes, src, props) &
                              bind(C, name="occaUMalloc")
      import occaUDim_t, C_void_ptr, occaJson
      implicit none
      integer(occaUDim_t), value, intent(in) :: bytes
      type(C_void_ptr), value, intent(in) :: src
      type(occaJson), value :: props
    end function

    ! void* occaTypedUMalloc(const occaUDim_t entries,
    !                        const occaDtype type,
    !                        const void *src,
    !                        occaJson props);
    type(C_void_ptr) function occaTypedUMalloc(entries, type, src, props) &
                              bind(C, name="occaTypedUMalloc")
      import occaUDim_t, occaDtype, C_void_ptr, occaJson
      implicit none
      integer(occaUDim_t), value, intent(in) :: entries
      type(occaDtype), value, intent(in) :: type
      type(C_void_ptr), value, intent(in) :: src
      type(occaJson), value :: props
    end function
    ! ======================================
  end interface

end module occa_base_m
