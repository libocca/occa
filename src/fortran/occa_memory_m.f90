module occa_memory_m
  ! occa/c/memory.h

  use occa_types_m

  implicit none

  interface
    ! bool occaMemoryIsInitialized(occaMemory memory);
    logical(kind=C_bool) function occaMemoryIsInitialized(memory) &
                                  bind(C, name="occaMemoryIsInitialized")
      import occaMemory, C_bool
      implicit none
      type(occaMemory), value :: memory
    end function

    ! void* occaMemoryPtr(occaMemory memory, occaJson props);
    type(C_void_ptr) function occaMemoryPtr(memory, props) &
                              bind(C, name="occaMemoryPtr")
      import occaMemory, occaJson, C_void_ptr
      implicit none
      type(occaMemory), value :: memory
      type(occaJson), value :: props
    end function

    ! occaDevice occaMemoryGetDevice(occaMemory memory);
    type(occaDevice) function occaMemoryGetDevice(memory) &
                              bind(C, name="occaMemoryGetDevice")
      import occaMemory, occaDevice
      implicit none
      type(occaMemory), value :: memory
    end function

    ! occaJson occaMemoryGetProperties(occaMemory memory);
    type(occaJson) function occaMemoryGetProperties(memory) &
                                  bind(C, name="occaMemoryGetProperties")
      import occaMemory, occaJson
      implicit none
      type(occaMemory), value :: memory
    end function

    ! occaUDim_t occaMemorySize(occaMemory memory);
    integer(occaUDim_t) function occaMemorySize(memory) &
                                 bind(C, name="occaMemorySize")
      import occaMemory, occaUDim_t
      implicit none
      type(occaMemory), value :: memory
    end function

    ! occaMemory occaMemorySlice(occaMemory memory,
    !                            const occaDim_t offset,
    !                            const occaDim_t bytes);
    type(occaMemory) function occaMemorySlice(memory, offset, bytes) &
                              bind(C, name="occaMemorySlice")
      import occaMemory, occaDim_t
      implicit none
      type(occaMemory), value :: memory
      integer(occaDim_t), value, intent(in) :: offset, bytes
    end function

    ! void occaMemcpy(void *dest,
    !                 const void *src,
    !                 const occaUDim_t bytes,
    !                 occaJson props);
    subroutine occaMemcpy(dest, src, bytes, props) &
               bind(C, name="occaMemcpy")
      import occaMemory, C_void_ptr, occaUDim_t, occaJson
      implicit none
      type(occaMemory), value :: dest
      type(C_void_ptr), value, intent(in) :: src
      integer(occaUDim_t), value, intent(in) :: bytes
      type(occaJson), value :: props
    end subroutine

    ! void occaCopyMemToMem(occaMemory dest,
    !                       occaMemory src,
    !                       const occaUDim_t bytes,
    !                       const occaUDim_t destOffset,
    !                       const occaUDim_t srcOffset,
    !                       occaJson props);
    subroutine occaCopyMemToMem(dest, src, bytes, destOffset, srcOffset, props) &
               bind(C, name="occaCopyMemToMem")
      import occaMemory, C_void_ptr, occaUDim_t, occaJson
      implicit none
      type(occaMemory), value :: dest
      type(C_void_ptr), value, intent(in) :: src
      integer(occaUDim_t), value, intent(in) :: bytes, destOffset, srcOffset
      type(occaJson), value :: props
    end subroutine

    ! void occaCopyPtrToMem(occaMemory dest,
    !                       const void *src,
    !                       const occaUDim_t bytes,
    !                       const occaUDim_t offset,
    !                       occaJson props);
    subroutine occaCopyPtrToMem(dest, src, bytes, offset, props) &
               bind(C, name="occaCopyPtrToMem")
      import occaMemory, C_void_ptr, occaUDim_t, occaJson
      implicit none
      type(occaMemory), value :: dest
      type(C_void_ptr), value, intent(in) :: src
      integer(occaUDim_t), value, intent(in) :: bytes, offset
      type(occaJson), value :: props
    end subroutine

    ! void occaCopyMemToPtr(void *dest,
    !                       occaMemory src,
    !                       const occaUDim_t bytes,
    !                       const occaUDim_t offset,
    !                       occaJson props);
    subroutine occaCopyMemToPtr(dest, src, bytes, offset, props) &
               bind(C, name="occaCopyMemToPtr")
      import C_void_ptr, occaMemory, occaUDim_t, occaJson
      implicit none
      type(C_void_ptr), value :: dest
      type(occaMemory), value :: src
      integer(occaUDim_t), value, intent(in) :: bytes, offset
      type(occaJson), value :: props
    end subroutine


    ! occaMemory occaMemoryClone(occaMemory memory);
    type(occaMemory) function occaMemoryClone(memory) &
                              bind(C, name="occaMemoryClone")
      import occaMemory
      implicit none
      type(occaMemory), value :: memory
    end function

    ! void occaMemoryDetach(occaMemory memory);
    subroutine occaMemoryDetach(memory) bind(C, name="occaMemoryDetach")
      import occaMemory
      implicit none
      type(occaMemory), value :: memory
    end subroutine

    ! occaMemory occaWrapCpuMemory(occaDevice device,
    !                              void *ptr,
    !                              occaUDim_t bytes,
    !                              occaJson props);
    type(occaMemory) function occaWrapCpuMemory(device, ptr, bytes, props) &
                              bind(C, name="occaWrapCpuMemory")
      import occaMemory, occaDevice, C_void_ptr, occaUDim_t, occaJson
      implicit none
      type(occaDevice), value :: device
      type(C_void_ptr), value :: ptr
      integer(occaUDim_t), value, intent(in) :: bytes
      type(occaJson), value :: props
    end function
  end interface

end module occa_memory_m
