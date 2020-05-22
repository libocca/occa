module occa_kernelBuilder_m
  ! occa/c/kernelBuilder.h

  use occa_types_m

  implicit none

  interface
    ! occaKernelBuilder occaKernelBuilderFromInlinedOkl(
    !   occaScope scope,
    !   const char *kernelSource,
    !   const char *kernelName
    ! );
    type(occaKernelBuilder) &
    function occaKernelBuilderFromInlinedOkl(scope, kernelSource, kernelName) &
             bind(C, name="occaKernelBuilderFromInlinedOkl")
      import occaKernelBuilder, occaScope, C_char
      implicit none
      type(occaScope), value :: scope
      character(len=1,kind=C_char), dimension(*), intent(in) :: kernelSource, &
                                                                kernelName
    end function

    ! void occaKernelBuilderRun(
    !   occaKernelBuilder kernelBuilder,
    !   occaScope scope
    ! );
    subroutine occaKernelBuilderRun(kernelBuilder, scope) &
               bind(C, name="occaKernelBuilderRun")
      import occaKernelBuilder, occaScope
      implicit none
      type(occaKernelBuilder), value :: kernelBuilder
      type(occaScope), value :: scope
    end subroutine
  end interface

end module occa_kernelBuilder_m
