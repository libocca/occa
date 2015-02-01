  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39,  const kernelArg &arg40){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg40.size, arg40.data()));
    if(arg40.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg40.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39,  const kernelArg &arg40,  const kernelArg &arg41){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg40.size, arg40.data()));
    if(arg40.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg40.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg41.size, arg41.data()));
    if(arg41.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg41.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39,  const kernelArg &arg40,  const kernelArg &arg41, 
                      const kernelArg &arg42){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg40.size, arg40.data()));
    if(arg40.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg40.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg41.size, arg41.data()));
    if(arg41.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg41.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg42.size, arg42.data()));
    if(arg42.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg42.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39,  const kernelArg &arg40,  const kernelArg &arg41, 
                      const kernelArg &arg42,  const kernelArg &arg43){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg40.size, arg40.data()));
    if(arg40.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg40.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg41.size, arg41.data()));
    if(arg41.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg41.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg42.size, arg42.data()));
    if(arg42.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg42.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg43.size, arg43.data()));
    if(arg43.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg43.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39,  const kernelArg &arg40,  const kernelArg &arg41, 
                      const kernelArg &arg42,  const kernelArg &arg43,  const kernelArg &arg44){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg40.size, arg40.data()));
    if(arg40.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg40.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg41.size, arg41.data()));
    if(arg41.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg41.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg42.size, arg42.data()));
    if(arg42.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg42.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg43.size, arg43.data()));
    if(arg43.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg43.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg44.size, arg44.data()));
    if(arg44.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg44.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39,  const kernelArg &arg40,  const kernelArg &arg41, 
                      const kernelArg &arg42,  const kernelArg &arg43,  const kernelArg &arg44, 
                      const kernelArg &arg45){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg40.size, arg40.data()));
    if(arg40.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg40.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg41.size, arg41.data()));
    if(arg41.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg41.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg42.size, arg42.data()));
    if(arg42.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg42.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg43.size, arg43.data()));
    if(arg43.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg43.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg44.size, arg44.data()));
    if(arg44.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg44.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg45.size, arg45.data()));
    if(arg45.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg45.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39,  const kernelArg &arg40,  const kernelArg &arg41, 
                      const kernelArg &arg42,  const kernelArg &arg43,  const kernelArg &arg44, 
                      const kernelArg &arg45,  const kernelArg &arg46){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg40.size, arg40.data()));
    if(arg40.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg40.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg41.size, arg41.data()));
    if(arg41.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg41.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg42.size, arg42.data()));
    if(arg42.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg42.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg43.size, arg43.data()));
    if(arg43.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg43.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg44.size, arg44.data()));
    if(arg44.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg44.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg45.size, arg45.data()));
    if(arg45.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg45.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg46.size, arg46.data()));
    if(arg46.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg46.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39,  const kernelArg &arg40,  const kernelArg &arg41, 
                      const kernelArg &arg42,  const kernelArg &arg43,  const kernelArg &arg44, 
                      const kernelArg &arg45,  const kernelArg &arg46,  const kernelArg &arg47){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg40.size, arg40.data()));
    if(arg40.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg40.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg41.size, arg41.data()));
    if(arg41.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg41.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg42.size, arg42.data()));
    if(arg42.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg42.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg43.size, arg43.data()));
    if(arg43.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg43.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg44.size, arg44.data()));
    if(arg44.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg44.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg45.size, arg45.data()));
    if(arg45.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg45.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg46.size, arg46.data()));
    if(arg46.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg46.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg47.size, arg47.data()));
    if(arg47.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg47.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39,  const kernelArg &arg40,  const kernelArg &arg41, 
                      const kernelArg &arg42,  const kernelArg &arg43,  const kernelArg &arg44, 
                      const kernelArg &arg45,  const kernelArg &arg46,  const kernelArg &arg47, 
                      const kernelArg &arg48){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg40.size, arg40.data()));
    if(arg40.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg40.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg41.size, arg41.data()));
    if(arg41.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg41.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg42.size, arg42.data()));
    if(arg42.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg42.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg43.size, arg43.data()));
    if(arg43.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg43.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg44.size, arg44.data()));
    if(arg44.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg44.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg45.size, arg45.data()));
    if(arg45.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg45.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg46.size, arg46.data()));
    if(arg46.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg46.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg47.size, arg47.data()));
    if(arg47.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg47.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg48.size, arg48.data()));
    if(arg48.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg48.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29, 
                      const kernelArg &arg30,  const kernelArg &arg31,  const kernelArg &arg32, 
                      const kernelArg &arg33,  const kernelArg &arg34,  const kernelArg &arg35, 
                      const kernelArg &arg36,  const kernelArg &arg37,  const kernelArg &arg38, 
                      const kernelArg &arg39,  const kernelArg &arg40,  const kernelArg &arg41, 
                      const kernelArg &arg42,  const kernelArg &arg43,  const kernelArg &arg44, 
                      const kernelArg &arg45,  const kernelArg &arg46,  const kernelArg &arg47, 
                      const kernelArg &arg48,  const kernelArg &arg49){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg0.size, arg0.data()));
    if(arg0.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg0.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg1.size, arg1.data()));
    if(arg1.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg1.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg2.size, arg2.data()));
    if(arg2.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg2.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg3.size, arg3.data()));
    if(arg3.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg3.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg4.size, arg4.data()));
    if(arg4.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg4.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg5.size, arg5.data()));
    if(arg5.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg5.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg6.size, arg6.data()));
    if(arg6.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg6.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg7.size, arg7.data()));
    if(arg7.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg7.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg8.size, arg8.data()));
    if(arg8.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg8.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg9.size, arg9.data()));
    if(arg9.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg9.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg10.size, arg10.data()));
    if(arg10.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg10.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg11.size, arg11.data()));
    if(arg11.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg11.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg12.size, arg12.data()));
    if(arg12.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg12.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg13.size, arg13.data()));
    if(arg13.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg13.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg14.size, arg14.data()));
    if(arg14.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg14.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg15.size, arg15.data()));
    if(arg15.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg15.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg16.size, arg16.data()));
    if(arg16.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg16.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg17.size, arg17.data()));
    if(arg17.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg17.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg18.size, arg18.data()));
    if(arg18.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg18.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg19.size, arg19.data()));
    if(arg19.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg19.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg20.size, arg20.data()));
    if(arg20.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg20.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg21.size, arg21.data()));
    if(arg21.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg21.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg22.size, arg22.data()));
    if(arg22.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg22.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg23.size, arg23.data()));
    if(arg23.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg23.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg24.size, arg24.data()));
    if(arg24.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg24.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg25.size, arg25.data()));
    if(arg25.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg25.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg26.size, arg26.data()));
    if(arg26.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg26.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg27.size, arg27.data()));
    if(arg27.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg27.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg28.size, arg28.data()));
    if(arg28.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg28.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg29.size, arg29.data()));
    if(arg29.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg29.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg30.size, arg30.data()));
    if(arg30.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg30.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg31.size, arg31.data()));
    if(arg31.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg31.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg32.size, arg32.data()));
    if(arg32.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg32.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg33.size, arg33.data()));
    if(arg33.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg33.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg34.size, arg34.data()));
    if(arg34.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg34.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg35.size, arg35.data()));
    if(arg35.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg35.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg36.size, arg36.data()));
    if(arg36.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg36.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg37.size, arg37.data()));
    if(arg37.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg37.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg38.size, arg38.data()));
    if(arg38.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg38.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg39.size, arg39.data()));
    if(arg39.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg39.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg40.size, arg40.data()));
    if(arg40.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg40.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg41.size, arg41.data()));
    if(arg41.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg41.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg42.size, arg42.data()));
    if(arg42.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg42.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg43.size, arg43.data()));
    if(arg43.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg43.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg44.size, arg44.data()));
    if(arg44.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg44.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg45.size, arg45.data()));
    if(arg45.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg45.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg46.size, arg46.data()));
    if(arg46.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg46.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg47.size, arg47.data()));
    if(arg47.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg47.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg48.size, arg48.data()));
    if(arg48.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg48.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << argPos << "]",
                  clSetKernelArg(kernel_, argPos++, arg49.size, arg49.data()));
    if(arg49.hasTwoArgs)
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for [" << (argPos - 1) << "]",
                    clSetKernelArg(kernel_, argPos++, sizeof(void*), arg49.arg2.void_));

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }