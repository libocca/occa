  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argPos = 0;

    const kernelArg *kArgs[1] = {&arg0};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 1; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[2] = {&arg0, &arg1};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 2; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[3] = {&arg0, &arg1, &arg2};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 3; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[4] = {&arg0, &arg1, &arg2, &arg3};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 4; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[5] = {&arg0, &arg1, &arg2, &arg3, &arg4};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 5; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[6] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 6; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[7] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5, &arg6};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 7; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[8] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5, &arg6, &arg7};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 8; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[9] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5, &arg6, &arg7, &arg8};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 9; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[10] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 10; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[11] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 11; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[12] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 12; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[13] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 13; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[14] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 14; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[15] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 15; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[16] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 16; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[17] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 17; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[18] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 18; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[19] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 19; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[20] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 20; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[21] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 21; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[22] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 22; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[23] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 23; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[24] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 24; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[25] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 25; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[26] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 26; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[27] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 27; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[28] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 28; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[29] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 29; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[30] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 30; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[31] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 31; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[32] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 32; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[33] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 33; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[34] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 34; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[35] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 35; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[36] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 36; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[37] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 37; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[38] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 38; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[39] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 39; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[40] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 40; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[41] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 41; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[42] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 42; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[43] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 43; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[44] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 44; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[45] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 45; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[46] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44, 
                                  &arg45};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 46; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[47] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44, 
                                  &arg45, &arg46};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 47; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[48] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44, 
                                  &arg45, &arg46, &arg47};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 48; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[49] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44, 
                                  &arg45, &arg46, &arg47, &arg48};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 49; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

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

    const kernelArg *kArgs[50] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44, 
                                  &arg45, &arg46, &arg47, &arg48, &arg49};

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argPos++, sizeof(void*), NULL));

    for(int i = 0; i < 50; ++i){
      OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                    clSetKernelArg(kernel_, argPos++, kArgs[i]->size, kArgs[i]->data()));

      if(kArgs[i]->hasTwoArgs)
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Texture Kernel Argument for Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argPos++, sizeof(void*), kArgs[i]->arg2.void_));
    }

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }