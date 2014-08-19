  template <>
  void kernel_t<OpenCL>::operator () (const kernelArg &arg0){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [41]",
                  clSetKernelArg(kernel_, 41, arg40.size, arg40.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [41]",
                  clSetKernelArg(kernel_, 41, arg40.size, arg40.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [42]",
                  clSetKernelArg(kernel_, 42, arg41.size, arg41.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [41]",
                  clSetKernelArg(kernel_, 41, arg40.size, arg40.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [42]",
                  clSetKernelArg(kernel_, 42, arg41.size, arg41.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [43]",
                  clSetKernelArg(kernel_, 43, arg42.size, arg42.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [41]",
                  clSetKernelArg(kernel_, 41, arg40.size, arg40.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [42]",
                  clSetKernelArg(kernel_, 42, arg41.size, arg41.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [43]",
                  clSetKernelArg(kernel_, 43, arg42.size, arg42.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [44]",
                  clSetKernelArg(kernel_, 44, arg43.size, arg43.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [41]",
                  clSetKernelArg(kernel_, 41, arg40.size, arg40.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [42]",
                  clSetKernelArg(kernel_, 42, arg41.size, arg41.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [43]",
                  clSetKernelArg(kernel_, 43, arg42.size, arg42.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [44]",
                  clSetKernelArg(kernel_, 44, arg43.size, arg43.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [45]",
                  clSetKernelArg(kernel_, 45, arg44.size, arg44.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [41]",
                  clSetKernelArg(kernel_, 41, arg40.size, arg40.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [42]",
                  clSetKernelArg(kernel_, 42, arg41.size, arg41.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [43]",
                  clSetKernelArg(kernel_, 43, arg42.size, arg42.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [44]",
                  clSetKernelArg(kernel_, 44, arg43.size, arg43.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [45]",
                  clSetKernelArg(kernel_, 45, arg44.size, arg44.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [46]",
                  clSetKernelArg(kernel_, 46, arg45.size, arg45.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [41]",
                  clSetKernelArg(kernel_, 41, arg40.size, arg40.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [42]",
                  clSetKernelArg(kernel_, 42, arg41.size, arg41.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [43]",
                  clSetKernelArg(kernel_, 43, arg42.size, arg42.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [44]",
                  clSetKernelArg(kernel_, 44, arg43.size, arg43.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [45]",
                  clSetKernelArg(kernel_, 45, arg44.size, arg44.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [46]",
                  clSetKernelArg(kernel_, 46, arg45.size, arg45.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [47]",
                  clSetKernelArg(kernel_, 47, arg46.size, arg46.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [41]",
                  clSetKernelArg(kernel_, 41, arg40.size, arg40.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [42]",
                  clSetKernelArg(kernel_, 42, arg41.size, arg41.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [43]",
                  clSetKernelArg(kernel_, 43, arg42.size, arg42.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [44]",
                  clSetKernelArg(kernel_, 44, arg43.size, arg43.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [45]",
                  clSetKernelArg(kernel_, 45, arg44.size, arg44.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [46]",
                  clSetKernelArg(kernel_, 46, arg45.size, arg45.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [47]",
                  clSetKernelArg(kernel_, 47, arg46.size, arg46.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [48]",
                  clSetKernelArg(kernel_, 48, arg47.size, arg47.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [41]",
                  clSetKernelArg(kernel_, 41, arg40.size, arg40.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [42]",
                  clSetKernelArg(kernel_, 42, arg41.size, arg41.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [43]",
                  clSetKernelArg(kernel_, 43, arg42.size, arg42.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [44]",
                  clSetKernelArg(kernel_, 44, arg43.size, arg43.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [45]",
                  clSetKernelArg(kernel_, 45, arg44.size, arg44.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [46]",
                  clSetKernelArg(kernel_, 46, arg45.size, arg45.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [47]",
                  clSetKernelArg(kernel_, 47, arg46.size, arg46.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [48]",
                  clSetKernelArg(kernel_, 48, arg47.size, arg47.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [49]",
                  clSetKernelArg(kernel_, 49, arg48.size, arg48.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
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

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, 0, sizeof(void*), NULL));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [1]",
                  clSetKernelArg(kernel_, 1, arg0.size, arg0.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [2]",
                  clSetKernelArg(kernel_, 2, arg1.size, arg1.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [3]",
                  clSetKernelArg(kernel_, 3, arg2.size, arg2.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [4]",
                  clSetKernelArg(kernel_, 4, arg3.size, arg3.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [5]",
                  clSetKernelArg(kernel_, 5, arg4.size, arg4.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [6]",
                  clSetKernelArg(kernel_, 6, arg5.size, arg5.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [7]",
                  clSetKernelArg(kernel_, 7, arg6.size, arg6.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [8]",
                  clSetKernelArg(kernel_, 8, arg7.size, arg7.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [9]",
                  clSetKernelArg(kernel_, 9, arg8.size, arg8.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [10]",
                  clSetKernelArg(kernel_, 10, arg9.size, arg9.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [11]",
                  clSetKernelArg(kernel_, 11, arg10.size, arg10.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [12]",
                  clSetKernelArg(kernel_, 12, arg11.size, arg11.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [13]",
                  clSetKernelArg(kernel_, 13, arg12.size, arg12.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [14]",
                  clSetKernelArg(kernel_, 14, arg13.size, arg13.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [15]",
                  clSetKernelArg(kernel_, 15, arg14.size, arg14.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [16]",
                  clSetKernelArg(kernel_, 16, arg15.size, arg15.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [17]",
                  clSetKernelArg(kernel_, 17, arg16.size, arg16.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [18]",
                  clSetKernelArg(kernel_, 18, arg17.size, arg17.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [19]",
                  clSetKernelArg(kernel_, 19, arg18.size, arg18.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [20]",
                  clSetKernelArg(kernel_, 20, arg19.size, arg19.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [21]",
                  clSetKernelArg(kernel_, 21, arg20.size, arg20.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [22]",
                  clSetKernelArg(kernel_, 22, arg21.size, arg21.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [23]",
                  clSetKernelArg(kernel_, 23, arg22.size, arg22.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [24]",
                  clSetKernelArg(kernel_, 24, arg23.size, arg23.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [25]",
                  clSetKernelArg(kernel_, 25, arg24.size, arg24.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [26]",
                  clSetKernelArg(kernel_, 26, arg25.size, arg25.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [27]",
                  clSetKernelArg(kernel_, 27, arg26.size, arg26.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [28]",
                  clSetKernelArg(kernel_, 28, arg27.size, arg27.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [29]",
                  clSetKernelArg(kernel_, 29, arg28.size, arg28.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [30]",
                  clSetKernelArg(kernel_, 30, arg29.size, arg29.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [31]",
                  clSetKernelArg(kernel_, 31, arg30.size, arg30.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [32]",
                  clSetKernelArg(kernel_, 32, arg31.size, arg31.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [33]",
                  clSetKernelArg(kernel_, 33, arg32.size, arg32.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [34]",
                  clSetKernelArg(kernel_, 34, arg33.size, arg33.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [35]",
                  clSetKernelArg(kernel_, 35, arg34.size, arg34.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [36]",
                  clSetKernelArg(kernel_, 36, arg35.size, arg35.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [37]",
                  clSetKernelArg(kernel_, 37, arg36.size, arg36.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [38]",
                  clSetKernelArg(kernel_, 38, arg37.size, arg37.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [39]",
                  clSetKernelArg(kernel_, 39, arg38.size, arg38.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [40]",
                  clSetKernelArg(kernel_, 40, arg39.size, arg39.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [41]",
                  clSetKernelArg(kernel_, 41, arg40.size, arg40.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [42]",
                  clSetKernelArg(kernel_, 42, arg41.size, arg41.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [43]",
                  clSetKernelArg(kernel_, 43, arg42.size, arg42.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [44]",
                  clSetKernelArg(kernel_, 44, arg43.size, arg43.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [45]",
                  clSetKernelArg(kernel_, 45, arg44.size, arg44.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [46]",
                  clSetKernelArg(kernel_, 46, arg45.size, arg45.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [47]",
                  clSetKernelArg(kernel_, 47, arg46.size, arg46.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [48]",
                  clSetKernelArg(kernel_, 48, arg47.size, arg47.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [49]",
                  clSetKernelArg(kernel_, 49, arg48.size, arg48.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Setting Kernel Argument [50]",
                  clSetKernelArg(kernel_, 50, arg49.size, arg49.data()));

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dev->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }