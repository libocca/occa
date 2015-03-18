  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[2];

    const kernelArg *kArgs[1] = {&arg0};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 1; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[4];

    const kernelArg *kArgs[2] = {&arg0, &arg1};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 2; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[6];

    const kernelArg *kArgs[3] = {&arg0, &arg1, &arg2};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 3; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[8];

    const kernelArg *kArgs[4] = {&arg0, &arg1, &arg2, &arg3};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 4; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[10];

    const kernelArg *kArgs[5] = {&arg0, &arg1, &arg2, &arg3, &arg4};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 5; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[12];

    const kernelArg *kArgs[6] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 6; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[14];

    const kernelArg *kArgs[7] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5, &arg6};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 7; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[16];

    const kernelArg *kArgs[8] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5, &arg6, &arg7};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 8; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[18];

    const kernelArg *kArgs[9] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5, &arg6, &arg7, &arg8};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 9; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[20];

    const kernelArg *kArgs[10] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 10; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[22];

    const kernelArg *kArgs[11] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 11; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[24];

    const kernelArg *kArgs[12] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 12; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[26];

    const kernelArg *kArgs[13] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 13; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[28];

    const kernelArg *kArgs[14] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 14; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[30];

    const kernelArg *kArgs[15] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 15; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[32];

    const kernelArg *kArgs[16] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 16; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[34];

    const kernelArg *kArgs[17] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 17; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[36];

    const kernelArg *kArgs[18] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 18; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[38];

    const kernelArg *kArgs[19] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 19; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[40];

    const kernelArg *kArgs[20] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 20; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[42];

    const kernelArg *kArgs[21] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 21; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[44];

    const kernelArg *kArgs[22] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 22; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[46];

    const kernelArg *kArgs[23] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 23; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[48];

    const kernelArg *kArgs[24] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 24; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[50];

    const kernelArg *kArgs[25] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 25; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[52];

    const kernelArg *kArgs[26] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 26; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[54];

    const kernelArg *kArgs[27] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 27; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[56];

    const kernelArg *kArgs[28] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 28; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[58];

    const kernelArg *kArgs[29] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 29; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[60];

    const kernelArg *kArgs[30] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 30; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[62];

    const kernelArg *kArgs[31] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 31; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[64];

    const kernelArg *kArgs[32] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 32; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[66];

    const kernelArg *kArgs[33] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 33; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[68];

    const kernelArg *kArgs[34] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 34; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[70];

    const kernelArg *kArgs[35] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 35; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[72];

    const kernelArg *kArgs[36] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 36; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[74];

    const kernelArg *kArgs[37] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 37; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[76];

    const kernelArg *kArgs[38] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 38; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[78];

    const kernelArg *kArgs[39] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 39; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[80];

    const kernelArg *kArgs[40] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 40; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[82];

    const kernelArg *kArgs[41] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 41; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[84];

    const kernelArg *kArgs[42] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 42; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[86];

    const kernelArg *kArgs[43] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 43; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[88];

    const kernelArg *kArgs[44] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 44; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[90];

    const kernelArg *kArgs[45] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44};

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 45; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[92];

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

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 46; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[94];

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

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 47; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[96];

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

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 48; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[98];

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

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 49; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[100];

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

    args[argCount++] = &occaKernelInfoArgs;

    for(int i = 0; i < 50; ++i){
      if(kArgs[i]->pointer){
        if(kArgs[i]->hasTwoArgs)
          args[argCount++] = (void*) &(((CUDATextureData_t*) kArgs[i]->arg.void_)->surface);
        else
          args[argCount++] = kArgs[i]->arg.void_;
      }
      else {
        args[argCount++] = (void*) &kArgs[i]->arg;
      }

      if(kArgs[i]->hasTwoArgs)
        args[argCount++] = kArgs[i]->arg2.void_;
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   args, 0));
  }