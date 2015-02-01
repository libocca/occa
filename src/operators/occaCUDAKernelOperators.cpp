  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[2];

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[4];

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[6];

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[8];

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[10];

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argCount = 0;

    void *args[12];

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;
    args[argCount++] = arg40.pointer ? (arg40.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg40.arg.void_)->surface) : arg40.arg.void_) : (void*) &arg40.arg;
    if(arg40.hasTwoArgs)
      args[argCount++] = arg40.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;
    args[argCount++] = arg40.pointer ? (arg40.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg40.arg.void_)->surface) : arg40.arg.void_) : (void*) &arg40.arg;
    if(arg40.hasTwoArgs)
      args[argCount++] = arg40.arg2.void_;
    args[argCount++] = arg41.pointer ? (arg41.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg41.arg.void_)->surface) : arg41.arg.void_) : (void*) &arg41.arg;
    if(arg41.hasTwoArgs)
      args[argCount++] = arg41.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;
    args[argCount++] = arg40.pointer ? (arg40.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg40.arg.void_)->surface) : arg40.arg.void_) : (void*) &arg40.arg;
    if(arg40.hasTwoArgs)
      args[argCount++] = arg40.arg2.void_;
    args[argCount++] = arg41.pointer ? (arg41.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg41.arg.void_)->surface) : arg41.arg.void_) : (void*) &arg41.arg;
    if(arg41.hasTwoArgs)
      args[argCount++] = arg41.arg2.void_;
    args[argCount++] = arg42.pointer ? (arg42.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg42.arg.void_)->surface) : arg42.arg.void_) : (void*) &arg42.arg;
    if(arg42.hasTwoArgs)
      args[argCount++] = arg42.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;
    args[argCount++] = arg40.pointer ? (arg40.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg40.arg.void_)->surface) : arg40.arg.void_) : (void*) &arg40.arg;
    if(arg40.hasTwoArgs)
      args[argCount++] = arg40.arg2.void_;
    args[argCount++] = arg41.pointer ? (arg41.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg41.arg.void_)->surface) : arg41.arg.void_) : (void*) &arg41.arg;
    if(arg41.hasTwoArgs)
      args[argCount++] = arg41.arg2.void_;
    args[argCount++] = arg42.pointer ? (arg42.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg42.arg.void_)->surface) : arg42.arg.void_) : (void*) &arg42.arg;
    if(arg42.hasTwoArgs)
      args[argCount++] = arg42.arg2.void_;
    args[argCount++] = arg43.pointer ? (arg43.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg43.arg.void_)->surface) : arg43.arg.void_) : (void*) &arg43.arg;
    if(arg43.hasTwoArgs)
      args[argCount++] = arg43.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;
    args[argCount++] = arg40.pointer ? (arg40.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg40.arg.void_)->surface) : arg40.arg.void_) : (void*) &arg40.arg;
    if(arg40.hasTwoArgs)
      args[argCount++] = arg40.arg2.void_;
    args[argCount++] = arg41.pointer ? (arg41.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg41.arg.void_)->surface) : arg41.arg.void_) : (void*) &arg41.arg;
    if(arg41.hasTwoArgs)
      args[argCount++] = arg41.arg2.void_;
    args[argCount++] = arg42.pointer ? (arg42.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg42.arg.void_)->surface) : arg42.arg.void_) : (void*) &arg42.arg;
    if(arg42.hasTwoArgs)
      args[argCount++] = arg42.arg2.void_;
    args[argCount++] = arg43.pointer ? (arg43.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg43.arg.void_)->surface) : arg43.arg.void_) : (void*) &arg43.arg;
    if(arg43.hasTwoArgs)
      args[argCount++] = arg43.arg2.void_;
    args[argCount++] = arg44.pointer ? (arg44.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg44.arg.void_)->surface) : arg44.arg.void_) : (void*) &arg44.arg;
    if(arg44.hasTwoArgs)
      args[argCount++] = arg44.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;
    args[argCount++] = arg40.pointer ? (arg40.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg40.arg.void_)->surface) : arg40.arg.void_) : (void*) &arg40.arg;
    if(arg40.hasTwoArgs)
      args[argCount++] = arg40.arg2.void_;
    args[argCount++] = arg41.pointer ? (arg41.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg41.arg.void_)->surface) : arg41.arg.void_) : (void*) &arg41.arg;
    if(arg41.hasTwoArgs)
      args[argCount++] = arg41.arg2.void_;
    args[argCount++] = arg42.pointer ? (arg42.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg42.arg.void_)->surface) : arg42.arg.void_) : (void*) &arg42.arg;
    if(arg42.hasTwoArgs)
      args[argCount++] = arg42.arg2.void_;
    args[argCount++] = arg43.pointer ? (arg43.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg43.arg.void_)->surface) : arg43.arg.void_) : (void*) &arg43.arg;
    if(arg43.hasTwoArgs)
      args[argCount++] = arg43.arg2.void_;
    args[argCount++] = arg44.pointer ? (arg44.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg44.arg.void_)->surface) : arg44.arg.void_) : (void*) &arg44.arg;
    if(arg44.hasTwoArgs)
      args[argCount++] = arg44.arg2.void_;
    args[argCount++] = arg45.pointer ? (arg45.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg45.arg.void_)->surface) : arg45.arg.void_) : (void*) &arg45.arg;
    if(arg45.hasTwoArgs)
      args[argCount++] = arg45.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;
    args[argCount++] = arg40.pointer ? (arg40.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg40.arg.void_)->surface) : arg40.arg.void_) : (void*) &arg40.arg;
    if(arg40.hasTwoArgs)
      args[argCount++] = arg40.arg2.void_;
    args[argCount++] = arg41.pointer ? (arg41.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg41.arg.void_)->surface) : arg41.arg.void_) : (void*) &arg41.arg;
    if(arg41.hasTwoArgs)
      args[argCount++] = arg41.arg2.void_;
    args[argCount++] = arg42.pointer ? (arg42.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg42.arg.void_)->surface) : arg42.arg.void_) : (void*) &arg42.arg;
    if(arg42.hasTwoArgs)
      args[argCount++] = arg42.arg2.void_;
    args[argCount++] = arg43.pointer ? (arg43.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg43.arg.void_)->surface) : arg43.arg.void_) : (void*) &arg43.arg;
    if(arg43.hasTwoArgs)
      args[argCount++] = arg43.arg2.void_;
    args[argCount++] = arg44.pointer ? (arg44.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg44.arg.void_)->surface) : arg44.arg.void_) : (void*) &arg44.arg;
    if(arg44.hasTwoArgs)
      args[argCount++] = arg44.arg2.void_;
    args[argCount++] = arg45.pointer ? (arg45.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg45.arg.void_)->surface) : arg45.arg.void_) : (void*) &arg45.arg;
    if(arg45.hasTwoArgs)
      args[argCount++] = arg45.arg2.void_;
    args[argCount++] = arg46.pointer ? (arg46.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg46.arg.void_)->surface) : arg46.arg.void_) : (void*) &arg46.arg;
    if(arg46.hasTwoArgs)
      args[argCount++] = arg46.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;
    args[argCount++] = arg40.pointer ? (arg40.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg40.arg.void_)->surface) : arg40.arg.void_) : (void*) &arg40.arg;
    if(arg40.hasTwoArgs)
      args[argCount++] = arg40.arg2.void_;
    args[argCount++] = arg41.pointer ? (arg41.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg41.arg.void_)->surface) : arg41.arg.void_) : (void*) &arg41.arg;
    if(arg41.hasTwoArgs)
      args[argCount++] = arg41.arg2.void_;
    args[argCount++] = arg42.pointer ? (arg42.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg42.arg.void_)->surface) : arg42.arg.void_) : (void*) &arg42.arg;
    if(arg42.hasTwoArgs)
      args[argCount++] = arg42.arg2.void_;
    args[argCount++] = arg43.pointer ? (arg43.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg43.arg.void_)->surface) : arg43.arg.void_) : (void*) &arg43.arg;
    if(arg43.hasTwoArgs)
      args[argCount++] = arg43.arg2.void_;
    args[argCount++] = arg44.pointer ? (arg44.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg44.arg.void_)->surface) : arg44.arg.void_) : (void*) &arg44.arg;
    if(arg44.hasTwoArgs)
      args[argCount++] = arg44.arg2.void_;
    args[argCount++] = arg45.pointer ? (arg45.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg45.arg.void_)->surface) : arg45.arg.void_) : (void*) &arg45.arg;
    if(arg45.hasTwoArgs)
      args[argCount++] = arg45.arg2.void_;
    args[argCount++] = arg46.pointer ? (arg46.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg46.arg.void_)->surface) : arg46.arg.void_) : (void*) &arg46.arg;
    if(arg46.hasTwoArgs)
      args[argCount++] = arg46.arg2.void_;
    args[argCount++] = arg47.pointer ? (arg47.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg47.arg.void_)->surface) : arg47.arg.void_) : (void*) &arg47.arg;
    if(arg47.hasTwoArgs)
      args[argCount++] = arg47.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;
    args[argCount++] = arg40.pointer ? (arg40.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg40.arg.void_)->surface) : arg40.arg.void_) : (void*) &arg40.arg;
    if(arg40.hasTwoArgs)
      args[argCount++] = arg40.arg2.void_;
    args[argCount++] = arg41.pointer ? (arg41.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg41.arg.void_)->surface) : arg41.arg.void_) : (void*) &arg41.arg;
    if(arg41.hasTwoArgs)
      args[argCount++] = arg41.arg2.void_;
    args[argCount++] = arg42.pointer ? (arg42.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg42.arg.void_)->surface) : arg42.arg.void_) : (void*) &arg42.arg;
    if(arg42.hasTwoArgs)
      args[argCount++] = arg42.arg2.void_;
    args[argCount++] = arg43.pointer ? (arg43.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg43.arg.void_)->surface) : arg43.arg.void_) : (void*) &arg43.arg;
    if(arg43.hasTwoArgs)
      args[argCount++] = arg43.arg2.void_;
    args[argCount++] = arg44.pointer ? (arg44.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg44.arg.void_)->surface) : arg44.arg.void_) : (void*) &arg44.arg;
    if(arg44.hasTwoArgs)
      args[argCount++] = arg44.arg2.void_;
    args[argCount++] = arg45.pointer ? (arg45.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg45.arg.void_)->surface) : arg45.arg.void_) : (void*) &arg45.arg;
    if(arg45.hasTwoArgs)
      args[argCount++] = arg45.arg2.void_;
    args[argCount++] = arg46.pointer ? (arg46.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg46.arg.void_)->surface) : arg46.arg.void_) : (void*) &arg46.arg;
    if(arg46.hasTwoArgs)
      args[argCount++] = arg46.arg2.void_;
    args[argCount++] = arg47.pointer ? (arg47.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg47.arg.void_)->surface) : arg47.arg.void_) : (void*) &arg47.arg;
    if(arg47.hasTwoArgs)
      args[argCount++] = arg47.arg2.void_;
    args[argCount++] = arg48.pointer ? (arg48.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg48.arg.void_)->surface) : arg48.arg.void_) : (void*) &arg48.arg;
    if(arg48.hasTwoArgs)
      args[argCount++] = arg48.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
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

    args[argCount++] = &occaKernelInfoArgs;

    args[argCount++] = arg0.pointer ? (arg0.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg0.arg.void_)->surface) : arg0.arg.void_) : (void*) &arg0.arg;
    if(arg0.hasTwoArgs)
      args[argCount++] = arg0.arg2.void_;
    args[argCount++] = arg1.pointer ? (arg1.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg1.arg.void_)->surface) : arg1.arg.void_) : (void*) &arg1.arg;
    if(arg1.hasTwoArgs)
      args[argCount++] = arg1.arg2.void_;
    args[argCount++] = arg2.pointer ? (arg2.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg2.arg.void_)->surface) : arg2.arg.void_) : (void*) &arg2.arg;
    if(arg2.hasTwoArgs)
      args[argCount++] = arg2.arg2.void_;
    args[argCount++] = arg3.pointer ? (arg3.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg3.arg.void_)->surface) : arg3.arg.void_) : (void*) &arg3.arg;
    if(arg3.hasTwoArgs)
      args[argCount++] = arg3.arg2.void_;
    args[argCount++] = arg4.pointer ? (arg4.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg4.arg.void_)->surface) : arg4.arg.void_) : (void*) &arg4.arg;
    if(arg4.hasTwoArgs)
      args[argCount++] = arg4.arg2.void_;
    args[argCount++] = arg5.pointer ? (arg5.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg5.arg.void_)->surface) : arg5.arg.void_) : (void*) &arg5.arg;
    if(arg5.hasTwoArgs)
      args[argCount++] = arg5.arg2.void_;
    args[argCount++] = arg6.pointer ? (arg6.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg6.arg.void_)->surface) : arg6.arg.void_) : (void*) &arg6.arg;
    if(arg6.hasTwoArgs)
      args[argCount++] = arg6.arg2.void_;
    args[argCount++] = arg7.pointer ? (arg7.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg7.arg.void_)->surface) : arg7.arg.void_) : (void*) &arg7.arg;
    if(arg7.hasTwoArgs)
      args[argCount++] = arg7.arg2.void_;
    args[argCount++] = arg8.pointer ? (arg8.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg8.arg.void_)->surface) : arg8.arg.void_) : (void*) &arg8.arg;
    if(arg8.hasTwoArgs)
      args[argCount++] = arg8.arg2.void_;
    args[argCount++] = arg9.pointer ? (arg9.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg9.arg.void_)->surface) : arg9.arg.void_) : (void*) &arg9.arg;
    if(arg9.hasTwoArgs)
      args[argCount++] = arg9.arg2.void_;
    args[argCount++] = arg10.pointer ? (arg10.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg10.arg.void_)->surface) : arg10.arg.void_) : (void*) &arg10.arg;
    if(arg10.hasTwoArgs)
      args[argCount++] = arg10.arg2.void_;
    args[argCount++] = arg11.pointer ? (arg11.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg11.arg.void_)->surface) : arg11.arg.void_) : (void*) &arg11.arg;
    if(arg11.hasTwoArgs)
      args[argCount++] = arg11.arg2.void_;
    args[argCount++] = arg12.pointer ? (arg12.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg12.arg.void_)->surface) : arg12.arg.void_) : (void*) &arg12.arg;
    if(arg12.hasTwoArgs)
      args[argCount++] = arg12.arg2.void_;
    args[argCount++] = arg13.pointer ? (arg13.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg13.arg.void_)->surface) : arg13.arg.void_) : (void*) &arg13.arg;
    if(arg13.hasTwoArgs)
      args[argCount++] = arg13.arg2.void_;
    args[argCount++] = arg14.pointer ? (arg14.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg14.arg.void_)->surface) : arg14.arg.void_) : (void*) &arg14.arg;
    if(arg14.hasTwoArgs)
      args[argCount++] = arg14.arg2.void_;
    args[argCount++] = arg15.pointer ? (arg15.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg15.arg.void_)->surface) : arg15.arg.void_) : (void*) &arg15.arg;
    if(arg15.hasTwoArgs)
      args[argCount++] = arg15.arg2.void_;
    args[argCount++] = arg16.pointer ? (arg16.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg16.arg.void_)->surface) : arg16.arg.void_) : (void*) &arg16.arg;
    if(arg16.hasTwoArgs)
      args[argCount++] = arg16.arg2.void_;
    args[argCount++] = arg17.pointer ? (arg17.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg17.arg.void_)->surface) : arg17.arg.void_) : (void*) &arg17.arg;
    if(arg17.hasTwoArgs)
      args[argCount++] = arg17.arg2.void_;
    args[argCount++] = arg18.pointer ? (arg18.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg18.arg.void_)->surface) : arg18.arg.void_) : (void*) &arg18.arg;
    if(arg18.hasTwoArgs)
      args[argCount++] = arg18.arg2.void_;
    args[argCount++] = arg19.pointer ? (arg19.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg19.arg.void_)->surface) : arg19.arg.void_) : (void*) &arg19.arg;
    if(arg19.hasTwoArgs)
      args[argCount++] = arg19.arg2.void_;
    args[argCount++] = arg20.pointer ? (arg20.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg20.arg.void_)->surface) : arg20.arg.void_) : (void*) &arg20.arg;
    if(arg20.hasTwoArgs)
      args[argCount++] = arg20.arg2.void_;
    args[argCount++] = arg21.pointer ? (arg21.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg21.arg.void_)->surface) : arg21.arg.void_) : (void*) &arg21.arg;
    if(arg21.hasTwoArgs)
      args[argCount++] = arg21.arg2.void_;
    args[argCount++] = arg22.pointer ? (arg22.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg22.arg.void_)->surface) : arg22.arg.void_) : (void*) &arg22.arg;
    if(arg22.hasTwoArgs)
      args[argCount++] = arg22.arg2.void_;
    args[argCount++] = arg23.pointer ? (arg23.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg23.arg.void_)->surface) : arg23.arg.void_) : (void*) &arg23.arg;
    if(arg23.hasTwoArgs)
      args[argCount++] = arg23.arg2.void_;
    args[argCount++] = arg24.pointer ? (arg24.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg24.arg.void_)->surface) : arg24.arg.void_) : (void*) &arg24.arg;
    if(arg24.hasTwoArgs)
      args[argCount++] = arg24.arg2.void_;
    args[argCount++] = arg25.pointer ? (arg25.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg25.arg.void_)->surface) : arg25.arg.void_) : (void*) &arg25.arg;
    if(arg25.hasTwoArgs)
      args[argCount++] = arg25.arg2.void_;
    args[argCount++] = arg26.pointer ? (arg26.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg26.arg.void_)->surface) : arg26.arg.void_) : (void*) &arg26.arg;
    if(arg26.hasTwoArgs)
      args[argCount++] = arg26.arg2.void_;
    args[argCount++] = arg27.pointer ? (arg27.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg27.arg.void_)->surface) : arg27.arg.void_) : (void*) &arg27.arg;
    if(arg27.hasTwoArgs)
      args[argCount++] = arg27.arg2.void_;
    args[argCount++] = arg28.pointer ? (arg28.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg28.arg.void_)->surface) : arg28.arg.void_) : (void*) &arg28.arg;
    if(arg28.hasTwoArgs)
      args[argCount++] = arg28.arg2.void_;
    args[argCount++] = arg29.pointer ? (arg29.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg29.arg.void_)->surface) : arg29.arg.void_) : (void*) &arg29.arg;
    if(arg29.hasTwoArgs)
      args[argCount++] = arg29.arg2.void_;
    args[argCount++] = arg30.pointer ? (arg30.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg30.arg.void_)->surface) : arg30.arg.void_) : (void*) &arg30.arg;
    if(arg30.hasTwoArgs)
      args[argCount++] = arg30.arg2.void_;
    args[argCount++] = arg31.pointer ? (arg31.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg31.arg.void_)->surface) : arg31.arg.void_) : (void*) &arg31.arg;
    if(arg31.hasTwoArgs)
      args[argCount++] = arg31.arg2.void_;
    args[argCount++] = arg32.pointer ? (arg32.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg32.arg.void_)->surface) : arg32.arg.void_) : (void*) &arg32.arg;
    if(arg32.hasTwoArgs)
      args[argCount++] = arg32.arg2.void_;
    args[argCount++] = arg33.pointer ? (arg33.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg33.arg.void_)->surface) : arg33.arg.void_) : (void*) &arg33.arg;
    if(arg33.hasTwoArgs)
      args[argCount++] = arg33.arg2.void_;
    args[argCount++] = arg34.pointer ? (arg34.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg34.arg.void_)->surface) : arg34.arg.void_) : (void*) &arg34.arg;
    if(arg34.hasTwoArgs)
      args[argCount++] = arg34.arg2.void_;
    args[argCount++] = arg35.pointer ? (arg35.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg35.arg.void_)->surface) : arg35.arg.void_) : (void*) &arg35.arg;
    if(arg35.hasTwoArgs)
      args[argCount++] = arg35.arg2.void_;
    args[argCount++] = arg36.pointer ? (arg36.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg36.arg.void_)->surface) : arg36.arg.void_) : (void*) &arg36.arg;
    if(arg36.hasTwoArgs)
      args[argCount++] = arg36.arg2.void_;
    args[argCount++] = arg37.pointer ? (arg37.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg37.arg.void_)->surface) : arg37.arg.void_) : (void*) &arg37.arg;
    if(arg37.hasTwoArgs)
      args[argCount++] = arg37.arg2.void_;
    args[argCount++] = arg38.pointer ? (arg38.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg38.arg.void_)->surface) : arg38.arg.void_) : (void*) &arg38.arg;
    if(arg38.hasTwoArgs)
      args[argCount++] = arg38.arg2.void_;
    args[argCount++] = arg39.pointer ? (arg39.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg39.arg.void_)->surface) : arg39.arg.void_) : (void*) &arg39.arg;
    if(arg39.hasTwoArgs)
      args[argCount++] = arg39.arg2.void_;
    args[argCount++] = arg40.pointer ? (arg40.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg40.arg.void_)->surface) : arg40.arg.void_) : (void*) &arg40.arg;
    if(arg40.hasTwoArgs)
      args[argCount++] = arg40.arg2.void_;
    args[argCount++] = arg41.pointer ? (arg41.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg41.arg.void_)->surface) : arg41.arg.void_) : (void*) &arg41.arg;
    if(arg41.hasTwoArgs)
      args[argCount++] = arg41.arg2.void_;
    args[argCount++] = arg42.pointer ? (arg42.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg42.arg.void_)->surface) : arg42.arg.void_) : (void*) &arg42.arg;
    if(arg42.hasTwoArgs)
      args[argCount++] = arg42.arg2.void_;
    args[argCount++] = arg43.pointer ? (arg43.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg43.arg.void_)->surface) : arg43.arg.void_) : (void*) &arg43.arg;
    if(arg43.hasTwoArgs)
      args[argCount++] = arg43.arg2.void_;
    args[argCount++] = arg44.pointer ? (arg44.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg44.arg.void_)->surface) : arg44.arg.void_) : (void*) &arg44.arg;
    if(arg44.hasTwoArgs)
      args[argCount++] = arg44.arg2.void_;
    args[argCount++] = arg45.pointer ? (arg45.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg45.arg.void_)->surface) : arg45.arg.void_) : (void*) &arg45.arg;
    if(arg45.hasTwoArgs)
      args[argCount++] = arg45.arg2.void_;
    args[argCount++] = arg46.pointer ? (arg46.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg46.arg.void_)->surface) : arg46.arg.void_) : (void*) &arg46.arg;
    if(arg46.hasTwoArgs)
      args[argCount++] = arg46.arg2.void_;
    args[argCount++] = arg47.pointer ? (arg47.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg47.arg.void_)->surface) : arg47.arg.void_) : (void*) &arg47.arg;
    if(arg47.hasTwoArgs)
      args[argCount++] = arg47.arg2.void_;
    args[argCount++] = arg48.pointer ? (arg48.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg48.arg.void_)->surface) : arg48.arg.void_) : (void*) &arg48.arg;
    if(arg48.hasTwoArgs)
      args[argCount++] = arg48.arg2.void_;
    args[argCount++] = arg49.pointer ? (arg49.hasTwoArgs ? (void*) &(((CUDATextureData_t*) arg49.arg.void_)->surface) : arg49.arg.void_) : (void*) &arg49.arg;
    if(arg49.hasTwoArgs)
      args[argCount++] = arg49.arg2.void_;

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dHandle->currentStream),
                   args, 0);
  }