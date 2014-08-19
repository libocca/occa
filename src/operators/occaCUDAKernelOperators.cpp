  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;

    void *args[2] = {&occaKernelInfoArgs,
                        arg0.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;

    void *args[3] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;

    void *args[4] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;

    void *args[5] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;

    void *args[6] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;

    void *args[7] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;

    void *args[8] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;

    void *args[9] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
                   args, 0);
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;

    void *args[10] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[11] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[12] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[13] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[14] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[15] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[16] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[17] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[18] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[19] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[20] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[21] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[22] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[23] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[24] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[25] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[26] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[27] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[28] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[29] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[30] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[31] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[32] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[33] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[34] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[35] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[36] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[37] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[38] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[39] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[40] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[41] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[42] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data(),
                        arg40.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[43] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data(),
                        arg40.data(),
                        arg41.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[44] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data(),
                        arg40.data(),
                        arg41.data(),
                        arg42.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[45] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data(),
                        arg40.data(),
                        arg41.data(),
                        arg42.data(),
                        arg43.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[46] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data(),
                        arg40.data(),
                        arg41.data(),
                        arg42.data(),
                        arg43.data(),
                        arg44.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[47] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data(),
                        arg40.data(),
                        arg41.data(),
                        arg42.data(),
                        arg43.data(),
                        arg44.data(),
                        arg45.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[48] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data(),
                        arg40.data(),
                        arg41.data(),
                        arg42.data(),
                        arg43.data(),
                        arg44.data(),
                        arg45.data(),
                        arg46.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[49] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data(),
                        arg40.data(),
                        arg41.data(),
                        arg42.data(),
                        arg43.data(),
                        arg44.data(),
                        arg45.data(),
                        arg46.data(),
                        arg47.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[50] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data(),
                        arg40.data(),
                        arg41.data(),
                        arg42.data(),
                        arg43.data(),
                        arg44.data(),
                        arg45.data(),
                        arg46.data(),
                        arg47.data(),
                        arg48.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
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

    void *args[51] = {&occaKernelInfoArgs,
                        arg0.data(),
                        arg1.data(),
                        arg2.data(),
                        arg3.data(),
                        arg4.data(),
                        arg5.data(),
                        arg6.data(),
                        arg7.data(),
                        arg8.data(),
                        arg9.data(),
                        arg10.data(),
                        arg11.data(),
                        arg12.data(),
                        arg13.data(),
                        arg14.data(),
                        arg15.data(),
                        arg16.data(),
                        arg17.data(),
                        arg18.data(),
                        arg19.data(),
                        arg20.data(),
                        arg21.data(),
                        arg22.data(),
                        arg23.data(),
                        arg24.data(),
                        arg25.data(),
                        arg26.data(),
                        arg27.data(),
                        arg28.data(),
                        arg29.data(),
                        arg30.data(),
                        arg31.data(),
                        arg32.data(),
                        arg33.data(),
                        arg34.data(),
                        arg35.data(),
                        arg36.data(),
                        arg37.data(),
                        arg38.data(),
                        arg39.data(),
                        arg40.data(),
                        arg41.data(),
                        arg42.data(),
                        arg43.data(),
                        arg44.data(),
                        arg45.data(),
                        arg46.data(),
                        arg47.data(),
                        arg48.data(),
                        arg49.data()};

    cuLaunchKernel(function_,
                   outer.x, outer.y, outer.z,
                   inner.x, inner.y, inner.z,
                   0, *((CUstream*) dev->currentStream),
                   args, 0);
  }