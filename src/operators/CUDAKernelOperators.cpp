  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[1] = {&arg0};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 1; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[2] = {&arg0, &arg1};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 2; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[3] = {&arg0, &arg1, &arg2};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 3; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[4] = {&arg0, &arg1, &arg2, &arg3};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 4; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[5] = {&arg0, &arg1, &arg2, &arg3, &arg4};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 5; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[6] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 6; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[7] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5, &arg6};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 7; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[8] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5, &arg6, &arg7};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 8; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[9] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                 &arg5, &arg6, &arg7, &arg8};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 9; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[10] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 10; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[11] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 11; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }

  template <>
  void kernel_t<CUDA>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
    CUfunction function_    = data_.function;

    int occaKernelInfoArgs = 0;
    int argc = 0;

    const kernelArg *args[12] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 12; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[13] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 13; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[14] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 14; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[15] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 15; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[16] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 16; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[17] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 17; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[18] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 18; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[19] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 19; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[20] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 20; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[21] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 21; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[22] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 22; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[23] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 23; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[24] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 24; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[25] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 25; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[26] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 26; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[27] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 27; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[28] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 28; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[29] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 29; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[30] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 30; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[31] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 31; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[32] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 32; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[33] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 33; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[34] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 34; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[35] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 35; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[36] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 36; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[37] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 37; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[38] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 38; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[39] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 39; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[40] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 40; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[41] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 41; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[42] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 42; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[43] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 43; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[44] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 44; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[45] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 45; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[46] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44, 
                                  &arg45};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 46; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[47] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44, 
                                  &arg45, &arg46};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 47; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[48] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44, 
                                  &arg45, &arg46, &arg47};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 48; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[49] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44, 
                                  &arg45, &arg46, &arg47, &arg48};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 49; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
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
    int argc = 0;

    const kernelArg *args[50] = {&arg0, &arg1, &arg2, &arg3, &arg4, 
                                  &arg5, &arg6, &arg7, &arg8, &arg9, 
                                  &arg10, &arg11, &arg12, &arg13, &arg14, 
                                  &arg15, &arg16, &arg17, &arg18, &arg19, 
                                  &arg20, &arg21, &arg22, &arg23, &arg24, 
                                  &arg25, &arg26, &arg27, &arg28, &arg29, 
                                  &arg30, &arg31, &arg32, &arg33, &arg34, 
                                  &arg35, &arg36, &arg37, &arg38, &arg39, 
                                  &arg40, &arg41, &arg42, &arg43, &arg44, 
                                  &arg45, &arg46, &arg47, &arg48, &arg49};

    data_.vArgs[argc++] = &occaKernelInfoArgs;

    for(int i = 0; i < 50; ++i){
      for(int j = 0; j < args[i]->argc; ++j){
        data_.vArgs[argc++] = args[i]->args[j].ptr();
      }
    }

    OCCA_CUDA_CHECK("Launching Kernel",
                    cuLaunchKernel(function_,
                                   outer.x, outer.y, outer.z,
                                   inner.x, inner.y, inner.z,
                                   0, *((CUstream*) dHandle->currentStream),
                                   data_.vArgs, 0));
  }
