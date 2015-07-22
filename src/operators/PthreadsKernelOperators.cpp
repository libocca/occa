  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[1] = {arg0};

    pthreads::runFromArguments(data_, dims, inner, outer, 1, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[2] = {arg0,  arg1};

    pthreads::runFromArguments(data_, dims, inner, outer, 2, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[3] = {arg0,  arg1,  arg2};

    pthreads::runFromArguments(data_, dims, inner, outer, 3, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[4] = {arg0,  arg1,  arg2, 
                      arg3};

    pthreads::runFromArguments(data_, dims, inner, outer, 4, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[5] = {arg0,  arg1,  arg2, 
                      arg3,  arg4};

    pthreads::runFromArguments(data_, dims, inner, outer, 5, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[6] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5};

    pthreads::runFromArguments(data_, dims, inner, outer, 6, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[7] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6};

    pthreads::runFromArguments(data_, dims, inner, outer, 7, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[8] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7};

    pthreads::runFromArguments(data_, dims, inner, outer, 8, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[9] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8};

    pthreads::runFromArguments(data_, dims, inner, outer, 9, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[10] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9};

    pthreads::runFromArguments(data_, dims, inner, outer, 10, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[11] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10};

    pthreads::runFromArguments(data_, dims, inner, outer, 11, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[12] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11};

    pthreads::runFromArguments(data_, dims, inner, outer, 12, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[13] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12};

    pthreads::runFromArguments(data_, dims, inner, outer, 13, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[14] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13};

    pthreads::runFromArguments(data_, dims, inner, outer, 14, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[15] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14};

    pthreads::runFromArguments(data_, dims, inner, outer, 15, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[16] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15};

    pthreads::runFromArguments(data_, dims, inner, outer, 16, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[17] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16};

    pthreads::runFromArguments(data_, dims, inner, outer, 17, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[18] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17};

    pthreads::runFromArguments(data_, dims, inner, outer, 18, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[19] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18};

    pthreads::runFromArguments(data_, dims, inner, outer, 19, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[20] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19};

    pthreads::runFromArguments(data_, dims, inner, outer, 20, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[21] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20};

    pthreads::runFromArguments(data_, dims, inner, outer, 21, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[22] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21};

    pthreads::runFromArguments(data_, dims, inner, outer, 22, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[23] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22};

    pthreads::runFromArguments(data_, dims, inner, outer, 23, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[24] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23};

    pthreads::runFromArguments(data_, dims, inner, outer, 24, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[25] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24};

    pthreads::runFromArguments(data_, dims, inner, outer, 25, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[26] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25};

    pthreads::runFromArguments(data_, dims, inner, outer, 26, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[27] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26};

    pthreads::runFromArguments(data_, dims, inner, outer, 27, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[28] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27};

    pthreads::runFromArguments(data_, dims, inner, outer, 28, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[29] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28};

    pthreads::runFromArguments(data_, dims, inner, outer, 29, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[30] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29};

    pthreads::runFromArguments(data_, dims, inner, outer, 30, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[31] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30};

    pthreads::runFromArguments(data_, dims, inner, outer, 31, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[32] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31};

    pthreads::runFromArguments(data_, dims, inner, outer, 32, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[33] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32};

    pthreads::runFromArguments(data_, dims, inner, outer, 33, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[34] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33};

    pthreads::runFromArguments(data_, dims, inner, outer, 34, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[35] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34};

    pthreads::runFromArguments(data_, dims, inner, outer, 35, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[36] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35};

    pthreads::runFromArguments(data_, dims, inner, outer, 36, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[37] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36};

    pthreads::runFromArguments(data_, dims, inner, outer, 37, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[38] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37};

    pthreads::runFromArguments(data_, dims, inner, outer, 38, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[39] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38};

    pthreads::runFromArguments(data_, dims, inner, outer, 39, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[40] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39};

    pthreads::runFromArguments(data_, dims, inner, outer, 40, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[41] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39,  arg40};

    pthreads::runFromArguments(data_, dims, inner, outer, 41, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[42] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39,  arg40,  arg41};

    pthreads::runFromArguments(data_, dims, inner, outer, 42, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[43] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39,  arg40,  arg41, 
                      arg42};

    pthreads::runFromArguments(data_, dims, inner, outer, 43, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[44] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39,  arg40,  arg41, 
                      arg42,  arg43};

    pthreads::runFromArguments(data_, dims, inner, outer, 44, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[45] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39,  arg40,  arg41, 
                      arg42,  arg43,  arg44};

    pthreads::runFromArguments(data_, dims, inner, outer, 45, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[46] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39,  arg40,  arg41, 
                      arg42,  arg43,  arg44, 
                      arg45};

    pthreads::runFromArguments(data_, dims, inner, outer, 46, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[47] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39,  arg40,  arg41, 
                      arg42,  arg43,  arg44, 
                      arg45,  arg46};

    pthreads::runFromArguments(data_, dims, inner, outer, 47, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[48] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39,  arg40,  arg41, 
                      arg42,  arg43,  arg44, 
                      arg45,  arg46,  arg47};

    pthreads::runFromArguments(data_, dims, inner, outer, 48, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[49] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39,  arg40,  arg41, 
                      arg42,  arg43,  arg44, 
                      arg45,  arg46,  arg47, 
                      arg48};

    pthreads::runFromArguments(data_, dims, inner, outer, 49, args);
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);

    kernelArg args[50] = {arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32, 
                      arg33,  arg34,  arg35, 
                      arg36,  arg37,  arg38, 
                      arg39,  arg40,  arg41, 
                      arg42,  arg43,  arg44, 
                      arg45,  arg46,  arg47, 
                      arg48,  arg49};

    pthreads::runFromArguments(data_, dims, inner, outer, 50, args);
  }
