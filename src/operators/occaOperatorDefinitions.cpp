  void kernel::operator() (const kernelArg &arg0){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0);
        (*nestedKernels[k])(arg0);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1);
        (*nestedKernels[k])(arg0,  arg1);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2);
        (*nestedKernels[k])(arg0,  arg1,  arg2);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29){
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23, 
                      arg24,  arg25,  arg26, 
                      arg27,  arg28,  arg29, 
                      arg30,  arg31,  arg32);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg33);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg33);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg33);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg33,  arg34);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg33,  arg34);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg33,  arg34);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg33,  arg34,  arg35);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg33,  arg34,  arg35);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg33,  arg34,  arg35);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg36);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg36);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg36);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg36,  arg37);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg36,  arg37);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg36,  arg37);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg36,  arg37,  arg38);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg36,  arg37,  arg38);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg36,  arg37,  arg38);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg39);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg39);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg39);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg39,  arg40);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg39,  arg40);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg39,  arg40);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg39,  arg40,  arg41);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg39,  arg40,  arg41);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg39,  arg40,  arg41);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg42);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg42);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg42);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg42,  arg43);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg42,  arg43);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg42,  arg43);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg42,  arg43,  arg44);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg42,  arg43,  arg44);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg42,  arg43,  arg44);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg45);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg45);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg45);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg45,  arg46);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg45,  arg46);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg45,  arg46);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg45,  arg46,  arg47);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg45,  arg46,  arg47);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg45,  arg46,  arg47);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg48);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg48);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg48);
      }
    }
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    if(nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2, 
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
                      arg48,  arg49);
    }
    else{
      for(int k = 0; k < nestedKernelCount; ++k){
        (*setDimsKernels[k])(arg0,  arg1,  arg2, 
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
                      arg48,  arg49);
        (*nestedKernels[k])(arg0,  arg1,  arg2, 
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
                      arg48,  arg49);
      }
    }
  }