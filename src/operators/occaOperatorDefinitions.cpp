  void kernel::operator() (const kernelArg &arg0){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0); */
        (*(kHandle->nestedKernels[k]))(arg0);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;

    (*this)[*launchDevice](arg0);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;

    (*this)[*launchDevice](arg0,  arg1);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20){
    if(kHandle->nestedKernelCount == 1){
      (*kHandle)(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20);
    }
    else{
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20);
      }
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21){
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21,  const kernelArg &arg22){
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21,  const kernelArg &arg22){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23){
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23);
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23,
                      const kernelArg &arg24){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24);
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24,  arg25); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23,
                      const kernelArg &arg24,  const kernelArg &arg25){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24,  arg25);
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24,  arg25,  arg26); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23,
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24,  arg25,  arg26);
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24,  arg25,  arg26,
                      arg27); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23,
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26,
                      const kernelArg &arg27){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24,  arg25,  arg26,
                      arg27,  arg28); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23,
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26,
                      const kernelArg &arg27,  const kernelArg &arg28){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24,  arg25,  arg26,
                      arg27,  arg28,  arg29); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5,
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8,
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11,
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14,
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17,
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20,
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23,
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26,
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29){
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24,  arg25,  arg26,
                      arg27,  arg28,  arg29,
                      arg30); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24,  arg25,  arg26,
                      arg27,  arg28,  arg29,
                      arg30,  arg31); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
                      arg3,  arg4,  arg5,
                      arg6,  arg7,  arg8,
                      arg9,  arg10,  arg11,
                      arg12,  arg13,  arg14,
                      arg15,  arg16,  arg17,
                      arg18,  arg19,  arg20,
                      arg21,  arg22,  arg23,
                      arg24,  arg25,  arg26,
                      arg27,  arg28,  arg29,
                      arg30,  arg31,  arg32); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg33); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg33,  arg34); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg33,  arg34,  arg35); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg36); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg36,  arg37); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg36,  arg37,  arg38); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg39); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg39,  arg40); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;
    else if(arg40.dev) launchDevice = arg40.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg39,  arg40,  arg41); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;
    else if(arg40.dev) launchDevice = arg40.dev;
    else if(arg41.dev) launchDevice = arg41.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg42); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;
    else if(arg40.dev) launchDevice = arg40.dev;
    else if(arg41.dev) launchDevice = arg41.dev;
    else if(arg42.dev) launchDevice = arg42.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg42,  arg43); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;
    else if(arg40.dev) launchDevice = arg40.dev;
    else if(arg41.dev) launchDevice = arg41.dev;
    else if(arg42.dev) launchDevice = arg42.dev;
    else if(arg43.dev) launchDevice = arg43.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg42,  arg43,  arg44); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;
    else if(arg40.dev) launchDevice = arg40.dev;
    else if(arg41.dev) launchDevice = arg41.dev;
    else if(arg42.dev) launchDevice = arg42.dev;
    else if(arg43.dev) launchDevice = arg43.dev;
    else if(arg44.dev) launchDevice = arg44.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg45); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;
    else if(arg40.dev) launchDevice = arg40.dev;
    else if(arg41.dev) launchDevice = arg41.dev;
    else if(arg42.dev) launchDevice = arg42.dev;
    else if(arg43.dev) launchDevice = arg43.dev;
    else if(arg44.dev) launchDevice = arg44.dev;
    else if(arg45.dev) launchDevice = arg45.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg45,  arg46); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;
    else if(arg40.dev) launchDevice = arg40.dev;
    else if(arg41.dev) launchDevice = arg41.dev;
    else if(arg42.dev) launchDevice = arg42.dev;
    else if(arg43.dev) launchDevice = arg43.dev;
    else if(arg44.dev) launchDevice = arg44.dev;
    else if(arg45.dev) launchDevice = arg45.dev;
    else if(arg46.dev) launchDevice = arg46.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg45,  arg46,  arg47); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;
    else if(arg40.dev) launchDevice = arg40.dev;
    else if(arg41.dev) launchDevice = arg41.dev;
    else if(arg42.dev) launchDevice = arg42.dev;
    else if(arg43.dev) launchDevice = arg43.dev;
    else if(arg44.dev) launchDevice = arg44.dev;
    else if(arg45.dev) launchDevice = arg45.dev;
    else if(arg46.dev) launchDevice = arg46.dev;
    else if(arg47.dev) launchDevice = arg47.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg48); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;
    else if(arg40.dev) launchDevice = arg40.dev;
    else if(arg41.dev) launchDevice = arg41.dev;
    else if(arg42.dev) launchDevice = arg42.dev;
    else if(arg43.dev) launchDevice = arg43.dev;
    else if(arg44.dev) launchDevice = arg44.dev;
    else if(arg45.dev) launchDevice = arg45.dev;
    else if(arg46.dev) launchDevice = arg46.dev;
    else if(arg47.dev) launchDevice = arg47.dev;
    else if(arg48.dev) launchDevice = arg48.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
    if(kHandle->nestedKernelCount == 1){
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
      for(int k = 0; k < kHandle->nestedKernelCount; ++k){
        /* (*(kHandle->setDimsKernels[k]))(arg0,  arg1,  arg2,
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
                      arg48,  arg49); */
        (*(kHandle->nestedKernels[k]))(arg0,  arg1,  arg2,
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2,
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
    occa::device *launchDevice;

    if(arg0.dev) launchDevice = arg0.dev;
    else if(arg1.dev) launchDevice = arg1.dev;
    else if(arg2.dev) launchDevice = arg2.dev;
    else if(arg3.dev) launchDevice = arg3.dev;
    else if(arg4.dev) launchDevice = arg4.dev;
    else if(arg5.dev) launchDevice = arg5.dev;
    else if(arg6.dev) launchDevice = arg6.dev;
    else if(arg7.dev) launchDevice = arg7.dev;
    else if(arg8.dev) launchDevice = arg8.dev;
    else if(arg9.dev) launchDevice = arg9.dev;
    else if(arg10.dev) launchDevice = arg10.dev;
    else if(arg11.dev) launchDevice = arg11.dev;
    else if(arg12.dev) launchDevice = arg12.dev;
    else if(arg13.dev) launchDevice = arg13.dev;
    else if(arg14.dev) launchDevice = arg14.dev;
    else if(arg15.dev) launchDevice = arg15.dev;
    else if(arg16.dev) launchDevice = arg16.dev;
    else if(arg17.dev) launchDevice = arg17.dev;
    else if(arg18.dev) launchDevice = arg18.dev;
    else if(arg19.dev) launchDevice = arg19.dev;
    else if(arg20.dev) launchDevice = arg20.dev;
    else if(arg21.dev) launchDevice = arg21.dev;
    else if(arg22.dev) launchDevice = arg22.dev;
    else if(arg23.dev) launchDevice = arg23.dev;
    else if(arg24.dev) launchDevice = arg24.dev;
    else if(arg25.dev) launchDevice = arg25.dev;
    else if(arg26.dev) launchDevice = arg26.dev;
    else if(arg27.dev) launchDevice = arg27.dev;
    else if(arg28.dev) launchDevice = arg28.dev;
    else if(arg29.dev) launchDevice = arg29.dev;
    else if(arg30.dev) launchDevice = arg30.dev;
    else if(arg31.dev) launchDevice = arg31.dev;
    else if(arg32.dev) launchDevice = arg32.dev;
    else if(arg33.dev) launchDevice = arg33.dev;
    else if(arg34.dev) launchDevice = arg34.dev;
    else if(arg35.dev) launchDevice = arg35.dev;
    else if(arg36.dev) launchDevice = arg36.dev;
    else if(arg37.dev) launchDevice = arg37.dev;
    else if(arg38.dev) launchDevice = arg38.dev;
    else if(arg39.dev) launchDevice = arg39.dev;
    else if(arg40.dev) launchDevice = arg40.dev;
    else if(arg41.dev) launchDevice = arg41.dev;
    else if(arg42.dev) launchDevice = arg42.dev;
    else if(arg43.dev) launchDevice = arg43.dev;
    else if(arg44.dev) launchDevice = arg44.dev;
    else if(arg45.dev) launchDevice = arg45.dev;
    else if(arg46.dev) launchDevice = arg46.dev;
    else if(arg47.dev) launchDevice = arg47.dev;
    else if(arg48.dev) launchDevice = arg48.dev;
    else if(arg49.dev) launchDevice = arg49.dev;

    (*this)[*launchDevice](arg0,  arg1,  arg2,
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
