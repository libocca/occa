  void kernel::operator() (const kernelArg &arg0){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    
    (*this)[launchDevice](arg0);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);

    (*this)[launchDevice](arg0,  arg1);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
                      arg3);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
                      arg3,  arg4);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11);
  }

  void kernel::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12){
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));

    if(kHandle->nestedKernelCount == 0){
      (*kHandle)(arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20);
    }
    else{
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20);
    }
  }

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21);
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22);
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
                      arg3,  arg4,  arg5, 
                      arg6,  arg7,  arg8, 
                      arg9,  arg10,  arg11, 
                      arg12,  arg13,  arg14, 
                      arg15,  arg16,  arg17, 
                      arg18,  arg19,  arg20, 
                      arg21,  arg22,  arg23);
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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

  void kernelDatabase::operator() (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26){
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));
    arg40.setupForKernelCall(kHandle->metaInfo.argIsConst(40));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);
    else if(arg40.dHandle) launchDevice = const_cast<occa::device_v*>(arg40.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));
    arg40.setupForKernelCall(kHandle->metaInfo.argIsConst(40));
    arg41.setupForKernelCall(kHandle->metaInfo.argIsConst(41));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);
    else if(arg40.dHandle) launchDevice = const_cast<occa::device_v*>(arg40.dHandle);
    else if(arg41.dHandle) launchDevice = const_cast<occa::device_v*>(arg41.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));
    arg40.setupForKernelCall(kHandle->metaInfo.argIsConst(40));
    arg41.setupForKernelCall(kHandle->metaInfo.argIsConst(41));
    arg42.setupForKernelCall(kHandle->metaInfo.argIsConst(42));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);
    else if(arg40.dHandle) launchDevice = const_cast<occa::device_v*>(arg40.dHandle);
    else if(arg41.dHandle) launchDevice = const_cast<occa::device_v*>(arg41.dHandle);
    else if(arg42.dHandle) launchDevice = const_cast<occa::device_v*>(arg42.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));
    arg40.setupForKernelCall(kHandle->metaInfo.argIsConst(40));
    arg41.setupForKernelCall(kHandle->metaInfo.argIsConst(41));
    arg42.setupForKernelCall(kHandle->metaInfo.argIsConst(42));
    arg43.setupForKernelCall(kHandle->metaInfo.argIsConst(43));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);
    else if(arg40.dHandle) launchDevice = const_cast<occa::device_v*>(arg40.dHandle);
    else if(arg41.dHandle) launchDevice = const_cast<occa::device_v*>(arg41.dHandle);
    else if(arg42.dHandle) launchDevice = const_cast<occa::device_v*>(arg42.dHandle);
    else if(arg43.dHandle) launchDevice = const_cast<occa::device_v*>(arg43.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));
    arg40.setupForKernelCall(kHandle->metaInfo.argIsConst(40));
    arg41.setupForKernelCall(kHandle->metaInfo.argIsConst(41));
    arg42.setupForKernelCall(kHandle->metaInfo.argIsConst(42));
    arg43.setupForKernelCall(kHandle->metaInfo.argIsConst(43));
    arg44.setupForKernelCall(kHandle->metaInfo.argIsConst(44));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);
    else if(arg40.dHandle) launchDevice = const_cast<occa::device_v*>(arg40.dHandle);
    else if(arg41.dHandle) launchDevice = const_cast<occa::device_v*>(arg41.dHandle);
    else if(arg42.dHandle) launchDevice = const_cast<occa::device_v*>(arg42.dHandle);
    else if(arg43.dHandle) launchDevice = const_cast<occa::device_v*>(arg43.dHandle);
    else if(arg44.dHandle) launchDevice = const_cast<occa::device_v*>(arg44.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));
    arg40.setupForKernelCall(kHandle->metaInfo.argIsConst(40));
    arg41.setupForKernelCall(kHandle->metaInfo.argIsConst(41));
    arg42.setupForKernelCall(kHandle->metaInfo.argIsConst(42));
    arg43.setupForKernelCall(kHandle->metaInfo.argIsConst(43));
    arg44.setupForKernelCall(kHandle->metaInfo.argIsConst(44));
    arg45.setupForKernelCall(kHandle->metaInfo.argIsConst(45));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);
    else if(arg40.dHandle) launchDevice = const_cast<occa::device_v*>(arg40.dHandle);
    else if(arg41.dHandle) launchDevice = const_cast<occa::device_v*>(arg41.dHandle);
    else if(arg42.dHandle) launchDevice = const_cast<occa::device_v*>(arg42.dHandle);
    else if(arg43.dHandle) launchDevice = const_cast<occa::device_v*>(arg43.dHandle);
    else if(arg44.dHandle) launchDevice = const_cast<occa::device_v*>(arg44.dHandle);
    else if(arg45.dHandle) launchDevice = const_cast<occa::device_v*>(arg45.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));
    arg40.setupForKernelCall(kHandle->metaInfo.argIsConst(40));
    arg41.setupForKernelCall(kHandle->metaInfo.argIsConst(41));
    arg42.setupForKernelCall(kHandle->metaInfo.argIsConst(42));
    arg43.setupForKernelCall(kHandle->metaInfo.argIsConst(43));
    arg44.setupForKernelCall(kHandle->metaInfo.argIsConst(44));
    arg45.setupForKernelCall(kHandle->metaInfo.argIsConst(45));
    arg46.setupForKernelCall(kHandle->metaInfo.argIsConst(46));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);
    else if(arg40.dHandle) launchDevice = const_cast<occa::device_v*>(arg40.dHandle);
    else if(arg41.dHandle) launchDevice = const_cast<occa::device_v*>(arg41.dHandle);
    else if(arg42.dHandle) launchDevice = const_cast<occa::device_v*>(arg42.dHandle);
    else if(arg43.dHandle) launchDevice = const_cast<occa::device_v*>(arg43.dHandle);
    else if(arg44.dHandle) launchDevice = const_cast<occa::device_v*>(arg44.dHandle);
    else if(arg45.dHandle) launchDevice = const_cast<occa::device_v*>(arg45.dHandle);
    else if(arg46.dHandle) launchDevice = const_cast<occa::device_v*>(arg46.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));
    arg40.setupForKernelCall(kHandle->metaInfo.argIsConst(40));
    arg41.setupForKernelCall(kHandle->metaInfo.argIsConst(41));
    arg42.setupForKernelCall(kHandle->metaInfo.argIsConst(42));
    arg43.setupForKernelCall(kHandle->metaInfo.argIsConst(43));
    arg44.setupForKernelCall(kHandle->metaInfo.argIsConst(44));
    arg45.setupForKernelCall(kHandle->metaInfo.argIsConst(45));
    arg46.setupForKernelCall(kHandle->metaInfo.argIsConst(46));
    arg47.setupForKernelCall(kHandle->metaInfo.argIsConst(47));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);
    else if(arg40.dHandle) launchDevice = const_cast<occa::device_v*>(arg40.dHandle);
    else if(arg41.dHandle) launchDevice = const_cast<occa::device_v*>(arg41.dHandle);
    else if(arg42.dHandle) launchDevice = const_cast<occa::device_v*>(arg42.dHandle);
    else if(arg43.dHandle) launchDevice = const_cast<occa::device_v*>(arg43.dHandle);
    else if(arg44.dHandle) launchDevice = const_cast<occa::device_v*>(arg44.dHandle);
    else if(arg45.dHandle) launchDevice = const_cast<occa::device_v*>(arg45.dHandle);
    else if(arg46.dHandle) launchDevice = const_cast<occa::device_v*>(arg46.dHandle);
    else if(arg47.dHandle) launchDevice = const_cast<occa::device_v*>(arg47.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));
    arg40.setupForKernelCall(kHandle->metaInfo.argIsConst(40));
    arg41.setupForKernelCall(kHandle->metaInfo.argIsConst(41));
    arg42.setupForKernelCall(kHandle->metaInfo.argIsConst(42));
    arg43.setupForKernelCall(kHandle->metaInfo.argIsConst(43));
    arg44.setupForKernelCall(kHandle->metaInfo.argIsConst(44));
    arg45.setupForKernelCall(kHandle->metaInfo.argIsConst(45));
    arg46.setupForKernelCall(kHandle->metaInfo.argIsConst(46));
    arg47.setupForKernelCall(kHandle->metaInfo.argIsConst(47));
    arg48.setupForKernelCall(kHandle->metaInfo.argIsConst(48));

    if(kHandle->nestedKernelCount == 0){
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
      (*kHandle)(kHandle->nestedKernels, arg0,  arg1,  arg2, 
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);
    else if(arg40.dHandle) launchDevice = const_cast<occa::device_v*>(arg40.dHandle);
    else if(arg41.dHandle) launchDevice = const_cast<occa::device_v*>(arg41.dHandle);
    else if(arg42.dHandle) launchDevice = const_cast<occa::device_v*>(arg42.dHandle);
    else if(arg43.dHandle) launchDevice = const_cast<occa::device_v*>(arg43.dHandle);
    else if(arg44.dHandle) launchDevice = const_cast<occa::device_v*>(arg44.dHandle);
    else if(arg45.dHandle) launchDevice = const_cast<occa::device_v*>(arg45.dHandle);
    else if(arg46.dHandle) launchDevice = const_cast<occa::device_v*>(arg46.dHandle);
    else if(arg47.dHandle) launchDevice = const_cast<occa::device_v*>(arg47.dHandle);
    else if(arg48.dHandle) launchDevice = const_cast<occa::device_v*>(arg48.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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
    arg0.setupForKernelCall(kHandle->metaInfo.argIsConst(0));
    arg1.setupForKernelCall(kHandle->metaInfo.argIsConst(1));
    arg2.setupForKernelCall(kHandle->metaInfo.argIsConst(2));
    arg3.setupForKernelCall(kHandle->metaInfo.argIsConst(3));
    arg4.setupForKernelCall(kHandle->metaInfo.argIsConst(4));
    arg5.setupForKernelCall(kHandle->metaInfo.argIsConst(5));
    arg6.setupForKernelCall(kHandle->metaInfo.argIsConst(6));
    arg7.setupForKernelCall(kHandle->metaInfo.argIsConst(7));
    arg8.setupForKernelCall(kHandle->metaInfo.argIsConst(8));
    arg9.setupForKernelCall(kHandle->metaInfo.argIsConst(9));
    arg10.setupForKernelCall(kHandle->metaInfo.argIsConst(10));
    arg11.setupForKernelCall(kHandle->metaInfo.argIsConst(11));
    arg12.setupForKernelCall(kHandle->metaInfo.argIsConst(12));
    arg13.setupForKernelCall(kHandle->metaInfo.argIsConst(13));
    arg14.setupForKernelCall(kHandle->metaInfo.argIsConst(14));
    arg15.setupForKernelCall(kHandle->metaInfo.argIsConst(15));
    arg16.setupForKernelCall(kHandle->metaInfo.argIsConst(16));
    arg17.setupForKernelCall(kHandle->metaInfo.argIsConst(17));
    arg18.setupForKernelCall(kHandle->metaInfo.argIsConst(18));
    arg19.setupForKernelCall(kHandle->metaInfo.argIsConst(19));
    arg20.setupForKernelCall(kHandle->metaInfo.argIsConst(20));
    arg21.setupForKernelCall(kHandle->metaInfo.argIsConst(21));
    arg22.setupForKernelCall(kHandle->metaInfo.argIsConst(22));
    arg23.setupForKernelCall(kHandle->metaInfo.argIsConst(23));
    arg24.setupForKernelCall(kHandle->metaInfo.argIsConst(24));
    arg25.setupForKernelCall(kHandle->metaInfo.argIsConst(25));
    arg26.setupForKernelCall(kHandle->metaInfo.argIsConst(26));
    arg27.setupForKernelCall(kHandle->metaInfo.argIsConst(27));
    arg28.setupForKernelCall(kHandle->metaInfo.argIsConst(28));
    arg29.setupForKernelCall(kHandle->metaInfo.argIsConst(29));
    arg30.setupForKernelCall(kHandle->metaInfo.argIsConst(30));
    arg31.setupForKernelCall(kHandle->metaInfo.argIsConst(31));
    arg32.setupForKernelCall(kHandle->metaInfo.argIsConst(32));
    arg33.setupForKernelCall(kHandle->metaInfo.argIsConst(33));
    arg34.setupForKernelCall(kHandle->metaInfo.argIsConst(34));
    arg35.setupForKernelCall(kHandle->metaInfo.argIsConst(35));
    arg36.setupForKernelCall(kHandle->metaInfo.argIsConst(36));
    arg37.setupForKernelCall(kHandle->metaInfo.argIsConst(37));
    arg38.setupForKernelCall(kHandle->metaInfo.argIsConst(38));
    arg39.setupForKernelCall(kHandle->metaInfo.argIsConst(39));
    arg40.setupForKernelCall(kHandle->metaInfo.argIsConst(40));
    arg41.setupForKernelCall(kHandle->metaInfo.argIsConst(41));
    arg42.setupForKernelCall(kHandle->metaInfo.argIsConst(42));
    arg43.setupForKernelCall(kHandle->metaInfo.argIsConst(43));
    arg44.setupForKernelCall(kHandle->metaInfo.argIsConst(44));
    arg45.setupForKernelCall(kHandle->metaInfo.argIsConst(45));
    arg46.setupForKernelCall(kHandle->metaInfo.argIsConst(46));
    arg47.setupForKernelCall(kHandle->metaInfo.argIsConst(47));
    arg48.setupForKernelCall(kHandle->metaInfo.argIsConst(48));
    arg49.setupForKernelCall(kHandle->metaInfo.argIsConst(49));

    if(kHandle->nestedKernelCount == 0){
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
    occa::device_v *launchDevice = NULL;

    if(arg0.dHandle) launchDevice = const_cast<occa::device_v*>(arg0.dHandle);
    else if(arg1.dHandle) launchDevice = const_cast<occa::device_v*>(arg1.dHandle);
    else if(arg2.dHandle) launchDevice = const_cast<occa::device_v*>(arg2.dHandle);
    else if(arg3.dHandle) launchDevice = const_cast<occa::device_v*>(arg3.dHandle);
    else if(arg4.dHandle) launchDevice = const_cast<occa::device_v*>(arg4.dHandle);
    else if(arg5.dHandle) launchDevice = const_cast<occa::device_v*>(arg5.dHandle);
    else if(arg6.dHandle) launchDevice = const_cast<occa::device_v*>(arg6.dHandle);
    else if(arg7.dHandle) launchDevice = const_cast<occa::device_v*>(arg7.dHandle);
    else if(arg8.dHandle) launchDevice = const_cast<occa::device_v*>(arg8.dHandle);
    else if(arg9.dHandle) launchDevice = const_cast<occa::device_v*>(arg9.dHandle);
    else if(arg10.dHandle) launchDevice = const_cast<occa::device_v*>(arg10.dHandle);
    else if(arg11.dHandle) launchDevice = const_cast<occa::device_v*>(arg11.dHandle);
    else if(arg12.dHandle) launchDevice = const_cast<occa::device_v*>(arg12.dHandle);
    else if(arg13.dHandle) launchDevice = const_cast<occa::device_v*>(arg13.dHandle);
    else if(arg14.dHandle) launchDevice = const_cast<occa::device_v*>(arg14.dHandle);
    else if(arg15.dHandle) launchDevice = const_cast<occa::device_v*>(arg15.dHandle);
    else if(arg16.dHandle) launchDevice = const_cast<occa::device_v*>(arg16.dHandle);
    else if(arg17.dHandle) launchDevice = const_cast<occa::device_v*>(arg17.dHandle);
    else if(arg18.dHandle) launchDevice = const_cast<occa::device_v*>(arg18.dHandle);
    else if(arg19.dHandle) launchDevice = const_cast<occa::device_v*>(arg19.dHandle);
    else if(arg20.dHandle) launchDevice = const_cast<occa::device_v*>(arg20.dHandle);
    else if(arg21.dHandle) launchDevice = const_cast<occa::device_v*>(arg21.dHandle);
    else if(arg22.dHandle) launchDevice = const_cast<occa::device_v*>(arg22.dHandle);
    else if(arg23.dHandle) launchDevice = const_cast<occa::device_v*>(arg23.dHandle);
    else if(arg24.dHandle) launchDevice = const_cast<occa::device_v*>(arg24.dHandle);
    else if(arg25.dHandle) launchDevice = const_cast<occa::device_v*>(arg25.dHandle);
    else if(arg26.dHandle) launchDevice = const_cast<occa::device_v*>(arg26.dHandle);
    else if(arg27.dHandle) launchDevice = const_cast<occa::device_v*>(arg27.dHandle);
    else if(arg28.dHandle) launchDevice = const_cast<occa::device_v*>(arg28.dHandle);
    else if(arg29.dHandle) launchDevice = const_cast<occa::device_v*>(arg29.dHandle);
    else if(arg30.dHandle) launchDevice = const_cast<occa::device_v*>(arg30.dHandle);
    else if(arg31.dHandle) launchDevice = const_cast<occa::device_v*>(arg31.dHandle);
    else if(arg32.dHandle) launchDevice = const_cast<occa::device_v*>(arg32.dHandle);
    else if(arg33.dHandle) launchDevice = const_cast<occa::device_v*>(arg33.dHandle);
    else if(arg34.dHandle) launchDevice = const_cast<occa::device_v*>(arg34.dHandle);
    else if(arg35.dHandle) launchDevice = const_cast<occa::device_v*>(arg35.dHandle);
    else if(arg36.dHandle) launchDevice = const_cast<occa::device_v*>(arg36.dHandle);
    else if(arg37.dHandle) launchDevice = const_cast<occa::device_v*>(arg37.dHandle);
    else if(arg38.dHandle) launchDevice = const_cast<occa::device_v*>(arg38.dHandle);
    else if(arg39.dHandle) launchDevice = const_cast<occa::device_v*>(arg39.dHandle);
    else if(arg40.dHandle) launchDevice = const_cast<occa::device_v*>(arg40.dHandle);
    else if(arg41.dHandle) launchDevice = const_cast<occa::device_v*>(arg41.dHandle);
    else if(arg42.dHandle) launchDevice = const_cast<occa::device_v*>(arg42.dHandle);
    else if(arg43.dHandle) launchDevice = const_cast<occa::device_v*>(arg43.dHandle);
    else if(arg44.dHandle) launchDevice = const_cast<occa::device_v*>(arg44.dHandle);
    else if(arg45.dHandle) launchDevice = const_cast<occa::device_v*>(arg45.dHandle);
    else if(arg46.dHandle) launchDevice = const_cast<occa::device_v*>(arg46.dHandle);
    else if(arg47.dHandle) launchDevice = const_cast<occa::device_v*>(arg47.dHandle);
    else if(arg48.dHandle) launchDevice = const_cast<occa::device_v*>(arg48.dHandle);
    else if(arg49.dHandle) launchDevice = const_cast<occa::device_v*>(arg49.dHandle);

    (*this)[launchDevice](arg0,  arg1,  arg2, 
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