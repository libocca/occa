    void OCCA_RFUNC occaKernelRun1(occaKernel kernel, void *arg0){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[1] = {(occaMemory_t*) arg0};
      
      for(int i = 0; i < 1; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun2(occaKernel kernel, void *arg0,  void *arg1){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[2] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1};
      
      for(int i = 0; i < 2; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun3(occaKernel kernel, void *arg0,  void *arg1,  void *arg2){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[3] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2};
      
      for(int i = 0; i < 3; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun4(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[4] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3};
      
      for(int i = 0; i < 4; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun5(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[5] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4};
      
      for(int i = 0; i < 5; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun6(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[6] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5};
      
      for(int i = 0; i < 6; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun7(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[7] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6};
      
      for(int i = 0; i < 7; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun8(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[8] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7};
      
      for(int i = 0; i < 8; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun9(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[9] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8};
      
      for(int i = 0; i < 9; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun10(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[10] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9};
      
      for(int i = 0; i < 10; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun11(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[11] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10};
      
      for(int i = 0; i < 11; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun12(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[12] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11};
      
      for(int i = 0; i < 12; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun13(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[13] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12};
      
      for(int i = 0; i < 13; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun14(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[14] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13};
      
      for(int i = 0; i < 14; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun15(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[15] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14};
      
      for(int i = 0; i < 15; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun16(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[16] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15};
      
      for(int i = 0; i < 16; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun17(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[17] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16};
      
      for(int i = 0; i < 17; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun18(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[18] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17};
      
      for(int i = 0; i < 18; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun19(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[19] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18};
      
      for(int i = 0; i < 19; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun20(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[20] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19};
      
      for(int i = 0; i < 20; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun21(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[21] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20};
      
      for(int i = 0; i < 21; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun22(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[22] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21};
      
      for(int i = 0; i < 22; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun23(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[23] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22};
      
      for(int i = 0; i < 23; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun24(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[24] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23};
      
      for(int i = 0; i < 24; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun25(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[25] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24};
      
      for(int i = 0; i < 25; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun26(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[26] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25};
      
      for(int i = 0; i < 26; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun27(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[27] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26};
      
      for(int i = 0; i < 27; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun28(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[28] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27};
      
      for(int i = 0; i < 28; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun29(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[29] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28};
      
      for(int i = 0; i < 29; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun30(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[30] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29};
      
      for(int i = 0; i < 30; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun31(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[31] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30};
      
      for(int i = 0; i < 31; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun32(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[32] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31};
      
      for(int i = 0; i < 32; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun33(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[33] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32};
      
      for(int i = 0; i < 33; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun34(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[34] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33};
      
      for(int i = 0; i < 34; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun35(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[35] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34};
      
      for(int i = 0; i < 35; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun36(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[36] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35};
      
      for(int i = 0; i < 36; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun37(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[37] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36};
      
      for(int i = 0; i < 37; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun38(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[38] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37};
      
      for(int i = 0; i < 38; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun39(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[39] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38};
      
      for(int i = 0; i < 39; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun40(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[40] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39};
      
      for(int i = 0; i < 40; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun41(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39,  void *arg40){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[41] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39, (occaMemory_t*) arg40};
      
      for(int i = 0; i < 41; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun42(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39,  void *arg40,  void *arg41){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[42] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39, (occaMemory_t*) arg40, (occaMemory_t*) arg41};
      
      for(int i = 0; i < 42; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun43(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39,  void *arg40,  void *arg41, 
                      void *arg42){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[43] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39, (occaMemory_t*) arg40, (occaMemory_t*) arg41, (occaMemory_t*) arg42};
      
      for(int i = 0; i < 43; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun44(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39,  void *arg40,  void *arg41, 
                      void *arg42,  void *arg43){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[44] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39, (occaMemory_t*) arg40, (occaMemory_t*) arg41, (occaMemory_t*) arg42, (occaMemory_t*) arg43};
      
      for(int i = 0; i < 44; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun45(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39,  void *arg40,  void *arg41, 
                      void *arg42,  void *arg43,  void *arg44){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[45] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39, (occaMemory_t*) arg40, (occaMemory_t*) arg41, (occaMemory_t*) arg42, (occaMemory_t*) arg43, (occaMemory_t*) arg44};
      
      for(int i = 0; i < 45; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun46(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39,  void *arg40,  void *arg41, 
                      void *arg42,  void *arg43,  void *arg44, 
                      void *arg45){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[46] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39, (occaMemory_t*) arg40, (occaMemory_t*) arg41, (occaMemory_t*) arg42, (occaMemory_t*) arg43, (occaMemory_t*) arg44, (occaMemory_t*) arg45};
      
      for(int i = 0; i < 46; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun47(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39,  void *arg40,  void *arg41, 
                      void *arg42,  void *arg43,  void *arg44, 
                      void *arg45,  void *arg46){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[47] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39, (occaMemory_t*) arg40, (occaMemory_t*) arg41, (occaMemory_t*) arg42, (occaMemory_t*) arg43, (occaMemory_t*) arg44, (occaMemory_t*) arg45, (occaMemory_t*) arg46};
      
      for(int i = 0; i < 47; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun48(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39,  void *arg40,  void *arg41, 
                      void *arg42,  void *arg43,  void *arg44, 
                      void *arg45,  void *arg46,  void *arg47){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[48] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39, (occaMemory_t*) arg40, (occaMemory_t*) arg41, (occaMemory_t*) arg42, (occaMemory_t*) arg43, (occaMemory_t*) arg44, (occaMemory_t*) arg45, (occaMemory_t*) arg46, (occaMemory_t*) arg47};
      
      for(int i = 0; i < 48; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun49(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39,  void *arg40,  void *arg41, 
                      void *arg42,  void *arg43,  void *arg44, 
                      void *arg45,  void *arg46,  void *arg47, 
                      void *arg48){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[49] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39, (occaMemory_t*) arg40, (occaMemory_t*) arg41, (occaMemory_t*) arg42, (occaMemory_t*) arg43, (occaMemory_t*) arg44, (occaMemory_t*) arg45, (occaMemory_t*) arg46, (occaMemory_t*) arg47, (occaMemory_t*) arg48};
      
      for(int i = 0; i < 49; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun50(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23, 
                      void *arg24,  void *arg25,  void *arg26, 
                      void *arg27,  void *arg28,  void *arg29, 
                      void *arg30,  void *arg31,  void *arg32, 
                      void *arg33,  void *arg34,  void *arg35, 
                      void *arg36,  void *arg37,  void *arg38, 
                      void *arg39,  void *arg40,  void *arg41, 
                      void *arg42,  void *arg43,  void *arg44, 
                      void *arg45,  void *arg46,  void *arg47, 
                      void *arg48,  void *arg49){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaMemory_t *args[50] = {(occaMemory_t*) arg0, (occaMemory_t*) arg1, (occaMemory_t*) arg2, (occaMemory_t*) arg3, (occaMemory_t*) arg4, (occaMemory_t*) arg5, (occaMemory_t*) arg6, (occaMemory_t*) arg7, (occaMemory_t*) arg8, (occaMemory_t*) arg9, (occaMemory_t*) arg10, (occaMemory_t*) arg11, (occaMemory_t*) arg12, (occaMemory_t*) arg13, (occaMemory_t*) arg14, (occaMemory_t*) arg15, (occaMemory_t*) arg16, (occaMemory_t*) arg17, (occaMemory_t*) arg18, (occaMemory_t*) arg19, (occaMemory_t*) arg20, (occaMemory_t*) arg21, (occaMemory_t*) arg22, (occaMemory_t*) arg23, (occaMemory_t*) arg24, (occaMemory_t*) arg25, (occaMemory_t*) arg26, (occaMemory_t*) arg27, (occaMemory_t*) arg28, (occaMemory_t*) arg29, (occaMemory_t*) arg30, (occaMemory_t*) arg31, (occaMemory_t*) arg32, (occaMemory_t*) arg33, (occaMemory_t*) arg34, (occaMemory_t*) arg35, (occaMemory_t*) arg36, (occaMemory_t*) arg37, (occaMemory_t*) arg38, (occaMemory_t*) arg39, (occaMemory_t*) arg40, (occaMemory_t*) arg41, (occaMemory_t*) arg42, (occaMemory_t*) arg43, (occaMemory_t*) arg44, (occaMemory_t*) arg45, (occaMemory_t*) arg46, (occaMemory_t*) arg47, (occaMemory_t*) arg48, (occaMemory_t*) arg49};
      
      for(int i = 0; i < 50; ++i){
        occaMemory_t &memory = *(args[i]);
        if(memory.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) memory.mHandle);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else{
          occaType_t &type_ = *((occaType_t*) args[i]);
          kernel_.addArgument(i, occa::kernelArg(type_.value, type_.bytes, (memory.type == OCCA_TYPE_STRUCT)));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }
