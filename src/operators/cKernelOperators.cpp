    void OCCA_RFUNC occaKernelRun1(occaKernel kernel, void *arg0){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaType_t *args[1] = {(occaType_t*) arg0};
      
      for(int i = 0; i < 1; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun2(occaKernel kernel, void *arg0,  void *arg1){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaType_t *args[2] = {(occaType_t*) arg0, (occaType_t*) arg1};
      
      for(int i = 0; i < 2; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun3(occaKernel kernel, void *arg0,  void *arg1,  void *arg2){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaType_t *args[3] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2};
      
      for(int i = 0; i < 3; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun4(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaType_t *args[4] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3};
      
      for(int i = 0; i < 4; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun5(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaType_t *args[5] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4};
      
      for(int i = 0; i < 5; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun6(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaType_t *args[6] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5};
      
      for(int i = 0; i < 6; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[7] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6};
      
      for(int i = 0; i < 7; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[8] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7};
      
      for(int i = 0; i < 8; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[9] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8};
      
      for(int i = 0; i < 9; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[10] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9};
      
      for(int i = 0; i < 10; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[11] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10};
      
      for(int i = 0; i < 11; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[12] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11};
      
      for(int i = 0; i < 12; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[13] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12};
      
      for(int i = 0; i < 13; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[14] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13};
      
      for(int i = 0; i < 14; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[15] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14};
      
      for(int i = 0; i < 15; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[16] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15};
      
      for(int i = 0; i < 16; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[17] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16};
      
      for(int i = 0; i < 17; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[18] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17};
      
      for(int i = 0; i < 18; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[19] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18};
      
      for(int i = 0; i < 19; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[20] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19};
      
      for(int i = 0; i < 20; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[21] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20};
      
      for(int i = 0; i < 21; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[22] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21};
      
      for(int i = 0; i < 22; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[23] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22};
      
      for(int i = 0; i < 23; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[24] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23};
      
      for(int i = 0; i < 24; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[25] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24};
      
      for(int i = 0; i < 25; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[26] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25};
      
      for(int i = 0; i < 26; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[27] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26};
      
      for(int i = 0; i < 27; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[28] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27};
      
      for(int i = 0; i < 28; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[29] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28};
      
      for(int i = 0; i < 29; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[30] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29};
      
      for(int i = 0; i < 30; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[31] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30};
      
      for(int i = 0; i < 31; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[32] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31};
      
      for(int i = 0; i < 32; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[33] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32};
      
      for(int i = 0; i < 33; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[34] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33};
      
      for(int i = 0; i < 34; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[35] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34};
      
      for(int i = 0; i < 35; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[36] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35};
      
      for(int i = 0; i < 36; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[37] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36};
      
      for(int i = 0; i < 37; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[38] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37};
      
      for(int i = 0; i < 38; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[39] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38};
      
      for(int i = 0; i < 39; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[40] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39};
      
      for(int i = 0; i < 40; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[41] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39, (occaType_t*) arg40};
      
      for(int i = 0; i < 41; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[42] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39, (occaType_t*) arg40, (occaType_t*) arg41};
      
      for(int i = 0; i < 42; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[43] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39, (occaType_t*) arg40, (occaType_t*) arg41, (occaType_t*) arg42};
      
      for(int i = 0; i < 43; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[44] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39, (occaType_t*) arg40, (occaType_t*) arg41, (occaType_t*) arg42, (occaType_t*) arg43};
      
      for(int i = 0; i < 44; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[45] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39, (occaType_t*) arg40, (occaType_t*) arg41, (occaType_t*) arg42, (occaType_t*) arg43, (occaType_t*) arg44};
      
      for(int i = 0; i < 45; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[46] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39, (occaType_t*) arg40, (occaType_t*) arg41, (occaType_t*) arg42, (occaType_t*) arg43, (occaType_t*) arg44, (occaType_t*) arg45};
      
      for(int i = 0; i < 46; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[47] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39, (occaType_t*) arg40, (occaType_t*) arg41, (occaType_t*) arg42, (occaType_t*) arg43, (occaType_t*) arg44, (occaType_t*) arg45, (occaType_t*) arg46};
      
      for(int i = 0; i < 47; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[48] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39, (occaType_t*) arg40, (occaType_t*) arg41, (occaType_t*) arg42, (occaType_t*) arg43, (occaType_t*) arg44, (occaType_t*) arg45, (occaType_t*) arg46, (occaType_t*) arg47};
      
      for(int i = 0; i < 48; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[49] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39, (occaType_t*) arg40, (occaType_t*) arg41, (occaType_t*) arg42, (occaType_t*) arg43, (occaType_t*) arg44, (occaType_t*) arg45, (occaType_t*) arg46, (occaType_t*) arg47, (occaType_t*) arg48};
      
      for(int i = 0; i < 49; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
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
      
      occaType_t *args[50] = {(occaType_t*) arg0, (occaType_t*) arg1, (occaType_t*) arg2, (occaType_t*) arg3, (occaType_t*) arg4, (occaType_t*) arg5, (occaType_t*) arg6, (occaType_t*) arg7, (occaType_t*) arg8, (occaType_t*) arg9, (occaType_t*) arg10, (occaType_t*) arg11, (occaType_t*) arg12, (occaType_t*) arg13, (occaType_t*) arg14, (occaType_t*) arg15, (occaType_t*) arg16, (occaType_t*) arg17, (occaType_t*) arg18, (occaType_t*) arg19, (occaType_t*) arg20, (occaType_t*) arg21, (occaType_t*) arg22, (occaType_t*) arg23, (occaType_t*) arg24, (occaType_t*) arg25, (occaType_t*) arg26, (occaType_t*) arg27, (occaType_t*) arg28, (occaType_t*) arg29, (occaType_t*) arg30, (occaType_t*) arg31, (occaType_t*) arg32, (occaType_t*) arg33, (occaType_t*) arg34, (occaType_t*) arg35, (occaType_t*) arg36, (occaType_t*) arg37, (occaType_t*) arg38, (occaType_t*) arg39, (occaType_t*) arg40, (occaType_t*) arg41, (occaType_t*) arg42, (occaType_t*) arg43, (occaType_t*) arg44, (occaType_t*) arg45, (occaType_t*) arg46, (occaType_t*) arg47, (occaType_t*) arg48, (occaType_t*) arg49};
      
      for(int i = 0; i < 50; ++i){
        occaType_t &arg = *(args[i]);
        void *argPtr    = arg.value.data.void_;
      
        if(arg.type == OCCA_TYPE_MEMORY){
          occa::memory memory_((occa::memory_v*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else if(arg.type == OCCA_TYPE_PTR){
          occa::memory memory_((void*) argPtr);
          kernel_.addArgument(i, occa::kernelArg(memory_));
        }
        else {
          kernel_.addArgument(i, occa::kernelArg(arg.value));
          delete (occaType_t*) args[i];
        }
      }
      
      kernel_.runFromArguments();
    }

