    void OCCA_RFUNC occaKernelRun1(occaKernel kernel, void *arg0){
      occa::kernel kernel_((occa::kernel_v*) kernel);
      kernel_.clearArgumentList();
      
      occaType_t *args[1] = {((occaType) arg0)->ptr};
      
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
      
      occaType_t *args[2] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr};
      
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
      
      occaType_t *args[3] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr};
      
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
      
      occaType_t *args[4] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr};
      
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
      
      occaType_t *args[5] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr};
      
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
      
      occaType_t *args[6] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr};
      
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
      
      occaType_t *args[7] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr};
      
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
      
      occaType_t *args[8] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr};
      
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
      
      occaType_t *args[9] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr};
      
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
      
      occaType_t *args[10] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr};
      
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
      
      occaType_t *args[11] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr};
      
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
      
      occaType_t *args[12] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr};
      
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
      
      occaType_t *args[13] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr};
      
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
      
      occaType_t *args[14] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr};
      
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
      
      occaType_t *args[15] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr};
      
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
      
      occaType_t *args[16] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr};
      
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
      
      occaType_t *args[17] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr};
      
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
      
      occaType_t *args[18] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr};
      
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
      
      occaType_t *args[19] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr};
      
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
      
      occaType_t *args[20] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr};
      
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
      
      occaType_t *args[21] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr};
      
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
      
      occaType_t *args[22] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr};
      
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
      
      occaType_t *args[23] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr};
      
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
      
      occaType_t *args[24] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr};
      
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
      
      occaType_t *args[25] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr};
      
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
      
      occaType_t *args[26] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr};
      
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
      
      occaType_t *args[27] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr};
      
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
      
      occaType_t *args[28] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr};
      
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
      
      occaType_t *args[29] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr};
      
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
      
      occaType_t *args[30] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr};
      
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
      
      occaType_t *args[31] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr};
      
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
      
      occaType_t *args[32] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr};
      
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
      
      occaType_t *args[33] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr};
      
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
      
      occaType_t *args[34] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr};
      
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
      
      occaType_t *args[35] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr};
      
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
      
      occaType_t *args[36] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr};
      
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
      
      occaType_t *args[37] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr};
      
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
      
      occaType_t *args[38] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr};
      
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
      
      occaType_t *args[39] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr};
      
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
      
      occaType_t *args[40] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr};
      
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
      
      occaType_t *args[41] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr};
      
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
      
      occaType_t *args[42] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr};
      
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
      
      occaType_t *args[43] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr};
      
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
      
      occaType_t *args[44] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr};
      
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
      
      occaType_t *args[45] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr};
      
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
      
      occaType_t *args[46] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr, ((occaType) arg45)->ptr};
      
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
      
      occaType_t *args[47] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr, ((occaType) arg45)->ptr, ((occaType) arg46)->ptr};
      
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
      
      occaType_t *args[48] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr, ((occaType) arg45)->ptr, ((occaType) arg46)->ptr, ((occaType) arg47)->ptr};
      
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
      
      occaType_t *args[49] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr, ((occaType) arg45)->ptr, ((occaType) arg46)->ptr, ((occaType) arg47)->ptr, ((occaType) arg48)->ptr};
      
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
      
      occaType_t *args[50] = {((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr, ((occaType) arg45)->ptr, ((occaType) arg46)->ptr, ((occaType) arg47)->ptr, ((occaType) arg48)->ptr, ((occaType) arg49)->ptr};
      
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

