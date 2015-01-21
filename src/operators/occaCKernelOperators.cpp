    void OCCA_RFUNC occaKernelRun1(occaKernel kernel, void *arg0){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun2(occaKernel kernel, void *arg0,  void *arg1){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun3(occaKernel kernel, void *arg0,  void *arg1,  void *arg2){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun4(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun5(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun6(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun7(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun8(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun9(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun10(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun11(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun12(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun13(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun14(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun15(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun16(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun17(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun18(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun19(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun20(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun21(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun22(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun23(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }


    void OCCA_RFUNC occaKernelRun24(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23){
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg40);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg40);
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg40);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg40);
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg41);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg41);
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg40);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg40);
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg41);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg41);
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg42);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg42);
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg40);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg40);
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg41);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg41);
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg42);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg42);
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg43);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg43);
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg40);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg40);
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg41);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg41);
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg42);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg42);
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg43);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg43);
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg44);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg44);
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg40);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg40);
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg41);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg41);
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg42);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg42);
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg43);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg43);
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg44);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg44);
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg45);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(45, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg45);
        __occa_kernel__.addArgument(45, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg40);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg40);
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg41);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg41);
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg42);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg42);
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg43);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg43);
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg44);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg44);
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg45);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(45, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg45);
        __occa_kernel__.addArgument(45, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg46);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(46, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg46);
        __occa_kernel__.addArgument(46, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg40);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg40);
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg41);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg41);
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg42);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg42);
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg43);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg43);
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg44);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg44);
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg45);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(45, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg45);
        __occa_kernel__.addArgument(45, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg46);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(46, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg46);
        __occa_kernel__.addArgument(46, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg47);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(47, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg47);
        __occa_kernel__.addArgument(47, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg40);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg40);
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg41);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg41);
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg42);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg42);
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg43);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg43);
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg44);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg44);
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg45);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(45, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg45);
        __occa_kernel__.addArgument(45, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg46);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(46, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg46);
        __occa_kernel__.addArgument(46, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg47);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(47, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg47);
        __occa_kernel__.addArgument(47, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg48);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(48, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg48);
        __occa_kernel__.addArgument(48, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
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
      occa::kernel &__occa_kernel__ = *((occa::kernel*) kernel);
    __occa_kernel__.clearArgumentList();
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg0);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg0);
        __occa_kernel__.addArgument(0, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg1);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg1);
        __occa_kernel__.addArgument(1, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg2);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg2);
        __occa_kernel__.addArgument(2, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg3);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg3);
        __occa_kernel__.addArgument(3, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg4);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg4);
        __occa_kernel__.addArgument(4, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg5);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg5);
        __occa_kernel__.addArgument(5, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg6);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg6);
        __occa_kernel__.addArgument(6, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg7);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg7);
        __occa_kernel__.addArgument(7, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg8);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg8);
        __occa_kernel__.addArgument(8, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg9);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg9);
        __occa_kernel__.addArgument(9, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg10);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg10);
        __occa_kernel__.addArgument(10, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg11);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg11);
        __occa_kernel__.addArgument(11, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg12);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg12);
        __occa_kernel__.addArgument(12, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg13);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg13);
        __occa_kernel__.addArgument(13, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg14);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg14);
        __occa_kernel__.addArgument(14, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg15);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg15);
        __occa_kernel__.addArgument(15, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg16);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg16);
        __occa_kernel__.addArgument(16, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg17);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg17);
        __occa_kernel__.addArgument(17, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg18);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg18);
        __occa_kernel__.addArgument(18, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg19);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg19);
        __occa_kernel__.addArgument(19, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg20);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg20);
        __occa_kernel__.addArgument(20, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg21);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg21);
        __occa_kernel__.addArgument(21, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg22);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg22);
        __occa_kernel__.addArgument(22, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg23);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg23);
        __occa_kernel__.addArgument(23, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg24);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg24);
        __occa_kernel__.addArgument(24, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg25);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg25);
        __occa_kernel__.addArgument(25, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg26);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg26);
        __occa_kernel__.addArgument(26, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg27);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg27);
        __occa_kernel__.addArgument(27, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg28);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg28);
        __occa_kernel__.addArgument(28, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg29);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg29);
        __occa_kernel__.addArgument(29, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg30);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg30);
        __occa_kernel__.addArgument(30, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg31);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg31);
        __occa_kernel__.addArgument(31, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg32);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg32);
        __occa_kernel__.addArgument(32, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg33);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg33);
        __occa_kernel__.addArgument(33, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg34);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg34);
        __occa_kernel__.addArgument(34, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg35);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg35);
        __occa_kernel__.addArgument(35, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg36);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg36);
        __occa_kernel__.addArgument(36, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg37);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg37);
        __occa_kernel__.addArgument(37, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg38);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg38);
        __occa_kernel__.addArgument(38, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg39);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg39);
        __occa_kernel__.addArgument(39, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg40);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg40);
        __occa_kernel__.addArgument(40, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg41);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg41);
        __occa_kernel__.addArgument(41, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg42);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg42);
        __occa_kernel__.addArgument(42, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg43);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg43);
        __occa_kernel__.addArgument(43, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg44);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg44);
        __occa_kernel__.addArgument(44, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg45);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(45, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg45);
        __occa_kernel__.addArgument(45, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg46);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(46, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg46);
        __occa_kernel__.addArgument(46, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg47);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(47, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg47);
        __occa_kernel__.addArgument(47, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg48);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(48, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg48);
        __occa_kernel__.addArgument(48, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    {
      occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg49);
      if(__occa_memory__.type == 0){
        __occa_kernel__.addArgument(49, occa::kernelArg(__occa_memory__.mem));
      }
      else{
        occaType_t &__occa_type__ = *((occaType_t*) arg49);
        __occa_kernel__.addArgument(49, occa::kernelArg(__occa_type__.value, occaTypeSize[__occa_type__.type], false));
      }
    }
    __occa_kernel__.runFromArguments();
    }
