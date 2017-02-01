void OCCA_RFUNC occaKernelRun1(occaKernel kernel, void *arg0){

  occaType_t *args[1] = { ((occaType) arg0)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);;
  occaKernelRunN(kernel, 1, args);
}

void OCCA_RFUNC occaKernelRun2(occaKernel kernel, void *arg0,  void *arg1){

  occaType_t *args[2] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);;
  occaKernelRunN(kernel, 2, args);
}

void OCCA_RFUNC occaKernelRun3(occaKernel kernel, void *arg0,  void *arg1,  void *arg2){

  occaType_t *args[3] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);;
  occaKernelRunN(kernel, 3, args);
}

void OCCA_RFUNC occaKernelRun4(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3){

  occaType_t *args[4] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);;
  occaKernelRunN(kernel, 4, args);
}

void OCCA_RFUNC occaKernelRun5(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4){

  occaType_t *args[5] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);;
  occaKernelRunN(kernel, 5, args);
}

void OCCA_RFUNC occaKernelRun6(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5){

  occaType_t *args[6] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);;
  occaKernelRunN(kernel, 6, args);
}

void OCCA_RFUNC occaKernelRun7(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6){

  occaType_t *args[7] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);;
  occaKernelRunN(kernel, 7, args);
}

void OCCA_RFUNC occaKernelRun8(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7){

  occaType_t *args[8] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);;
  occaKernelRunN(kernel, 8, args);
}

void OCCA_RFUNC occaKernelRun9(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8){

  occaType_t *args[9] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);;
  occaKernelRunN(kernel, 9, args);
}

void OCCA_RFUNC occaKernelRun10(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9){

  occaType_t *args[10] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);;
  occaKernelRunN(kernel, 10, args);
}

void OCCA_RFUNC occaKernelRun11(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10){

  occaType_t *args[11] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);;
  occaKernelRunN(kernel, 11, args);
}

void OCCA_RFUNC occaKernelRun12(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11){

  occaType_t *args[12] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);;
  occaKernelRunN(kernel, 12, args);
}

void OCCA_RFUNC occaKernelRun13(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12){

  occaType_t *args[13] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);;
  occaKernelRunN(kernel, 13, args);
}

void OCCA_RFUNC occaKernelRun14(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13){

  occaType_t *args[14] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);;
  occaKernelRunN(kernel, 14, args);
}

void OCCA_RFUNC occaKernelRun15(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14){

  occaType_t *args[15] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);;
  occaKernelRunN(kernel, 15, args);
}

void OCCA_RFUNC occaKernelRun16(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15){

  occaType_t *args[16] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);;
  occaKernelRunN(kernel, 16, args);
}

void OCCA_RFUNC occaKernelRun17(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16){

  occaType_t *args[17] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);;
  occaKernelRunN(kernel, 17, args);
}

void OCCA_RFUNC occaKernelRun18(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17){

  occaType_t *args[18] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);;
  occaKernelRunN(kernel, 18, args);
}

void OCCA_RFUNC occaKernelRun19(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18){

  occaType_t *args[19] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);;
  occaKernelRunN(kernel, 19, args);
}

void OCCA_RFUNC occaKernelRun20(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19){

  occaType_t *args[20] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);;
  occaKernelRunN(kernel, 20, args);
}

void OCCA_RFUNC occaKernelRun21(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20){

  occaType_t *args[21] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);;
  occaKernelRunN(kernel, 21, args);
}

void OCCA_RFUNC occaKernelRun22(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21){

  occaType_t *args[22] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);;
  occaKernelRunN(kernel, 22, args);
}

void OCCA_RFUNC occaKernelRun23(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22){

  occaType_t *args[23] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);;
  occaKernelRunN(kernel, 23, args);
}

void OCCA_RFUNC occaKernelRun24(occaKernel kernel, void *arg0,  void *arg1,  void *arg2, 
                      void *arg3,  void *arg4,  void *arg5, 
                      void *arg6,  void *arg7,  void *arg8, 
                      void *arg9,  void *arg10,  void *arg11, 
                      void *arg12,  void *arg13,  void *arg14, 
                      void *arg15,  void *arg16,  void *arg17, 
                      void *arg18,  void *arg19,  void *arg20, 
                      void *arg21,  void *arg22,  void *arg23){

  occaType_t *args[24] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);;
  occaKernelRunN(kernel, 24, args);
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

  occaType_t *args[25] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);;
  occaKernelRunN(kernel, 25, args);
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

  occaType_t *args[26] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);;
  occaKernelRunN(kernel, 26, args);
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

  occaType_t *args[27] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);;
  occaKernelRunN(kernel, 27, args);
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

  occaType_t *args[28] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);;
  occaKernelRunN(kernel, 28, args);
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

  occaType_t *args[29] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);;
  occaKernelRunN(kernel, 29, args);
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

  occaType_t *args[30] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);;
  occaKernelRunN(kernel, 30, args);
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

  occaType_t *args[31] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);;
  occaKernelRunN(kernel, 31, args);
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

  occaType_t *args[32] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);;
  occaKernelRunN(kernel, 32, args);
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

  occaType_t *args[33] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);;
  occaKernelRunN(kernel, 33, args);
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

  occaType_t *args[34] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);;
  occaKernelRunN(kernel, 34, args);
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

  occaType_t *args[35] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);;
  occaKernelRunN(kernel, 35, args);
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

  occaType_t *args[36] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);;
  occaKernelRunN(kernel, 36, args);
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

  occaType_t *args[37] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);;
  occaKernelRunN(kernel, 37, args);
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

  occaType_t *args[38] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);;
  occaKernelRunN(kernel, 38, args);
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

  occaType_t *args[39] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);;
  occaKernelRunN(kernel, 39, args);
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

  occaType_t *args[40] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);;
  occaKernelRunN(kernel, 40, args);
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

  occaType_t *args[41] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);
  if(((occaType) arg40)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg40)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg40);;
  occaKernelRunN(kernel, 41, args);
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

  occaType_t *args[42] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);
  if(((occaType) arg40)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg40)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg40);
  if(((occaType) arg41)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg41)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg41);;
  occaKernelRunN(kernel, 42, args);
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

  occaType_t *args[43] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);
  if(((occaType) arg40)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg40)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg40);
  if(((occaType) arg41)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg41)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg41);
  if(((occaType) arg42)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg42)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg42);;
  occaKernelRunN(kernel, 43, args);
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

  occaType_t *args[44] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);
  if(((occaType) arg40)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg40)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg40);
  if(((occaType) arg41)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg41)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg41);
  if(((occaType) arg42)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg42)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg42);
  if(((occaType) arg43)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg43)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg43);;
  occaKernelRunN(kernel, 44, args);
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

  occaType_t *args[45] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);
  if(((occaType) arg40)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg40)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg40);
  if(((occaType) arg41)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg41)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg41);
  if(((occaType) arg42)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg42)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg42);
  if(((occaType) arg43)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg43)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg43);
  if(((occaType) arg44)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg44)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg44);;
  occaKernelRunN(kernel, 45, args);
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

  occaType_t *args[46] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr, ((occaType) arg45)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);
  if(((occaType) arg40)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg40)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg40);
  if(((occaType) arg41)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg41)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg41);
  if(((occaType) arg42)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg42)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg42);
  if(((occaType) arg43)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg43)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg43);
  if(((occaType) arg44)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg44)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg44);
  if(((occaType) arg45)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg45)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg45);;
  occaKernelRunN(kernel, 46, args);
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

  occaType_t *args[47] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr, ((occaType) arg45)->ptr, ((occaType) arg46)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);
  if(((occaType) arg40)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg40)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg40);
  if(((occaType) arg41)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg41)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg41);
  if(((occaType) arg42)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg42)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg42);
  if(((occaType) arg43)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg43)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg43);
  if(((occaType) arg44)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg44)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg44);
  if(((occaType) arg45)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg45)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg45);
  if(((occaType) arg46)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg46)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg46);;
  occaKernelRunN(kernel, 47, args);
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

  occaType_t *args[48] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr, ((occaType) arg45)->ptr, ((occaType) arg46)->ptr, ((occaType) arg47)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);
  if(((occaType) arg40)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg40)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg40);
  if(((occaType) arg41)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg41)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg41);
  if(((occaType) arg42)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg42)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg42);
  if(((occaType) arg43)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg43)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg43);
  if(((occaType) arg44)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg44)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg44);
  if(((occaType) arg45)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg45)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg45);
  if(((occaType) arg46)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg46)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg46);
  if(((occaType) arg47)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg47)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg47);;
  occaKernelRunN(kernel, 48, args);
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

  occaType_t *args[49] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr, ((occaType) arg45)->ptr, ((occaType) arg46)->ptr, ((occaType) arg47)->ptr, ((occaType) arg48)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);
  if(((occaType) arg40)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg40)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg40);
  if(((occaType) arg41)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg41)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg41);
  if(((occaType) arg42)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg42)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg42);
  if(((occaType) arg43)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg43)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg43);
  if(((occaType) arg44)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg44)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg44);
  if(((occaType) arg45)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg45)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg45);
  if(((occaType) arg46)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg46)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg46);
  if(((occaType) arg47)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg47)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg47);
  if(((occaType) arg48)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg48)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg48);;
  occaKernelRunN(kernel, 49, args);
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

  occaType_t *args[50] = { ((occaType) arg0)->ptr, ((occaType) arg1)->ptr, ((occaType) arg2)->ptr, ((occaType) arg3)->ptr, ((occaType) arg4)->ptr, ((occaType) arg5)->ptr, ((occaType) arg6)->ptr, ((occaType) arg7)->ptr, ((occaType) arg8)->ptr, ((occaType) arg9)->ptr, ((occaType) arg10)->ptr, ((occaType) arg11)->ptr, ((occaType) arg12)->ptr, ((occaType) arg13)->ptr, ((occaType) arg14)->ptr, ((occaType) arg15)->ptr, ((occaType) arg16)->ptr, ((occaType) arg17)->ptr, ((occaType) arg18)->ptr, ((occaType) arg19)->ptr, ((occaType) arg20)->ptr, ((occaType) arg21)->ptr, ((occaType) arg22)->ptr, ((occaType) arg23)->ptr, ((occaType) arg24)->ptr, ((occaType) arg25)->ptr, ((occaType) arg26)->ptr, ((occaType) arg27)->ptr, ((occaType) arg28)->ptr, ((occaType) arg29)->ptr, ((occaType) arg30)->ptr, ((occaType) arg31)->ptr, ((occaType) arg32)->ptr, ((occaType) arg33)->ptr, ((occaType) arg34)->ptr, ((occaType) arg35)->ptr, ((occaType) arg36)->ptr, ((occaType) arg37)->ptr, ((occaType) arg38)->ptr, ((occaType) arg39)->ptr, ((occaType) arg40)->ptr, ((occaType) arg41)->ptr, ((occaType) arg42)->ptr, ((occaType) arg43)->ptr, ((occaType) arg44)->ptr, ((occaType) arg45)->ptr, ((occaType) arg46)->ptr, ((occaType) arg47)->ptr, ((occaType) arg48)->ptr, ((occaType) arg49)->ptr };
  if(((occaType) arg0)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg0)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg0);
  if(((occaType) arg1)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg1)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg1);
  if(((occaType) arg2)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg2)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg2);
  if(((occaType) arg3)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg3)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg3);
  if(((occaType) arg4)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg4)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg4);
  if(((occaType) arg5)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg5)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg5);
  if(((occaType) arg6)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg6)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg6);
  if(((occaType) arg7)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg7)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg7);
  if(((occaType) arg8)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg8)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg8);
  if(((occaType) arg9)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg9)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg9);
  if(((occaType) arg10)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg10)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg10);
  if(((occaType) arg11)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg11)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg11);
  if(((occaType) arg12)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg12)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg12);
  if(((occaType) arg13)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg13)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg13);
  if(((occaType) arg14)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg14)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg14);
  if(((occaType) arg15)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg15)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg15);
  if(((occaType) arg16)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg16)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg16);
  if(((occaType) arg17)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg17)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg17);
  if(((occaType) arg18)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg18)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg18);
  if(((occaType) arg19)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg19)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg19);
  if(((occaType) arg20)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg20)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg20);
  if(((occaType) arg21)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg21)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg21);
  if(((occaType) arg22)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg22)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg22);
  if(((occaType) arg23)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg23)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg23);
  if(((occaType) arg24)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg24)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg24);
  if(((occaType) arg25)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg25)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg25);
  if(((occaType) arg26)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg26)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg26);
  if(((occaType) arg27)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg27)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg27);
  if(((occaType) arg28)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg28)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg28);
  if(((occaType) arg29)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg29)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg29);
  if(((occaType) arg30)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg30)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg30);
  if(((occaType) arg31)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg31)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg31);
  if(((occaType) arg32)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg32)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg32);
  if(((occaType) arg33)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg33)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg33);
  if(((occaType) arg34)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg34)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg34);
  if(((occaType) arg35)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg35)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg35);
  if(((occaType) arg36)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg36)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg36);
  if(((occaType) arg37)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg37)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg37);
  if(((occaType) arg38)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg38)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg38);
  if(((occaType) arg39)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg39)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg39);
  if(((occaType) arg40)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg40)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg40);
  if(((occaType) arg41)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg41)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg41);
  if(((occaType) arg42)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg42)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg42);
  if(((occaType) arg43)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg43)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg43);
  if(((occaType) arg44)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg44)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg44);
  if(((occaType) arg45)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg45)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg45);
  if(((occaType) arg46)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg46)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg46);
  if(((occaType) arg47)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg47)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg47);
  if(((occaType) arg48)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg48)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg48);
  if(((occaType) arg49)->ptr->type != OCCA_TYPE_MEMORY && ((occaType) arg49)->ptr->type != OCCA_TYPE_PTR) delete ((occaType) arg49);;
  occaKernelRunN(kernel, 50, args);
}
