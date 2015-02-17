  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel1);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel1(PthreadKernelArg_t &args){
    functionPointer1 tmpKernel = (functionPointer1) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel2);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel2(PthreadKernelArg_t &args){
    functionPointer2 tmpKernel = (functionPointer2) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel3);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel3(PthreadKernelArg_t &args){
    functionPointer3 tmpKernel = (functionPointer3) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel4);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel4(PthreadKernelArg_t &args){
    functionPointer4 tmpKernel = (functionPointer4) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel5);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel5(PthreadKernelArg_t &args){
    functionPointer5 tmpKernel = (functionPointer5) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel6);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel6(PthreadKernelArg_t &args){
    functionPointer6 tmpKernel = (functionPointer6) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel7);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel7(PthreadKernelArg_t &args){
    functionPointer7 tmpKernel = (functionPointer7) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel8);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel8(PthreadKernelArg_t &args){
    functionPointer8 tmpKernel = (functionPointer8) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel9);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel9(PthreadKernelArg_t &args){
    functionPointer9 tmpKernel = (functionPointer9) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel10);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel10(PthreadKernelArg_t &args){
    functionPointer10 tmpKernel = (functionPointer10) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel11);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel11(PthreadKernelArg_t &args){
    functionPointer11 tmpKernel = (functionPointer11) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel12);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel12(PthreadKernelArg_t &args){
    functionPointer12 tmpKernel = (functionPointer12) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel13);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel13(PthreadKernelArg_t &args){
    functionPointer13 tmpKernel = (functionPointer13) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel14);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel14(PthreadKernelArg_t &args){
    functionPointer14 tmpKernel = (functionPointer14) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel15);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel15(PthreadKernelArg_t &args){
    functionPointer15 tmpKernel = (functionPointer15) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel16);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel16(PthreadKernelArg_t &args){
    functionPointer16 tmpKernel = (functionPointer16) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel17);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel17(PthreadKernelArg_t &args){
    functionPointer17 tmpKernel = (functionPointer17) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data());

    delete &args;
  }

  template <>
  void kernel_t<Pthreads>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17){
    PthreadsKernelData_t &data_ = *((PthreadsKernelData_t*) data);
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel18);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel18(PthreadKernelArg_t &args){
    functionPointer18 tmpKernel = (functionPointer18) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel19);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel19(PthreadKernelArg_t &args){
    functionPointer19 tmpKernel = (functionPointer19) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel20);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel20(PthreadKernelArg_t &args){
    functionPointer20 tmpKernel = (functionPointer20) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel21);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel21(PthreadKernelArg_t &args){
    functionPointer21 tmpKernel = (functionPointer21) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel22);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel22(PthreadKernelArg_t &args){
    functionPointer22 tmpKernel = (functionPointer22) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel23);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel23(PthreadKernelArg_t &args){
    functionPointer23 tmpKernel = (functionPointer23) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel24);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel24(PthreadKernelArg_t &args){
    functionPointer24 tmpKernel = (functionPointer24) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel25);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel25(PthreadKernelArg_t &args){
    functionPointer25 tmpKernel = (functionPointer25) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel26);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel26(PthreadKernelArg_t &args){
    functionPointer26 tmpKernel = (functionPointer26) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel27);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel27(PthreadKernelArg_t &args){
    functionPointer27 tmpKernel = (functionPointer27) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel28);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel28(PthreadKernelArg_t &args){
    functionPointer28 tmpKernel = (functionPointer28) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel29);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel29(PthreadKernelArg_t &args){
    functionPointer29 tmpKernel = (functionPointer29) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel30);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel30(PthreadKernelArg_t &args){
    functionPointer30 tmpKernel = (functionPointer30) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel31);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel31(PthreadKernelArg_t &args){
    functionPointer31 tmpKernel = (functionPointer31) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel32);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel32(PthreadKernelArg_t &args){
    functionPointer32 tmpKernel = (functionPointer32) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel33);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel33(PthreadKernelArg_t &args){
    functionPointer33 tmpKernel = (functionPointer33) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel34);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel34(PthreadKernelArg_t &args){
    functionPointer34 tmpKernel = (functionPointer34) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel35);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel35(PthreadKernelArg_t &args){
    functionPointer35 tmpKernel = (functionPointer35) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel36);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel36(PthreadKernelArg_t &args){
    functionPointer36 tmpKernel = (functionPointer36) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel37);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel37(PthreadKernelArg_t &args){
    functionPointer37 tmpKernel = (functionPointer37) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel38);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel38(PthreadKernelArg_t &args){
    functionPointer38 tmpKernel = (functionPointer38) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel39);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel39(PthreadKernelArg_t &args){
    functionPointer39 tmpKernel = (functionPointer39) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel40);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel40(PthreadKernelArg_t &args){
    functionPointer40 tmpKernel = (functionPointer40) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;
    args->args[40] = arg40;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel41);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel41(PthreadKernelArg_t &args){
    functionPointer41 tmpKernel = (functionPointer41) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data(),
              args.args[40].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;
    args->args[40] = arg40;
    args->args[41] = arg41;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel42);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel42(PthreadKernelArg_t &args){
    functionPointer42 tmpKernel = (functionPointer42) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data(),
              args.args[40].data(),
              args.args[41].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;
    args->args[40] = arg40;
    args->args[41] = arg41;
    args->args[42] = arg42;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel43);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel43(PthreadKernelArg_t &args){
    functionPointer43 tmpKernel = (functionPointer43) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data(),
              args.args[40].data(),
              args.args[41].data(),
              args.args[42].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;
    args->args[40] = arg40;
    args->args[41] = arg41;
    args->args[42] = arg42;
    args->args[43] = arg43;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel44);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel44(PthreadKernelArg_t &args){
    functionPointer44 tmpKernel = (functionPointer44) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data(),
              args.args[40].data(),
              args.args[41].data(),
              args.args[42].data(),
              args.args[43].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;
    args->args[40] = arg40;
    args->args[41] = arg41;
    args->args[42] = arg42;
    args->args[43] = arg43;
    args->args[44] = arg44;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel45);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel45(PthreadKernelArg_t &args){
    functionPointer45 tmpKernel = (functionPointer45) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data(),
              args.args[40].data(),
              args.args[41].data(),
              args.args[42].data(),
              args.args[43].data(),
              args.args[44].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;
    args->args[40] = arg40;
    args->args[41] = arg41;
    args->args[42] = arg42;
    args->args[43] = arg43;
    args->args[44] = arg44;
    args->args[45] = arg45;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel46);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel46(PthreadKernelArg_t &args){
    functionPointer46 tmpKernel = (functionPointer46) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data(),
              args.args[40].data(),
              args.args[41].data(),
              args.args[42].data(),
              args.args[43].data(),
              args.args[44].data(),
              args.args[45].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;
    args->args[40] = arg40;
    args->args[41] = arg41;
    args->args[42] = arg42;
    args->args[43] = arg43;
    args->args[44] = arg44;
    args->args[45] = arg45;
    args->args[46] = arg46;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel47);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel47(PthreadKernelArg_t &args){
    functionPointer47 tmpKernel = (functionPointer47) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data(),
              args.args[40].data(),
              args.args[41].data(),
              args.args[42].data(),
              args.args[43].data(),
              args.args[44].data(),
              args.args[45].data(),
              args.args[46].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;
    args->args[40] = arg40;
    args->args[41] = arg41;
    args->args[42] = arg42;
    args->args[43] = arg43;
    args->args[44] = arg44;
    args->args[45] = arg45;
    args->args[46] = arg46;
    args->args[47] = arg47;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel48);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel48(PthreadKernelArg_t &args){
    functionPointer48 tmpKernel = (functionPointer48) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data(),
              args.args[40].data(),
              args.args[41].data(),
              args.args[42].data(),
              args.args[43].data(),
              args.args[44].data(),
              args.args[45].data(),
              args.args[46].data(),
              args.args[47].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;
    args->args[40] = arg40;
    args->args[41] = arg41;
    args->args[42] = arg42;
    args->args[43] = arg43;
    args->args[44] = arg44;
    args->args[45] = arg45;
    args->args[46] = arg46;
    args->args[47] = arg47;
    args->args[48] = arg48;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel49);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel49(PthreadKernelArg_t &args){
    functionPointer49 tmpKernel = (functionPointer49) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data(),
              args.args[40].data(),
              args.args[41].data(),
              args.args[42].data(),
              args.args[43].data(),
              args.args[44].data(),
              args.args[45].data(),
              args.args[46].data(),
              args.args[47].data(),
              args.args[48].data());

    delete &args;
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
    int pThreadCount = data_.pThreadCount;

    for(int p = 0; p < pThreadCount; ++p){
      PthreadKernelArg_t *args = new PthreadKernelArg_t;
      args->rank  = p;
      args->count = pThreadCount;

      args->kernelHandle = data_.handle;

      args->dims  = dims;
      args->inner = inner;
      args->outer = outer;

      args->args[0] = arg0;
    args->args[1] = arg1;
    args->args[2] = arg2;
    args->args[3] = arg3;
    args->args[4] = arg4;
    args->args[5] = arg5;
    args->args[6] = arg6;
    args->args[7] = arg7;
    args->args[8] = arg8;
    args->args[9] = arg9;
    args->args[10] = arg10;
    args->args[11] = arg11;
    args->args[12] = arg12;
    args->args[13] = arg13;
    args->args[14] = arg14;
    args->args[15] = arg15;
    args->args[16] = arg16;
    args->args[17] = arg17;
    args->args[18] = arg18;
    args->args[19] = arg19;
    args->args[20] = arg20;
    args->args[21] = arg21;
    args->args[22] = arg22;
    args->args[23] = arg23;
    args->args[24] = arg24;
    args->args[25] = arg25;
    args->args[26] = arg26;
    args->args[27] = arg27;
    args->args[28] = arg28;
    args->args[29] = arg29;
    args->args[30] = arg30;
    args->args[31] = arg31;
    args->args[32] = arg32;
    args->args[33] = arg33;
    args->args[34] = arg34;
    args->args[35] = arg35;
    args->args[36] = arg36;
    args->args[37] = arg37;
    args->args[38] = arg38;
    args->args[39] = arg39;
    args->args[40] = arg40;
    args->args[41] = arg41;
    args->args[42] = arg42;
    args->args[43] = arg43;
    args->args[44] = arg44;
    args->args[45] = arg45;
    args->args[46] = arg46;
    args->args[47] = arg47;
    args->args[48] = arg48;
    args->args[49] = arg49;

      pthread_mutex_lock(data_.kernelMutex);
      data_.kernelLaunch[p]->push(launchKernel50);
      data_.kernelArgs[p]->push(args);
      pthread_mutex_unlock(data_.kernelMutex);
    }

    pthread_mutex_lock(data_.pendingJobsMutex);
    *(data_.pendingJobs) += data_.pThreadCount;
    pthread_mutex_unlock(data_.pendingJobsMutex);
  }

  void launchKernel50(PthreadKernelArg_t &args){
    functionPointer50 tmpKernel = (functionPointer50) args.kernelHandle;

    int dp = args.dims - 1;
    occa::dim &outer = args.outer;
    occa::dim &inner = args.inner;

    occa::dim start(0,0,0), end(outer);

    int loops     = outer[dp]/args.count;
    int coolRanks = (outer[dp] - loops*args.count);

    if(args.rank < coolRanks){
      start[dp] = (args.rank)*(loops + 1);
      end[dp] = start[dp] + (loops + 1);
    }
    else{
      start[dp] = args.rank*loops + coolRanks;
      end[dp] = start[dp] + loops;
    }
    int occaKernelArgs[12];

    occaKernelArgs[0]  = outer.z;
    occaKernelArgs[1]  = outer.y;
    occaKernelArgs[2]  = outer.x;
    occaKernelArgs[3]  = inner.z;
    occaKernelArgs[4]  = inner.y;
    occaKernelArgs[5]  = inner.x;
    occaKernelArgs[6]  = start.z;
    occaKernelArgs[7]  = end.z;
    occaKernelArgs[8]  = start.y;
    occaKernelArgs[9]  = end.y;
    occaKernelArgs[10] = start.x;
    occaKernelArgs[11] = end.x;

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    tmpKernel(occaKernelArgs,
              occaInnerId0, occaInnerId1, occaInnerId2,
              args.args[0].data(),
              args.args[1].data(),
              args.args[2].data(),
              args.args[3].data(),
              args.args[4].data(),
              args.args[5].data(),
              args.args[6].data(),
              args.args[7].data(),
              args.args[8].data(),
              args.args[9].data(),
              args.args[10].data(),
              args.args[11].data(),
              args.args[12].data(),
              args.args[13].data(),
              args.args[14].data(),
              args.args[15].data(),
              args.args[16].data(),
              args.args[17].data(),
              args.args[18].data(),
              args.args[19].data(),
              args.args[20].data(),
              args.args[21].data(),
              args.args[22].data(),
              args.args[23].data(),
              args.args[24].data(),
              args.args[25].data(),
              args.args[26].data(),
              args.args[27].data(),
              args.args[28].data(),
              args.args[29].data(),
              args.args[30].data(),
              args.args[31].data(),
              args.args[32].data(),
              args.args[33].data(),
              args.args[34].data(),
              args.args[35].data(),
              args.args[36].data(),
              args.args[37].data(),
              args.args[38].data(),
              args.args[39].data(),
              args.args[40].data(),
              args.args[41].data(),
              args.args[42].data(),
              args.args[43].data(),
              args.args[44].data(),
              args.args[45].data(),
              args.args[46].data(),
              args.args[47].data(),
              args.args[48].data(),
              args.args[49].data());

    delete &args;
  }