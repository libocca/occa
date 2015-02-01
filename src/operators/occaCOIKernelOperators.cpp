  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 1*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[0],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 2*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[1],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 3*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[2],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 4*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[3],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 5*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[4],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 6*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[5],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 7*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[6],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 8*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[7],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 9*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[8],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 10*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[9],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 11*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[10],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 12*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[11],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 13*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[12],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 14*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[13],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 15*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[14],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 16*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[15],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 17*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[16],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 18*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[17],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 19*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[18],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 20*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[19],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 21*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[20],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 22*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[21],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 23*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[22],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 24*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[23],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 25*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[24],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 26*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[25],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 27*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[26],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 28*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[27],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 29*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[28],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
                      const kernelArg &arg3,  const kernelArg &arg4,  const kernelArg &arg5, 
                      const kernelArg &arg6,  const kernelArg &arg7,  const kernelArg &arg8, 
                      const kernelArg &arg9,  const kernelArg &arg10,  const kernelArg &arg11, 
                      const kernelArg &arg12,  const kernelArg &arg13,  const kernelArg &arg14, 
                      const kernelArg &arg15,  const kernelArg &arg16,  const kernelArg &arg17, 
                      const kernelArg &arg18,  const kernelArg &arg19,  const kernelArg &arg20, 
                      const kernelArg &arg21,  const kernelArg &arg22,  const kernelArg &arg23, 
                      const kernelArg &arg24,  const kernelArg &arg25,  const kernelArg &arg26, 
                      const kernelArg &arg27,  const kernelArg &arg28,  const kernelArg &arg29){
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 30*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[29],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 31*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[30],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 32*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[31],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 33*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[32],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 34*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[33],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 35*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[34],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 36*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[35],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 37*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[36],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 38*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[37],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 39*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[38],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 40*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[39],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 41*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }
    if(arg40.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg40.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg40.arg), arg40.size);
      hostPos += arg40.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[40],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 42*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }
    if(arg40.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg40.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg40.arg), arg40.size);
      hostPos += arg40.size;
    }
    if(arg41.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg41.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg41.arg), arg41.size);
      hostPos += arg41.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[41],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 43*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }
    if(arg40.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg40.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg40.arg), arg40.size);
      hostPos += arg40.size;
    }
    if(arg41.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg41.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg41.arg), arg41.size);
      hostPos += arg41.size;
    }
    if(arg42.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg42.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg42.arg), arg42.size);
      hostPos += arg42.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[42],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 44*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }
    if(arg40.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg40.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg40.arg), arg40.size);
      hostPos += arg40.size;
    }
    if(arg41.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg41.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg41.arg), arg41.size);
      hostPos += arg41.size;
    }
    if(arg42.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg42.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg42.arg), arg42.size);
      hostPos += arg42.size;
    }
    if(arg43.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg43.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg43.arg), arg43.size);
      hostPos += arg43.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[43],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 45*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }
    if(arg40.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg40.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg40.arg), arg40.size);
      hostPos += arg40.size;
    }
    if(arg41.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg41.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg41.arg), arg41.size);
      hostPos += arg41.size;
    }
    if(arg42.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg42.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg42.arg), arg42.size);
      hostPos += arg42.size;
    }
    if(arg43.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg43.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg43.arg), arg43.size);
      hostPos += arg43.size;
    }
    if(arg44.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg44.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg44.arg), arg44.size);
      hostPos += arg44.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[44],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 46*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }
    if(arg40.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg40.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg40.arg), arg40.size);
      hostPos += arg40.size;
    }
    if(arg41.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg41.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg41.arg), arg41.size);
      hostPos += arg41.size;
    }
    if(arg42.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg42.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg42.arg), arg42.size);
      hostPos += arg42.size;
    }
    if(arg43.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg43.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg43.arg), arg43.size);
      hostPos += arg43.size;
    }
    if(arg44.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg44.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg44.arg), arg44.size);
      hostPos += arg44.size;
    }
    if(arg45.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg45.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg45.arg), arg45.size);
      hostPos += arg45.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[45],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 47*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }
    if(arg40.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg40.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg40.arg), arg40.size);
      hostPos += arg40.size;
    }
    if(arg41.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg41.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg41.arg), arg41.size);
      hostPos += arg41.size;
    }
    if(arg42.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg42.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg42.arg), arg42.size);
      hostPos += arg42.size;
    }
    if(arg43.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg43.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg43.arg), arg43.size);
      hostPos += arg43.size;
    }
    if(arg44.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg44.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg44.arg), arg44.size);
      hostPos += arg44.size;
    }
    if(arg45.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg45.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg45.arg), arg45.size);
      hostPos += arg45.size;
    }
    if(arg46.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg46.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg46.arg), arg46.size);
      hostPos += arg46.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[46],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 48*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }
    if(arg40.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg40.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg40.arg), arg40.size);
      hostPos += arg40.size;
    }
    if(arg41.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg41.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg41.arg), arg41.size);
      hostPos += arg41.size;
    }
    if(arg42.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg42.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg42.arg), arg42.size);
      hostPos += arg42.size;
    }
    if(arg43.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg43.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg43.arg), arg43.size);
      hostPos += arg43.size;
    }
    if(arg44.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg44.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg44.arg), arg44.size);
      hostPos += arg44.size;
    }
    if(arg45.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg45.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg45.arg), arg45.size);
      hostPos += arg45.size;
    }
    if(arg46.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg46.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg46.arg), arg46.size);
      hostPos += arg46.size;
    }
    if(arg47.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg47.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg47.arg), arg47.size);
      hostPos += arg47.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[47],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 49*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }
    if(arg40.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg40.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg40.arg), arg40.size);
      hostPos += arg40.size;
    }
    if(arg41.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg41.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg41.arg), arg41.size);
      hostPos += arg41.size;
    }
    if(arg42.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg42.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg42.arg), arg42.size);
      hostPos += arg42.size;
    }
    if(arg43.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg43.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg43.arg), arg43.size);
      hostPos += arg43.size;
    }
    if(arg44.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg44.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg44.arg), arg44.size);
      hostPos += arg44.size;
    }
    if(arg45.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg45.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg45.arg), arg45.size);
      hostPos += arg45.size;
    }
    if(arg46.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg46.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg46.arg), arg46.size);
      hostPos += arg46.size;
    }
    if(arg47.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg47.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg47.arg), arg47.size);
      hostPos += arg47.size;
    }
    if(arg48.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg48.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg48.arg), arg48.size);
      hostPos += arg48.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[48],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }

  template <>
  void kernel_t<COI>::operator () (const kernelArg &arg0,  const kernelArg &arg1,  const kernelArg &arg2, 
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
    COIKernelData_t &data_ = *((COIKernelData_t*) data);
    COIDeviceData_t &dData = *((COIDeviceData_t*) ((device_t<COI>*) dHandle)->data);
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z;
    occaKernelArgs[1] = outer.y;
    occaKernelArgs[2] = outer.x;
    occaKernelArgs[3] = inner.z;
    occaKernelArgs[4] = inner.y;
    occaKernelArgs[5] = inner.x;

    uintptr_t kSize = sizeof(data_.kernel);

    ::memcpy(&(data_.hostArgv[0])    , &(data_.kernel)     , kSize);
    ::memcpy(&(data_.hostArgv[kSize]), &(occaKernelArgs[0]), 6*sizeof(int));

    int hostPos = kSize + 6*sizeof(int) + 50*sizeof(int);

    int typePos = 0;
    int devicePos = 0;

    int *typePtr = (int*) (data_.hostArgv + kSize + 6*sizeof(int));

    if(arg0.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg0.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg0.arg), arg0.size);
      hostPos += arg0.size;
    }
    if(arg1.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg1.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg1.arg), arg1.size);
      hostPos += arg1.size;
    }
    if(arg2.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg2.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg2.arg), arg2.size);
      hostPos += arg2.size;
    }
    if(arg3.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg3.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg3.arg), arg3.size);
      hostPos += arg3.size;
    }
    if(arg4.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg4.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg4.arg), arg4.size);
      hostPos += arg4.size;
    }
    if(arg5.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg5.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg5.arg), arg5.size);
      hostPos += arg5.size;
    }
    if(arg6.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg6.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg6.arg), arg6.size);
      hostPos += arg6.size;
    }
    if(arg7.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg7.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg7.arg), arg7.size);
      hostPos += arg7.size;
    }
    if(arg8.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg8.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg8.arg), arg8.size);
      hostPos += arg8.size;
    }
    if(arg9.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg9.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg9.arg), arg9.size);
      hostPos += arg9.size;
    }
    if(arg10.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg10.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg10.arg), arg10.size);
      hostPos += arg10.size;
    }
    if(arg11.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg11.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg11.arg), arg11.size);
      hostPos += arg11.size;
    }
    if(arg12.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg12.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg12.arg), arg12.size);
      hostPos += arg12.size;
    }
    if(arg13.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg13.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg13.arg), arg13.size);
      hostPos += arg13.size;
    }
    if(arg14.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg14.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg14.arg), arg14.size);
      hostPos += arg14.size;
    }
    if(arg15.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg15.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg15.arg), arg15.size);
      hostPos += arg15.size;
    }
    if(arg16.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg16.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg16.arg), arg16.size);
      hostPos += arg16.size;
    }
    if(arg17.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg17.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg17.arg), arg17.size);
      hostPos += arg17.size;
    }
    if(arg18.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg18.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg18.arg), arg18.size);
      hostPos += arg18.size;
    }
    if(arg19.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg19.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg19.arg), arg19.size);
      hostPos += arg19.size;
    }
    if(arg20.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg20.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg20.arg), arg20.size);
      hostPos += arg20.size;
    }
    if(arg21.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg21.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg21.arg), arg21.size);
      hostPos += arg21.size;
    }
    if(arg22.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg22.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg22.arg), arg22.size);
      hostPos += arg22.size;
    }
    if(arg23.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg23.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg23.arg), arg23.size);
      hostPos += arg23.size;
    }
    if(arg24.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg24.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg24.arg), arg24.size);
      hostPos += arg24.size;
    }
    if(arg25.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg25.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg25.arg), arg25.size);
      hostPos += arg25.size;
    }
    if(arg26.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg26.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg26.arg), arg26.size);
      hostPos += arg26.size;
    }
    if(arg27.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg27.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg27.arg), arg27.size);
      hostPos += arg27.size;
    }
    if(arg28.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg28.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg28.arg), arg28.size);
      hostPos += arg28.size;
    }
    if(arg29.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg29.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg29.arg), arg29.size);
      hostPos += arg29.size;
    }
    if(arg30.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg30.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg30.arg), arg30.size);
      hostPos += arg30.size;
    }
    if(arg31.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg31.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg31.arg), arg31.size);
      hostPos += arg31.size;
    }
    if(arg32.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg32.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg32.arg), arg32.size);
      hostPos += arg32.size;
    }
    if(arg33.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg33.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg33.arg), arg33.size);
      hostPos += arg33.size;
    }
    if(arg34.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg34.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg34.arg), arg34.size);
      hostPos += arg34.size;
    }
    if(arg35.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg35.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg35.arg), arg35.size);
      hostPos += arg35.size;
    }
    if(arg36.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg36.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg36.arg), arg36.size);
      hostPos += arg36.size;
    }
    if(arg37.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg37.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg37.arg), arg37.size);
      hostPos += arg37.size;
    }
    if(arg38.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg38.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg38.arg), arg38.size);
      hostPos += arg38.size;
    }
    if(arg39.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg39.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg39.arg), arg39.size);
      hostPos += arg39.size;
    }
    if(arg40.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg40.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg40.arg), arg40.size);
      hostPos += arg40.size;
    }
    if(arg41.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg41.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg41.arg), arg41.size);
      hostPos += arg41.size;
    }
    if(arg42.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg42.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg42.arg), arg42.size);
      hostPos += arg42.size;
    }
    if(arg43.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg43.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg43.arg), arg43.size);
      hostPos += arg43.size;
    }
    if(arg44.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg44.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg44.arg), arg44.size);
      hostPos += arg44.size;
    }
    if(arg45.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg45.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg45.arg), arg45.size);
      hostPos += arg45.size;
    }
    if(arg46.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg46.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg46.arg), arg46.size);
      hostPos += arg46.size;
    }
    if(arg47.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg47.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg47.arg), arg47.size);
      hostPos += arg47.size;
    }
    if(arg48.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg48.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg48.arg), arg48.size);
      hostPos += arg48.size;
    }
    if(arg49.pointer){
      typePtr[typePos++] = (int) ((1 << 31) + devicePos);
      data_.deviceArgv[devicePos] = *((coiMemory*) arg49.data());
      data_.deviceFlags[devicePos] = COI_SINK_WRITE;
      ++devicePos;
    }
    else{
      typePtr[typePos++] = (int) ((0 << 31) + hostPos);
      ::memcpy(&(data_.hostArgv[hostPos]), &(arg49.arg), arg49.size);
      hostPos += arg49.size;
    }

    coiStream &stream = *((coiStream*) dHandle->currentStream);

    COIPipelineRunFunction(stream.handle,
                           dData.kernelWrapper[49],
                           devicePos,
                           (const coiMemory*) data_.deviceArgv,
                           (const coiMemoryFlags*) data_.deviceFlags,
                           false,
                           __null,
                           data_.hostArgv,
                           hostPos,
                           __null,
                           0,
                           &(stream.lastEvent));
  }