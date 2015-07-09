#include "_C_occa.h"

//---[ Globals & Flags ]----------------
static PyObject* py_occaSetVerboseCompilation(PyObject *self, PyObject *args){
  int value;

  if(!PyArg_ParseTuple(args, "i", &value))
    return NULL;

  occaSetVerboseCompilation(value);

  return Py_None;
}
//======================================

//----[ Background Device ]-------------
//  |---[ Device ]----------------------
static PyObject* py_occaSetDevice(PyObject *self, PyObject *args){
  occaDevice device;

  if(!PyArg_ParseTuple(args, "n", &device))
    return NULL;

  occaSetDevice(device);

  return Py_None;
}

static PyObject* py_occaSetDeviceFromInfo(PyObject *self, PyObject *args){
  const char *infos;

  if(!PyArg_ParseTuple(args, "s", &infos))
    return NULL;

  occaSetDeviceFromInfo(infos);

  return Py_None;
}

static PyObject* py_occaGetCurrentDevice(PyObject *self, PyObject *args){
  occaDevice device = occaGetCurrentDevice();

  return PyLong_FromVoidPtr(device);
}

static PyObject* py_occaSetCompiler(PyObject *self, PyObject *args){
  const char *compiler;

  if(!PyArg_ParseTuple(args, "s", &compiler))
    return NULL;

  occaSetCompiler(compiler);

  return Py_None;
}

static PyObject* py_occaSetCompilerEnvScript(PyObject *self, PyObject *args){
  const char *compilerEnvScript;

  if(!PyArg_ParseTuple(args, "s", &compilerEnvScript))
    return NULL;

  occaSetCompilerEnvScript(compilerEnvScript);

  return Py_None;
}

static PyObject* py_occaSetCompilerFlags(PyObject *self, PyObject *args){
  const char *compilerFlags;

  if(!PyArg_ParseTuple(args, "s", &compilerFlags))
    return NULL;

  occaSetCompilerFlags(compilerFlags);

  return Py_None;
}

static PyObject* py_occaGetCompiler(PyObject *self, PyObject *args){
  const char *compiler = occaGetCompiler();
  return PyString_FromString(compiler);
}

static PyObject* py_occaGetCompilerEnvScript(PyObject *self, PyObject *args){
  const char *compilerEnvScript = occaGetCompilerEnvScript();
  return PyString_FromString(compilerEnvScript);
}

static PyObject* py_occaGetCompilerFlags(PyObject *self, PyObject *args){
  const char *compilerFlags = occaGetCompilerFlags();
  return PyString_FromString(compilerFlags);
}

static PyObject* py_occaFlush(PyObject *self, PyObject *args){
  occaFlush();

  return Py_None;
}

static PyObject* py_occaFinish(PyObject *self, PyObject *args){
  occaFinish();

  return Py_None;
}

/* static PyObject* py_occaWaitFor(PyObject *self, PyObject *args){ */
/*   occaStreamTag tag; */

/*   if(!PyArg_ParseTuple(args, "n", &tag)) */
/*     return NULL; */

/*   occaWaitFor(tag); */

/*   return Py_None; */
/* } */

static PyObject* py_occaCreateStream(PyObject *self, PyObject *args){
  occaStream stream = occaCreateStream();

  return PyLong_FromVoidPtr(stream);
}

static PyObject* py_occaGetStream(PyObject *self, PyObject *args){
  occaStream stream = occaGetStream();

  return PyLong_FromVoidPtr(stream);
}

static PyObject* py_occaSetStream(PyObject *self, PyObject *args){
  occaStream stream;

  if(!PyArg_ParseTuple(args, "n", &stream))
    return NULL;

  occaSetStream(stream);

  return Py_None;
}

static PyObject* py_occaWrapStream(PyObject *self, PyObject *args){
  occaStream stream;
  void *handle;

  if(!PyArg_ParseTuple(args, "n", &handle))
    return NULL;

  stream = occaWrapStream(handle);

  return PyLong_FromVoidPtr(stream);
}

/* static PyObject* py_occaTagStream(PyObject *self, PyObject *args){ */
/*   occaStreamTag tag = occaTagStream(); */

/*   return PyLong_FromVoidPtr(tag); */
/* } */

//  |===================================

//  |---[ Kernel ]----------------------
static PyObject* py_occaBuildKernel(PyObject *self, PyObject *args){
  const char *str, *functionName;
  occaKernelInfo *kInfo;

  if(!PyArg_ParseTuple(args, "ssn", &str, &functionName, &kInfo))
    return NULL;

  occaKernel kernel = occaBuildKernel(str, functionName, kInfo);

  return PyLong_FromVoidPtr(kernel);
}

static PyObject* py_occaBuildKernelFromSource(PyObject *self, PyObject *args){
  const char *filename, *functionName;
  occaKernelInfo *kInfo;

  if(!PyArg_ParseTuple(args, "ssn", &filename, &functionName, &kInfo))
    return NULL;

  occaKernel kernel = occaBuildKernelFromSource(filename, functionName, kInfo);

  return PyLong_FromVoidPtr(kernel);
}

static PyObject* py_occaBuildKernelFromString(PyObject *self, PyObject *args){
  const char *filename, *functionName, *language;
  occaKernelInfo *kInfo;
  occaKernel kernel;

  if(!PyArg_ParseTuple(args, "sssn", &filename, &functionName, &kInfo, &language))
    return NULL;

  if(strcmp(language, "OFL") == 0)
    kernel = occaBuildKernelFromSource(filename, functionName, kInfo, occaUsingOFL);
  else if(strcmp(language, "Native") == 0)
    kernel = occaBuildKernelFromSource(filename, functionName, kInfo, occaUsingNative);
  else
    kernel = occaBuildKernelFromSource(filename, functionName, kInfo, occaUsingOKL);

  return PyLong_FromVoidPtr(kernel);
}

static PyObject* py_occaBuildKernelFromBinary(PyObject *self, PyObject *args){
  const char *filename, *functionName;

  if(!PyArg_ParseTuple(args, "ss", &filename, &functionName))
    return NULL;

  occaKernel kernel = occaBuildKernelFromBinary(filename, functionName);

  return PyLong_FromVoidPtr(kernel);
}

static PyObject* py_occaBuildKernelFromLoopy(PyObject *self, PyObject *args){
  const char *filename, *functionName;
  occaKernelInfo *kInfo;

  if(!PyArg_ParseTuple(args, "ssn", &filename, &functionName, &kInfo))
    return NULL;

  occaKernel kernel = occaBuildKernelFromLoopy(filename, functionName, kInfo);

  return PyLong_FromVoidPtr(kernel);
}

static PyObject* py_occaBuildKernelFromFloopy(PyObject *self, PyObject *args){
  const char *filename, *functionName;
  occaKernelInfo *kInfo;

  if(!PyArg_ParseTuple(args, "ssn", &filename, &functionName, &kInfo))
    return NULL;

  occaKernel kernel = occaBuildKernelFromFloopy(filename, functionName, kInfo);

  return PyLong_FromVoidPtr(kernel);
}

//  |===================================

//  |---[ Memory ]----------------------
static PyObject* py_occaWrapMemory(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaWrapManagedMemory(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaMalloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaManagedAlloc(PyObject *self, PyObject *args){
  size_t entries;
  int typeSize, typenum;

  if(!PyArg_ParseTuple(args, "nii", &entries, &typeSize, &typenum))
    return NULL;

  const size_t bytes = (entries * typeSize);

  int nd         = 1;
  npy_intp *dims = (npy_intp*) malloc(1*sizeof(npy_intp));
  dims[0]        = entries;
  void *data     = occaManagedAlloc(bytes, NULL);

  return PyArray_SimpleNewFromData(nd, dims, typenum, data);
}

static PyObject* py_occaUvaAlloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaManagedUvaAlloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaMappedAlloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaManagedMappedAlloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

//  |===================================
//======================================

//---[ Device ]-------------------------
static PyObject* py_occaPrintAvailableDevices(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaCreateDeviceInfo(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceInfoAppend(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceInfoAppendType(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceInfoFree(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaCreateDevice(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaCreateDeviceFromInfo(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceMode(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceSetCompiler(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceSetCompilerFlags(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceBytesAllocated(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceBuildKernel(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceBuildKernelFromSource(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceBuildKernelFromString(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceBuildKernelFromBinary(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceBuildKernelFromLoopy(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceBuildKernelFromFloopy(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceMalloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceManagedAlloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceUvaAlloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceManagedUvaAlloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceMappedAlloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceManagedMappedAlloc(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceFlush(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceFinish(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceCreateStream(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceGetStream(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceSetStream(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

/* static PyObject* py_occaDeviceTagStream(PyObject *self, PyObject *args){ */
/*   if(!PyArg_ParseTuple(args, "")) */
/*     return NULL; */

/*   return Py_None; */
/* } */

/* static PyObject* py_occaDeviceWaitForTag(PyObject *self, PyObject *args){ */
/*   if(!PyArg_ParseTuple(args, "")) */
/*     return NULL; */

/*   return Py_None; */
/* } */

/* static PyObject* py_occaDeviceTimeBetweenTags(PyObject *self, PyObject *args){ */
/*   if(!PyArg_ParseTuple(args, "")) */
/*     return NULL; */

/*   return Py_None; */
/* } */

static PyObject* py_occaDeviceStreamFree(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaDeviceFree(PyObject *self, PyObject *args){
  occaDevice device;

  if(!PyArg_ParseTuple(args, "n", &device))
    return NULL;

  occaDeviceFree(device);

  return Py_None;
}
//======================================

//---[ Kernel ]-------------------------
static PyObject* py_occaKernelMode(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaKernelName(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaKernelGetDevice(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaCreateArgumentList(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaArgumentListClear(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaArgumentListFree(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaArgumentListAddArg(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaKernelRun_(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaCreateKernelInfo(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaKernelInfoAddDefine(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaKernelInfoAddInclude(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaKernelFree(PyObject *self, PyObject *args){
  occaKernel kernel;

  if(!PyArg_ParseTuple(args, "n", &kernel))
    return NULL;

  occaKernelFree(kernel);

  return Py_None;
}
//======================================

//---[ Memory ]-------------------------
static PyObject* py_occaMemoryMode(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaMemoryGetMemoryHandle(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaMemoryGetMappedPointer(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaMemoryGetTextureHandle(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaMemcpy(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaCopyMemToMem(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaCopyPtrToMem(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaCopyMemToPtr(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaAsyncCopyMemToMem(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaAsyncCopyPtrToMem(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaAsyncCopyMemToPtr(PyObject *self, PyObject *args){
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_None;
}

static PyObject* py_occaMemoryFree(PyObject *self, PyObject *args){
  occaMemory memory;

  if(!PyArg_ParseTuple(args, "n", &memory))
    return NULL;

  occaMemoryFree(memory);

  return Py_None;
}
//======================================