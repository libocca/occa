#include "occa/python/_C_occa.h"

OCCA_START_EXTERN_C

#if OCCA_PY == 3
#  define STR_TO_PYOBJECT(charPtr) PyUnicode_FromString(charPtr)
#else
#  define STR_TO_PYOBJECT(charPtr) PyString_FromString(charPtr)
#endif

//---[ Globals & Flags ]----------------
static PyObject* py_occaSetVerboseCompilation(PyObject *self, PyObject *args) {
  int value;
  if (!PyArg_ParseTuple(args, "i", &value)) {
    return NULL;
  }
  occaSetVerboseCompilation(value);

  return Py_None;
}
//======================================

//---[ TypeCasting ]--------------------
static PyObject* py_occaPtr(PyObject *self, PyObject *args) {
  void *ptr;

  if (!PyArg_ParseTuple(args, "n", &ptr)) {
    return NULL;
  }
  occaType type = occaPtr(ptr);

  return PyLong_FromVoidPtr(type->ptr);
}

static PyObject* py_occaBool(PyObject *self, PyObject *args) {
  size_t value;

  if (!PyArg_ParseTuple(args, "n", &value)) {
    return NULL;
  }
  occaType type = occaChar((char) value);

  return PyLong_FromVoidPtr(type);
}

static PyObject* py_occaInt8(PyObject *self, PyObject *args) {
  size_t value;

  if (!PyArg_ParseTuple(args, "n", &value)) {
    return NULL;
  }
  occaType type = occaChar((char) value);

  return PyLong_FromVoidPtr(type);
}

static PyObject* py_occaUInt8(PyObject *self, PyObject *args) {
  size_t value;

  if (!PyArg_ParseTuple(args, "n", &value)) {
    return NULL;
  }
  occaType type = occaUChar((unsigned char) value);

  return PyLong_FromVoidPtr(type);
}

static PyObject* py_occaInt16(PyObject *self, PyObject *args) {
  size_t value;

  if (!PyArg_ParseTuple(args, "n", &value)) {
    return NULL;
  }
  occaType type = occaShort((short) value);

  return PyLong_FromVoidPtr(type);
}

static PyObject* py_occaUInt16(PyObject *self, PyObject *args) {
  size_t value;

  if (!PyArg_ParseTuple(args, "n", &value)) {
    return NULL;
  }
  occaType type = occaUShort((unsigned short) value);

  return PyLong_FromVoidPtr(type);
}

static PyObject* py_occaInt32(PyObject *self, PyObject *args) {
  size_t value;

  if (!PyArg_ParseTuple(args, "n", &value)) {
    return NULL;
  }
  occaType type = occaInt((int) value);

  return PyLong_FromVoidPtr(type);
}

static PyObject* py_occaUInt32(PyObject *self, PyObject *args) {
  size_t value;

  if (!PyArg_ParseTuple(args, "n", &value)) {
    return NULL;
  }
  occaType type = occaUInt((unsigned int) value);

  return PyLong_FromVoidPtr(type);
}

static PyObject* py_occaInt64(PyObject *self, PyObject *args) {
  size_t value;

  if (!PyArg_ParseTuple(args, "n", &value)) {
    return NULL;
  }
  occaType type = occaLong((long) value);

  return PyLong_FromVoidPtr(type);
}

static PyObject* py_occaUInt64(PyObject *self, PyObject *args) {
  size_t value;

  if (!PyArg_ParseTuple(args, "n", &value)) {
    return NULL;
  }
  occaType type = occaULong((unsigned long) value);

  return PyLong_FromVoidPtr(type);
}

static PyObject* py_occaFloat32(PyObject *self, PyObject *args) {
  double value;

  if (!PyArg_ParseTuple(args, "d", &value)) {
    return NULL;
  }
  occaType type = occaFloat((float) value);

  return PyLong_FromVoidPtr(type);
}

static PyObject* py_occaFloat64(PyObject *self, PyObject *args) {
  double value;

  if (!PyArg_ParseTuple(args, "d", &value)) {
    return NULL;
  }
  occaType type = occaDouble((double) value);

  return PyLong_FromVoidPtr(type);
}
//======================================

//----[ Background Device ]-------------
//  |---[ Device ]----------------------
static PyObject* py_occaSetDevice(PyObject *self, PyObject *args) {
  occaDevice device;

  if (!PyArg_ParseTuple(args, "n", &device)) {
    return NULL;
  }
  occaSetDevice(device);

  return Py_None;
}

static PyObject* py_occaSetDeviceFromInfo(PyObject *self, PyObject *args) {
  const char *infos;

  if (!PyArg_ParseTuple(args, "s", &infos)) {
    return NULL;
  }
  occaSetDeviceFromInfo(infos);

  return Py_None;
}

static PyObject* py_occaGetCurrentDevice(PyObject *self, PyObject *args) {
  occaDevice device = occaCurrentDevice();

  return PyLong_FromVoidPtr(device);
}

static PyObject* py_occaSetCompiler(PyObject *self, PyObject *args) {
  const char *compiler;

  if (!PyArg_ParseTuple(args, "s", &compiler)) {
    return NULL;
  }
  occaSetCompiler(compiler);

  return Py_None;
}

static PyObject* py_occaSetCompilerEnvScript(PyObject *self, PyObject *args) {
  const char *compilerEnvScript;

  if (!PyArg_ParseTuple(args, "s", &compilerEnvScript)) {
    return NULL;
  }
  occaSetCompilerEnvScript(compilerEnvScript);

  return Py_None;
}

static PyObject* py_occaSetCompilerFlags(PyObject *self, PyObject *args) {
  const char *compilerFlags;

  if (!PyArg_ParseTuple(args, "s", &compilerFlags)) {
    return NULL;
  }
  occaSetCompilerFlags(compilerFlags);

  return Py_None;
}

static PyObject* py_occaGetCompiler(PyObject *self, PyObject *args) {
  const char *compiler = occaGetCompiler();
  return STR_TO_PYOBJECT(compiler);
}

static PyObject* py_occaGetCompilerEnvScript(PyObject *self, PyObject *args) {
  const char *compilerEnvScript = occaGetCompilerEnvScript();
  return STR_TO_PYOBJECT(compilerEnvScript);
}

static PyObject* py_occaGetCompilerFlags(PyObject *self, PyObject *args) {
  const char *compilerFlags = occaGetCompilerFlags();
  return STR_TO_PYOBJECT(compilerFlags);
}

static PyObject* py_occaFinish(PyObject *self, PyObject *args) {
  occaFinish();

  return Py_None;
}

static PyObject* py_occaCreateStream(PyObject *self, PyObject *args) {
  occaStream stream = occaCreateStream();

  return PyLong_FromVoidPtr(stream);
}

static PyObject* py_occaGetStream(PyObject *self, PyObject *args) {
  occaStream stream = occaGetStream();

  return PyLong_FromVoidPtr(stream);
}

static PyObject* py_occaSetStream(PyObject *self, PyObject *args) {
  occaStream stream;

  if (!PyArg_ParseTuple(args, "n", &stream)) {
    return NULL;
  }
  occaSetStream(stream);

  return Py_None;
}

static PyObject* py_occaWrapStream(PyObject *self, PyObject *args) {
  occaStream stream;
  void *handle;

  if (!PyArg_ParseTuple(args, "n", &handle)) {
    return NULL;
  }
  stream = occaWrapStream(handle);

  return PyLong_FromVoidPtr(stream);
}
//  |===================================

//  |---[ Kernel ]----------------------
static PyObject* py_occaBuildKernel(PyObject *self, PyObject *args) {
  const char *filename, *functionName;
  occaKernelInfo *kInfo;

  if (!PyArg_ParseTuple(args, "ssn", &filename, &functionName, &kInfo)) {
    return NULL;
  }
  occaKernel kernel = occaBuildKernel(filename, functionName, kInfo);

  return PyLong_FromVoidPtr(kernel);
}

static PyObject* py_occaBuildKernelFromString(PyObject *self, PyObject *args) {
  const char *source, *functionName, *language;
  occaKernelInfo *kInfo;
  occaKernel kernel;

  if (!PyArg_ParseTuple(args, "sssn", &source, &functionName, &kInfo, &language)) {
    return NULL;
  }
  if (strcmp(language, "OFL") == 0) {
    kernel = occaBuildKernelFromString(source, functionName, kInfo, occaUsingOFL);
  } else if (strcmp(language, "Native") == 0) {
    kernel = occaBuildKernelFromString(source, functionName, kInfo, occaUsingNative);
  } else {
    kernel = occaBuildKernelFromString(source, functionName, kInfo, occaUsingOKL);
  }

  return PyLong_FromVoidPtr(kernel);
}

static PyObject* py_occaBuildKernelFromBinary(PyObject *self, PyObject *args) {
  const char *binary, *functionName;

  if (!PyArg_ParseTuple(args, "ss", &binary, &functionName)) {
    return NULL;
  }
  occaKernel kernel = occaBuildKernelFromBinary(binary, functionName);

  return PyLong_FromVoidPtr(kernel);
}
//  |===================================

//  |---[ Memory ]----------------------
static PyObject* py_occaWrapMemory(PyObject *self, PyObject *args) {
  void *handle;
  size_t entries;
  int typeSize;

  if (!PyArg_ParseTuple(args, "nni", &handle, &entries, &typeSize)) {
    return NULL;
  }
  const size_t bytes = (entries * typeSize);

  occaMemory memory = occaWrapMemory(handle, bytes);

  return PyLong_FromVoidPtr(memory);
}

static PyObject* py_occaWrapManagedMemory(PyObject *self, PyObject *args) {
  void *handle;
  size_t entries;
  int typeSize, typenum;

  if (!PyArg_ParseTuple(args, "nnii", &handle, &entries, &typeSize, &typenum)) {
    return NULL;
  }
  const size_t bytes = (entries * typeSize);

  occaWrapManagedMemory(handle, bytes);

  return Py_None;
}

static PyObject* py_occaMalloc(PyObject *self, PyObject *args) {
  size_t entries;
  int typeSize;

  if (!PyArg_ParseTuple(args, "ni", &entries, &typeSize)) {
    return NULL;
  }
  const size_t bytes = (entries * typeSize);

  occaMemory memory = occaMalloc(bytes, NULL);

  return PyLong_FromVoidPtr(memory);
}

static PyObject* py_occaManagedAlloc(PyObject *self, PyObject *args) {
  size_t entries;
  int typeSize, typenum;

  if (!PyArg_ParseTuple(args, "nii", &entries, &typeSize, &typenum)) {
    return NULL;
  }
  const size_t bytes = (entries * typeSize);

  int nd         = 1;
  npy_intp *dims = (npy_intp*) malloc(1*sizeof(npy_intp));
  dims[0]        = entries;
  void *data     = occaManagedAlloc(bytes, NULL);

  return PyArray_SimpleNewFromData(nd, dims, typenum, data);
}

static PyObject* py_occaMappedAlloc(PyObject *self, PyObject *args) {
  size_t entries;
  int typeSize;

  if (!PyArg_ParseTuple(args, "ni", &entries, &typeSize)) {
    return NULL;
  }
  const size_t bytes = (entries * typeSize);

  occaMemory memory = occaMappedAlloc(bytes, NULL);

  return PyLong_FromVoidPtr(memory);
}

static PyObject* py_occaManagedMappedAlloc(PyObject *self, PyObject *args) {
  size_t entries;
  int typeSize, typenum;

  if (!PyArg_ParseTuple(args, "nii", &entries, &typeSize, &typenum)) {
    return NULL;
  }
  const size_t bytes = (entries * typeSize);

  int nd         = 1;
  npy_intp *dims = (npy_intp*) malloc(1*sizeof(npy_intp));
  dims[0]        = entries;
  void *data     = occaManagedMappedAlloc(bytes, NULL);

  return PyArray_SimpleNewFromData(nd, dims, typenum, data);
}
//  |===================================
//======================================

//---[ Device ]-------------------------
static PyObject* py_occaPrintModeInfo(PyObject *self, PyObject *args) {
  occaPrintModeInfo();

  return Py_None;
}

static PyObject* py_occaCreateDevice(PyObject *self, PyObject *args) {
  const char *infos;

  if (!PyArg_ParseTuple(args, "s", &infos)) {
    return NULL;
  }
  occaDevice device = occaCreateDevice(infos);

  return PyLong_FromVoidPtr(device);
}

static PyObject* py_occaDeviceMode(PyObject *self, PyObject *args) {
  occaDevice device;
  const char *mode;

  if (!PyArg_ParseTuple(args, "n", &device)) {
    return NULL;
  }
  mode = occaDeviceMode(device);

  return STR_TO_PYOBJECT(mode);
}

static PyObject* py_occaDeviceSetCompiler(PyObject *self, PyObject *args) {
  occaDevice device;
  const char *compiler;

  if (!PyArg_ParseTuple(args, "ns", &device, &compiler)) {
    return NULL;
  }
  occaDeviceSetCompiler(device, compiler);

  return Py_None;
}

static PyObject* py_occaDeviceSetCompilerEnvScript(PyObject *self, PyObject *args) {
  occaDevice device;
  const char *compilerEnvScript;

  if (!PyArg_ParseTuple(args, "ns", &device, &compilerEnvScript)) {
    return NULL;
  }
  occaDeviceSetCompilerEnvScript(device, compilerEnvScript);

  return Py_None;
}

static PyObject* py_occaDeviceSetCompilerFlags(PyObject *self, PyObject *args) {
  occaDevice device;
  const char *compilerFlags;

  if (!PyArg_ParseTuple(args, "ns", &device, &compilerFlags)) {
    return NULL;
  }
  occaDeviceSetCompilerFlags(device, compilerFlags);

  return Py_None;
}

static PyObject* py_occaDeviceGetCompiler(PyObject *self, PyObject *args) {
  occaDevice device;
  const char *compiler;

  if (!PyArg_ParseTuple(args, "n", &device)) {
    return NULL;
  }
  compiler = occaDeviceGetCompiler(device);

  return STR_TO_PYOBJECT(compiler);
}

static PyObject* py_occaDeviceGetCompilerEnvScript(PyObject *self, PyObject *args) {
  occaDevice device;
  const char *compilerEnvScript;

  if (!PyArg_ParseTuple(args, "n", &device)) {
    return NULL;
  }
  compilerEnvScript = occaDeviceGetCompilerEnvScript(device);

  return STR_TO_PYOBJECT(compilerEnvScript);
}

static PyObject* py_occaDeviceGetCompilerFlags(PyObject *self, PyObject *args) {
  occaDevice device;
  const char *compilerFlags;

  if (!PyArg_ParseTuple(args, "n", &device)) {
    return NULL;
  }
  compilerFlags = occaDeviceGetCompilerFlags(device);

  return STR_TO_PYOBJECT(compilerFlags);
}

static PyObject* py_occaDeviceBytesAllocated(PyObject *self, PyObject *args) {
  occaDevice device;

  if (!PyArg_ParseTuple(args, "n", &device)) {
    return NULL;
  }
  udim_t bytes = occaDeviceBytesAllocated(device);

  return PyLong_FromSize_t((size_t) bytes);
}

static PyObject* py_occaDeviceBuildKernel(PyObject *self, PyObject *args) {
  occaDevice device;
  const char *filename, *functionName;
  occaKernelInfo *kInfo;

  if (!PyArg_ParseTuple(args, "nssn", &device, &filename, &functionName, &kInfo)) {
    return NULL;
  }
  occaKernel kernel = occaDeviceBuildKernel(device, filename, functionName, kInfo);

  return PyLong_FromVoidPtr(kernel);
}

static PyObject* py_occaDeviceBuildKernelFromString(PyObject *self, PyObject *args) {
  const char *filename, *functionName, *language;
  occaKernelInfo *kInfo;
  occaKernel kernel;

  if (!PyArg_ParseTuple(args, "sssn", &filename, &functionName, &kInfo, &language)) {
    return NULL;
  }
  if (strcmp(language, "OFL") == 0) {
    kernel = occaBuildKernelFromString(filename, functionName, kInfo, occaUsingOFL);
  } else if (strcmp(language, "Native") == 0) {
    kernel = occaBuildKernelFromString(filename, functionName, kInfo, occaUsingNative);
  } else {
    kernel = occaBuildKernelFromString(filename, functionName, kInfo, occaUsingOKL);
  }

  return PyLong_FromVoidPtr(kernel);
}

static PyObject* py_occaDeviceBuildKernelFromBinary(PyObject *self, PyObject *args) {
  const char *filename, *functionName;

  if (!PyArg_ParseTuple(args, "ss", &filename, &functionName)) {
    return NULL;
  }
  occaKernel kernel = occaBuildKernelFromBinary(filename, functionName);

  return PyLong_FromVoidPtr(kernel);
}

static PyObject* py_occaDeviceMalloc(PyObject *self, PyObject *args) {
  occaDevice device;
  size_t entries;
  int typeSize;

  if (!PyArg_ParseTuple(args, "nni", &device, &entries, &typeSize)) {
    return NULL;
  }
  const size_t bytes = (entries * typeSize);

  occaMemory memory = occaDeviceMalloc(device, bytes, NULL);

  return PyLong_FromVoidPtr(memory);
}

static PyObject* py_occaDeviceManagedAlloc(PyObject *self, PyObject *args) {
  occaDevice device;
  size_t entries;
  int typeSize, typenum;

  if (!PyArg_ParseTuple(args, "nnii", &device, &entries, &typeSize, &typenum)) {
    return NULL;
  }
  const size_t bytes = (entries * typeSize);

  int nd         = 1;
  npy_intp *dims = (npy_intp*) malloc(1*sizeof(npy_intp));
  dims[0]        = entries;
  void *data     = occaManagedAlloc(bytes, NULL);

  return PyArray_SimpleNewFromData(nd, dims, typenum, data);
}

static PyObject* py_occaDeviceMappedAlloc(PyObject *self, PyObject *args) {
  occaDevice device;
  size_t entries;
  int typeSize;

  if (!PyArg_ParseTuple(args, "nni", &device, &entries, &typeSize)) {
    return NULL;
  }
  const size_t bytes = (entries * typeSize);

  occaMemory memory = occaDeviceMappedAlloc(device, bytes, NULL);

  return PyLong_FromVoidPtr(memory);
}

static PyObject* py_occaDeviceManagedMappedAlloc(PyObject *self, PyObject *args) {
  occaDevice device;
  size_t entries;
  int typeSize, typenum;

  if (!PyArg_ParseTuple(args, "nnii", &device, &entries, &typeSize, &typenum)) {
    return NULL;
  }
  const size_t bytes = (entries * typeSize);

  int nd         = 1;
  npy_intp *dims = (npy_intp*) malloc(1*sizeof(npy_intp));
  dims[0]        = entries;
  void *data     = occaDeviceManagedMappedAlloc(device, bytes, NULL);

  return PyArray_SimpleNewFromData(nd, dims, typenum, data);
}

static PyObject* py_occaDeviceFinish(PyObject *self, PyObject *args) {
  occaDevice device;

  if (!PyArg_ParseTuple(args, "n", &device)) {
    return NULL;
  }
  occaDeviceFinish(device);

  return Py_None;
}

static PyObject* py_occaDeviceCreateStream(PyObject *self, PyObject *args) {
  occaDevice device;

  if (!PyArg_ParseTuple(args, "n", &device)) {
    return NULL;
  }
  return PyLong_FromVoidPtr(device);
}

static PyObject* py_occaDeviceGetStream(PyObject *self, PyObject *args) {
  occaDevice device;

  if (!PyArg_ParseTuple(args, "n", &device)) {
    return NULL;
  }
  occaStream stream = occaDeviceGetStream(device);

  return PyLong_FromVoidPtr(stream);
}

static PyObject* py_occaDeviceSetStream(PyObject *self, PyObject *args) {
  occaDevice device;
  occaStream stream;

  if (!PyArg_ParseTuple(args, "nn", &device, &stream)) {
    return NULL;
  }
  occaDeviceSetStream(device, stream);

  return Py_None;
}

static PyObject* py_occaDeviceWrapStream(PyObject *self, PyObject *args) {
  occaDevice device;
  void *handle;

  if (!PyArg_ParseTuple(args, "nn", &device, &handle)) {
    return NULL;
  }
  occaStream stream = occaDeviceWrapStream(device, handle);

  return PyLong_FromVoidPtr(stream);
}

static PyObject* py_occaStreamFree(PyObject *self, PyObject *args) {
  occaStream stream;

  if (!PyArg_ParseTuple(args, "n", &stream)) {
    return NULL;
  }
  occaStreamFree(stream);

  return Py_None;
}

static PyObject* py_occaDeviceFree(PyObject *self, PyObject *args) {
  occaDevice device;

  if (!PyArg_ParseTuple(args, "n", &device)) {
    return NULL;
  }
  occaDeviceFree(device);

  return Py_None;
}
//======================================

//---[ Kernel ]-------------------------
static PyObject* py_occaKernelMode(PyObject *self, PyObject *args) {
  occaKernel kernel;
  const char *mode;

  if (!PyArg_ParseTuple(args, "n", &kernel)) {
    return NULL;
  }
  mode = occaKernelMode(kernel);

  return STR_TO_PYOBJECT(mode);
}

static PyObject* py_occaKernelName(PyObject *self, PyObject *args) {
  occaKernel kernel;
  const char *name;

  if (!PyArg_ParseTuple(args, "n", &kernel)) {
    return NULL;
  }
  name = occaKernelName(kernel);

  return STR_TO_PYOBJECT(name);
}

static PyObject* py_occaKernelGetDevice(PyObject *self, PyObject *args) {
  occaKernel kernel;
  occaDevice device;

  if (!PyArg_ParseTuple(args, "n", &kernel)) {
    return NULL;
  }
  device = occaKernelGetDevice(kernel);

  return PyLong_FromVoidPtr(device);
}

static PyObject* py_occaCreateArgumentList(PyObject *self, PyObject *args) {
  occaArgumentList argList = occaCreateArgumentList();

  return PyLong_FromVoidPtr(argList);
}

static PyObject* py_occaArgumentListClear(PyObject *self, PyObject *args) {
  occaArgumentList argList;

  if (!PyArg_ParseTuple(args, "n", &argList)) {
    return NULL;
  }
  occaArgumentListClear(argList);

  return Py_None;
}

static PyObject* py_occaArgumentListFree(PyObject *self, PyObject *args) {
  occaArgumentList argList;

  if (!PyArg_ParseTuple(args, "n", &argList)) {
    return NULL;
  }
  occaArgumentListFree(argList);

  return Py_None;
}

static PyObject* py_occaArgumentListAddArg(PyObject *self, PyObject *args) {
  occaArgumentList argList;
  int argPos;
  occaType type;

  if (!PyArg_ParseTuple(args, "nin", &argList, &argPos, &type)) {
    return NULL;
  }
  occaArgumentListAddArg(argList, argPos, type);

  return Py_None;
}

static PyObject* py_occaKernelRun(PyObject *self, PyObject *args) {
  occaKernel kernel;
  occaArgumentList argList;

  if (!PyArg_ParseTuple(args, "nn", &kernel, &argList)) {
    return NULL;
  }
  occaKernelRun_(kernel, argList);

  return Py_None;
}

static PyObject* py_occaKernelFree(PyObject *self, PyObject *args) {
  occaKernel kernel;

  if (!PyArg_ParseTuple(args, "n", &kernel)) {
    return NULL;
  }
  occaKernelFree(kernel);

  return Py_None;
}
//======================================

//---[ Memory ]-------------------------
static PyObject* py_occaMemoryMode(PyObject *self, PyObject *args) {
  occaMemory memory;
  const char *mode;

  if (!PyArg_ParseTuple(args, "n", &memory)) {
    return NULL;
  }
  mode = occaMemoryMode(memory);

  return STR_TO_PYOBJECT(mode);
}

static PyObject* py_occaMemoryGetMemoryHandle(PyObject *self, PyObject *args) {
  occaMemory memory;

  if (!PyArg_ParseTuple(args, "n", &memory)) {
    return NULL;
  }
  void *handle = occaMemoryGetMemoryHandle(memory);

  return PyLong_FromVoidPtr(handle);
}

static PyObject* py_occaMemoryGetMappedPointer(PyObject *self, PyObject *args) {
  occaMemory memory;

  if (!PyArg_ParseTuple(args, "n", &memory)) {
    return NULL;
  }
  void *ptr = occaMemoryGetMappedPointer(memory);

  return PyLong_FromVoidPtr(ptr);
}

static PyObject* py_occaMemoryGetTextureHandle(PyObject *self, PyObject *args) {
  occaMemory memory;

  if (!PyArg_ParseTuple(args, "n", &memory)) {
    return NULL;
  }
  void *handle = occaMemoryGetTextureHandle(memory);

  return PyLong_FromVoidPtr(handle);
}

static PyObject* py_occaMemcpy(PyObject *self, PyObject *args) {
  void *dest, *str;
  size_t bytes;

  if (!PyArg_ParseTuple(args, "nnn", &dest, &str, &bytes)) {
    return NULL;
  }
  occaMemcpy(dest, str, bytes);

  return Py_None;
}

static PyObject* py_occaAsyncMemcpy(PyObject *self, PyObject *args) {
  void *dest, *str;
  size_t bytes;

  if (!PyArg_ParseTuple(args, "nnn", &dest, &str, &bytes)) {
    return NULL;
  }
  occaAsyncMemcpy(dest, str, bytes);

  return Py_None;
}

static PyObject* py_occaCopyMemToMem(PyObject *self, PyObject *args) {
  occaMemory dest, src;
  size_t bytes, destOffset, srcOffset;

  if (!PyArg_ParseTuple(args, "nnnnn", &dest, &src, &bytes, &destOffset, &srcOffset)) {
    return NULL;
  }
  occaCopyMemToMem(dest, src, bytes, destOffset, srcOffset);

  return Py_None;
}

static PyObject* py_occaCopyPtrToMem(PyObject *self, PyObject *args) {
  occaMemory dest;
  void *src;
  size_t bytes, destOffset;

  if (!PyArg_ParseTuple(args, "nnnn", &dest, &src, &bytes, &destOffset)) {
    return NULL;
  }
  occaCopyPtrToMem(dest, src, bytes, destOffset);

  return Py_None;
}

static PyObject* py_occaCopyMemToPtr(PyObject *self, PyObject *args) {
  void *dest;
  occaMemory src;
  size_t bytes, srcOffset;

  if (!PyArg_ParseTuple(args, "nnnn", &dest, &src, &bytes, &srcOffset)) {
    return NULL;
  }
  occaCopyMemToPtr(dest, src, bytes, srcOffset);

  return Py_None;
}

static PyObject* py_occaAsyncCopyMemToMem(PyObject *self, PyObject *args) {
  occaMemory dest, src;
  size_t bytes, destOffset, srcOffset;

  if (!PyArg_ParseTuple(args, "nnnnn", &dest, &src, &bytes, &destOffset, &srcOffset)) {
    return NULL;
  }
  occaAsyncCopyMemToMem(dest, src, bytes, destOffset, srcOffset);

  return Py_None;
}

static PyObject* py_occaAsyncCopyPtrToMem(PyObject *self, PyObject *args) {
  occaMemory dest;
  void *src;
  size_t bytes, destOffset;

  if (!PyArg_ParseTuple(args, "nnnn", &dest, &src, &bytes, &destOffset)) {
    return NULL;
  }
  occaAsyncCopyPtrToMem(dest, src, bytes, destOffset);

  return Py_None;
}

static PyObject* py_occaAsyncCopyMemToPtr(PyObject *self, PyObject *args) {
  void *dest;
  occaMemory src;
  size_t bytes, srcOffset;

  if (!PyArg_ParseTuple(args, "nnnn", &dest, &src, &bytes, &srcOffset)) {
    return NULL;
  }
  occaAsyncCopyMemToPtr(dest, src, bytes, srcOffset);

  return Py_None;
}

static PyObject* py_occaMemoryFree(PyObject *self, PyObject *args) {
  occaMemory memory;

  if (!PyArg_ParseTuple(args, "n", &memory)) {
    return NULL;
  }
  occaMemoryFree(memory);

  return Py_None;
}
//======================================

OCCA_END_EXTERN_C
