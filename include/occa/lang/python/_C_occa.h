#ifndef OCCA_C_PYTHON_HEADER
#define OCCA_C_PYTHON_HEADER

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "occa/defines.hpp"

OCCA_START_EXTERN_C

#include "Python.h"
#include "numpy/arrayobject.h"

#if PY_MAJOR_VERSION >= 3
#  define OCCA_PY 3
#else
#  define OCCA_PY 2
#endif

#include "occa_c.h"

//---[ Globals & Flags ]----------------
static PyObject* py_occaSetVerboseCompilation(PyObject *self, PyObject *args);
//======================================

//---[ TypeCasting ]--------------------
static PyObject* py_occaPtr(PyObject *self, PyObject *args);

static PyObject* py_occaBool(PyObject *self, PyObject *args);

static PyObject* py_occaInt8(PyObject *self, PyObject *args);
static PyObject* py_occaUInt8(PyObject *self, PyObject *args);

static PyObject* py_occaInt16(PyObject *self, PyObject *args);
static PyObject* py_occaUInt16(PyObject *self, PyObject *args);

static PyObject* py_occaInt32(PyObject *self, PyObject *args);
static PyObject* py_occaUInt32(PyObject *self, PyObject *args);

static PyObject* py_occaInt64(PyObject *self, PyObject *args);
static PyObject* py_occaUInt64(PyObject *self, PyObject *args);

static PyObject* py_occaFloat32(PyObject *self, PyObject *args);
static PyObject* py_occaFloat64(PyObject *self, PyObject *args);
//======================================

//----[ Background Device ]-------------
//  |---[ Device ]----------------------
static PyObject* py_occaSetDevice(PyObject *self, PyObject *args);
static PyObject* py_occaSetDeviceFromInfo(PyObject *self, PyObject *args);

static PyObject* py_occaGetCurrentDevice(PyObject *self, PyObject *args);

static PyObject* py_occaSetCompiler(PyObject *self, PyObject *args);
static PyObject* py_occaSetCompilerEnvScript(PyObject *self, PyObject *args);
static PyObject* py_occaSetCompilerFlags(PyObject *self, PyObject *args);

static PyObject* py_occaGetCompiler(PyObject *self, PyObject *args);
static PyObject* py_occaGetCompilerEnvScript(PyObject *self, PyObject *args);
static PyObject* py_occaGetCompilerFlags(PyObject *self, PyObject *args);

static PyObject* py_occaFinish(PyObject *self, PyObject *args);

// static PyObject* py_occaWaitFor(PyObject *self, PyObject *args);

static PyObject* py_occaCreateStream(PyObject *self, PyObject *args);
static PyObject* py_occaGetStream(PyObject *self, PyObject *args);
static PyObject* py_occaSetStream(PyObject *self, PyObject *args);
static PyObject* py_occaWrapStream(PyObject *self, PyObject *args);

// static PyObject* py_occaTagStream(PyObject *self, PyObject *args);
//  |===================================

//  |---[ Kernel ]----------------------
static PyObject* py_occaBuildKernel(PyObject *self, PyObject *args);
static PyObject* py_occaBuildKernelFromString(PyObject *self, PyObject *args);
static PyObject* py_occaBuildKernelFromBinary(PyObject *self, PyObject *args);
//  |===================================

//  |---[ Memory ]----------------------
static PyObject* py_occaWrapMemory(PyObject *self, PyObject *args);
static PyObject* py_occaWrapManagedMemory(PyObject *self, PyObject *args);

static PyObject* py_occaMalloc(PyObject *self, PyObject *args);
static PyObject* py_occaManagedAlloc(PyObject *self, PyObject *args);

static PyObject* py_occaMappedAlloc(PyObject *self, PyObject *args);
static PyObject* py_occaManagedMappedAlloc(PyObject *self, PyObject *args);
//  |===================================
//======================================

//---[ Device ]-------------------------
static PyObject* py_occaPrintModeInfo(PyObject *self, PyObject *args);

static PyObject* py_occaCreateDevice(PyObject *self, PyObject *args);

static PyObject* py_occaDeviceMode(PyObject *self, PyObject *args);

static PyObject* py_occaDeviceSetCompiler(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceSetCompilerEnvScript(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceSetCompilerFlags(PyObject *self, PyObject *args);

static PyObject* py_occaDeviceGetCompiler(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceGetCompilerEnvScript(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceGetCompilerFlags(PyObject *self, PyObject *args);

static PyObject* py_occaDeviceBytesAllocated(PyObject *self, PyObject *args);

static PyObject* py_occaDeviceBuildKernel(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceBuildKernelFromString(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceBuildKernelFromBinary(PyObject *self, PyObject *args);

static PyObject* py_occaDeviceMalloc(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceManagedAlloc(PyObject *self, PyObject *args);

static PyObject* py_occaDeviceMappedAlloc(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceManagedMappedAlloc(PyObject *self, PyObject *args);

static PyObject* py_occaDeviceFinish(PyObject *self, PyObject *args);

static PyObject* py_occaDeviceCreateStream(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceGetStream(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceSetStream(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceWrapStream(PyObject *self, PyObject *args);

// static PyObject* py_occaDeviceTagStream(PyObject *self, PyObject *args);
// static PyObject* py_occaDeviceWaitForTag(PyObject *self, PyObject *args);
// static PyObject* py_occaDeviceTimeBetweenTags(PyObject *self, PyObject *args);

static PyObject* py_occaStreamFree(PyObject *self, PyObject *args);
static PyObject* py_occaDeviceFree(PyObject *self, PyObject *args);
//======================================

//---[ Kernel ]-------------------------
static PyObject* py_occaKernelMode(PyObject *self, PyObject *args);
static PyObject* py_occaKernelName(PyObject *self, PyObject *args);

static PyObject* py_occaKernelGetDevice(PyObject *self, PyObject *args);

static PyObject* py_occaCreateArgumentList(PyObject *self, PyObject *args);
static PyObject* py_occaArgumentListClear(PyObject *self, PyObject *args);
static PyObject* py_occaArgumentListFree(PyObject *self, PyObject *args);
static PyObject* py_occaArgumentListAddArg(PyObject *self, PyObject *args);

static PyObject* py_occaKernelRun(PyObject *self, PyObject *args);

static PyObject* py_occaKernelFree(PyObject *self, PyObject *args);
//======================================

//---[ Memory ]-------------------------
static PyObject* py_occaMemoryMode(PyObject *self, PyObject *args);

static PyObject* py_occaMemoryGetMemoryHandle(PyObject *self, PyObject *args);
static PyObject* py_occaMemoryGetMappedPointer(PyObject *self, PyObject *args);
static PyObject* py_occaMemoryGetTextureHandle(PyObject *self, PyObject *args);

static PyObject* py_occaMemcpy(PyObject *self, PyObject *args);
static PyObject* py_occaAsyncMemcpy(PyObject *self, PyObject *args);

static PyObject* py_occaCopyMemToMem(PyObject *self, PyObject *args);
static PyObject* py_occaCopyPtrToMem(PyObject *self, PyObject *args);
static PyObject* py_occaCopyMemToPtr(PyObject *self, PyObject *args);

static PyObject* py_occaAsyncCopyMemToMem(PyObject *self, PyObject *args);
static PyObject* py_occaAsyncCopyPtrToMem(PyObject *self, PyObject *args);
static PyObject* py_occaAsyncCopyMemToPtr(PyObject *self, PyObject *args);

static PyObject* py_occaMemoryFree(PyObject *self, PyObject *args);
//======================================

// Init stuff

static PyMethodDef _C_occaMethods[] = {
  //---[ Globals & Flags ]--------------
  {"setVerboseCompilation", py_occaSetVerboseCompilation, METH_VARARGS},
  //====================================

  //---[ TypeCasting ]------------------
  {"ptr"    , py_occaPtr    , METH_VARARGS},

  {"bool"   , py_occaBool   , METH_VARARGS},

  {"int8"   , py_occaInt8   , METH_VARARGS},
  {"uint8"  , py_occaUInt8  , METH_VARARGS},

  {"int16"  , py_occaInt16  , METH_VARARGS},
  {"uint16" , py_occaUInt16 , METH_VARARGS},

  {"int32"  , py_occaInt32  , METH_VARARGS},
  {"uint32" , py_occaUInt32 , METH_VARARGS},

  {"int64"  , py_occaInt64  , METH_VARARGS},
  {"uint64" , py_occaUInt64 , METH_VARARGS},

  {"float32", py_occaFloat32, METH_VARARGS},
  {"float64", py_occaFloat64, METH_VARARGS},
  //====================================

  //----[ Background Device ]-----------
  //  |---[ Device ]--------------------
  {"setDevice"        , py_occaSetDevice        , METH_VARARGS},
  {"setDeviceFromInfo", py_occaSetDeviceFromInfo, METH_VARARGS},

  {"getCurrentDevice", py_occaGetCurrentDevice, METH_VARARGS},

  {"setCompiler"         , py_occaSetCompiler         , METH_VARARGS},
  {"setCompilerEnvScript", py_occaSetCompilerEnvScript, METH_VARARGS},
  {"setCompilerFlags"    , py_occaSetCompilerFlags    , METH_VARARGS},

  {"getCompiler"         , py_occaGetCompiler         , METH_VARARGS},
  {"getCompilerEnvScript", py_occaGetCompilerEnvScript, METH_VARARGS},
  {"getCompilerFlags"    , py_occaGetCompilerFlags    , METH_VARARGS},

  {"finish", py_occaFinish, METH_VARARGS},

  // {"waitFor", py_occaWaitFor, METH_VARARGS},

  {"createStream", py_occaCreateStream, METH_VARARGS},
  {"getStream"   , py_occaGetStream   , METH_VARARGS},
  {"setStream"   , py_occaSetStream   , METH_VARARGS},
  {"wrapStream"  , py_occaWrapStream  , METH_VARARGS},

  // {"tagStream", py_occaTagStream, METH_VARARGS},
  //  |=================================

  //  |---[ Kernel ]--------------------
  {"buildKernel"          , py_occaBuildKernel          , METH_VARARGS},
  {"buildKernelFromString", py_occaBuildKernelFromString, METH_VARARGS},
  {"buildKernelFromBinary", py_occaBuildKernelFromBinary, METH_VARARGS},
  //  |=================================

  //  |---[ Memory ]--------------------
  {"wrapMemory"       , py_occaWrapMemory       , METH_VARARGS},
  {"wrapManagedMemory", py_occaWrapManagedMemory, METH_VARARGS},

  {"malloc"      , py_occaMalloc      , METH_VARARGS},
  {"managedAlloc", py_occaManagedAlloc, METH_VARARGS},

  {"mappedAlloc"       , py_occaMappedAlloc       , METH_VARARGS},
  {"managedMappedAlloc", py_occaManagedMappedAlloc, METH_VARARGS},
  //  |=================================
  //====================================

  //---[ Device ]-----------------------
  {"printModeInfo", py_occaPrintModeInfo, METH_VARARGS},

  {"createDevice", py_occaCreateDevice, METH_VARARGS},

  {"deviceMode", py_occaDeviceMode, METH_VARARGS},

  {"deviceSetCompiler"         , py_occaDeviceSetCompiler         , METH_VARARGS},
  {"deviceSetCompilerEnvScript", py_occaDeviceSetCompilerEnvScript, METH_VARARGS},
  {"deviceSetCompilerFlags"    , py_occaDeviceSetCompilerFlags    , METH_VARARGS},

  {"deviceGetCompiler"         , py_occaDeviceGetCompiler         , METH_VARARGS},
  {"deviceGetCompilerEnvScript", py_occaDeviceGetCompilerEnvScript, METH_VARARGS},
  {"deviceGetCompilerFlags"    , py_occaDeviceGetCompilerFlags    , METH_VARARGS},

  {"deviceBytesAllocated", py_occaDeviceBytesAllocated, METH_VARARGS},

  {"deviceBuildKernel"          , py_occaDeviceBuildKernel          , METH_VARARGS},
  {"deviceBuildKernelFromString", py_occaDeviceBuildKernelFromString, METH_VARARGS},
  {"deviceBuildKernelFromBinary", py_occaDeviceBuildKernelFromBinary, METH_VARARGS},

  {"deviceMalloc"      , py_occaDeviceMalloc      , METH_VARARGS},
  {"deviceManagedAlloc", py_occaDeviceManagedAlloc, METH_VARARGS},

  {"deviceMappedAlloc"       , py_occaDeviceMappedAlloc       , METH_VARARGS},
  {"deviceManagedMappedAlloc", py_occaDeviceManagedMappedAlloc, METH_VARARGS},

  {"deviceFinish", py_occaDeviceFinish, METH_VARARGS},

  {"deviceCreateStream", py_occaDeviceCreateStream, METH_VARARGS},
  {"deviceGetStream"   , py_occaDeviceGetStream   , METH_VARARGS},
  {"deviceSetStream"   , py_occaDeviceSetStream   , METH_VARARGS},
  {"deviceWrapStream"  , py_occaDeviceWrapStream  , METH_VARARGS},

  // {"deviceTagStream", py_occaDeviceTagStream, METH_VARARGS},
  // {"deviceWaitForTag", py_occaDeviceWaitForTag, METH_VARARGS},
  // {"deviceTimeBetweenTags", py_occaDeviceTimeBetweenTags, METH_VARARGS},

  {"streamFree", py_occaStreamFree, METH_VARARGS},
  {"deviceFree", py_occaDeviceFree, METH_VARARGS},
  //====================================

  //---[ Kernel ]-----------------------
  {"kernelMode", py_occaKernelMode, METH_VARARGS},
  {"kernelName", py_occaKernelName, METH_VARARGS},

  {"kernelGetDevice", py_occaKernelGetDevice, METH_VARARGS},

  {"createArgumentList", py_occaCreateArgumentList, METH_VARARGS},
  {"argumentListClear" , py_occaArgumentListClear , METH_VARARGS},
  {"argumentListFree"  , py_occaArgumentListFree  , METH_VARARGS},
  {"argumentListAddArg", py_occaArgumentListAddArg, METH_VARARGS},

  {"kernelRun", py_occaKernelRun, METH_VARARGS},

  {"kernelFree"    , py_occaKernelFree    , METH_VARARGS},
  //====================================

  //---[ Memory ]-----------------------
  {"memoryMode", py_occaMemoryMode, METH_VARARGS},

  {"memoryGetMemoryHandle" , py_occaMemoryGetMemoryHandle , METH_VARARGS},
  {"memoryGetMappedPointer", py_occaMemoryGetMappedPointer, METH_VARARGS},
  {"memoryGetTextureHandle", py_occaMemoryGetTextureHandle, METH_VARARGS},

  {"memcpy"     , py_occaMemcpy     , METH_VARARGS},
  {"asyncMemcpy", py_occaAsyncMemcpy, METH_VARARGS},

  {"copyMemToMem", py_occaCopyMemToMem, METH_VARARGS},
  {"copyPtrToMem", py_occaCopyPtrToMem, METH_VARARGS},
  {"copyMemToPtr", py_occaCopyMemToPtr, METH_VARARGS},

  {"asyncCopyMemToMem", py_occaAsyncCopyMemToMem, METH_VARARGS},
  {"asyncCopyPtrToMem", py_occaAsyncCopyPtrToMem, METH_VARARGS},
  {"asyncCopyMemToPtr", py_occaAsyncCopyMemToPtr, METH_VARARGS},

  {"memoryFree", py_occaMemoryFree, METH_VARARGS}
  //====================================
};

#if OCCA_PY == 3
static struct PyModuleDef _occaModule = { PyModuleDef_HEAD_INIT, "occa", NULL, -1, _C_occaMethods };

PyMODINIT_FUNC init_C_occa() {
  import_array();
  return PyModule_Create(&_occaModule);
}
#else
PyMODINIT_FUNC init_C_occa() {
  (void) Py_InitModule("_C_occa", _C_occaMethods);
  import_array();
}
#endif

OCCA_END_EXTERN_C

#endif
