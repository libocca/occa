#include "_C_occa.h"

static PyMethodDef _C_occaMethods[] = {
  //---[ Setup ]--------------------------
  //======================================

  //----[ Background Device ]-------------
  //  |---[ Device ]----------------------
  //  |===================================

  //  |---[ Kernel ]----------------------
  {"buildKernel" , py_occaBuildKernel , METH_VARARGS},
  //  |===================================

  //  |---[ Memory ]----------------------
  {"managedAlloc", py_occaManagedAlloc, METH_VARARGS},
  //  |===================================
  //======================================

  //---[ Device ]-------------------------
  {"deviceFree", py_occaDeviceFree, METH_VARARGS},
  //======================================

  //---[ Kernel ]-------------------------
  {"kernelFree", py_occaKernelFree, METH_VARARGS},
  //======================================

  //---[ Memory ]-------------------------
  {"memoryFree", py_occaMemoryFree, METH_VARARGS}
  //======================================
};

void init_C_occa(){
  (void) Py_InitModule("_C_occa", _C_occaMethods);
  import_array();
}

//---[ Setup ]--------------------------
//======================================

//----[ Background Device ]-------------
//  |---[ Device ]----------------------
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
//  |===================================

//  |---[ Memory ]----------------------
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
//  |===================================
//======================================

//---[ Device ]-------------------------
static PyObject* py_occaDeviceFree(PyObject *self, PyObject *args){
  occaDevice device;

  if(!PyArg_ParseTuple(args, "n", &device))
    return NULL;

  occaDeviceFree(device);

  return Py_None;
}
//======================================

//---[ Kernel ]-------------------------
static PyObject* py_occaKernelFree(PyObject *self, PyObject *args){
  occaKernel kernel;

  if(!PyArg_ParseTuple(args, "n", &kernel))
    return NULL;

  occaKernelFree(kernel);

  return Py_None;
}
//======================================

//---[ Memory ]-------------------------
static PyObject* py_occaMemoryFree(PyObject *self, PyObject *args){
  occaMemory memory;

  if(!PyArg_ParseTuple(args, "n", &memory))
    return NULL;

  occaMemoryFree(memory);

  return Py_None;
}
//======================================
