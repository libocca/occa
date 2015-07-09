#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"

#include "occa_c.h"

//---[ Setup ]--------------------------
//======================================

//----[ Background Device ]-------------
//  |---[ Device ]----------------------
//  |===================================

//  |---[ Kernel ]----------------------
static PyObject* py_occaBuildKernel(PyObject *self, PyObject *args);
//  |===================================

//  |---[ Memory ]----------------------
static PyObject* py_occaManagedAlloc(PyObject *self, PyObject *args);
//  |===================================
//======================================

//---[ Device ]-------------------------
static PyObject* py_occaDeviceFree(PyObject *self, PyObject *args);
//======================================

//---[ Kernel ]-------------------------
static PyObject* py_occaKernelFree(PyObject *self, PyObject *args);
//======================================

//---[ Memory ]-------------------------
static PyObject* py_occaMemoryFree(PyObject *self, PyObject *args);
//======================================