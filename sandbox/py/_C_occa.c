#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject* test(PyObject *self, PyObject *args);

static PyMethodDef _C_occaMethods[] = {
  {"test", test, METH_VARARGS}
};

void init_C_occa(){
  (void) Py_InitModule("_C_occa", _C_occaMethods);
  import_array();
}

static PyObject* test(PyObject *self, PyObject *args){
  const char *str;
  if(!PyArg_ParseTuple(args, "s",
                       &str)){

    return NULL;
  }

  printf("test(%s)\n", str);

  return Py_None;
}