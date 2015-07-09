import  numpy as np

import ctypes
from ctypes import c_bool   as cBool
from ctypes import c_char   as cChar
from ctypes import c_int    as cInt
from ctypes import c_long   as cLong
from ctypes import c_float  as cFloat
from ctypes import c_double as cDouble
from ctypes import c_size_t as cSizeT
from ctypes import c_void_p as cVoidP

libocca       = None
typeToNpyType = {}

#---[ Setup ]---------------------------
def loadLibrary():
    global libocca
    global typeToNpyType

    if libocca:
        return

    libocca = ctypes.CDLL('libocca.so')

    typeToNpyType = {'bool'   : np.NPY_BOOL   ,\
                     'int8'   : np.NPY_INT8   ,\
                     'int16'  : np.NPY_INT16  ,\
                     'int32'  : np.NPY_INT32  ,\
                     'int64'  : np.NPY_INT64  ,\
                     'uInt8'  : np.NPY_UINT8  ,\
                     'uInt16' : np.NPY_UINT16 ,\
                     'uInt32' : np.NPY_UINT32 ,\
                     'uInt64' : np.NPY_UINT64 ,\
                     'float32': np.NPY_FLOAT32,\
                     'float64': np.NPY_FLOAT64}

    setupFunction(cVoidP, "occaBuildKernel")
    setupFunction(cVoidP, "occaManagedAlloc")
    setupFunction(cVoidP, "occaCreateArgumentList")
    setupFunction(cVoidP, "occaKernelRun_")
    setupFunction(cVoidP, "occaArgumentListFree")

    occaTypes = ['occaInt'  , 'occaUInt'  ,\
                 'occaChar' , 'occaUChar' ,\
                 'occaShort', 'occaUShort',\
                 'occaLong' , 'occaULong' ,\
                 'occaFloat', 'occaDouble']

    for occaType in occaTypes:
        setupFunction(cVoidP, occaType)

def setupFunction(returnType, functionName):
    global libocca

    f         = getattr(libocca, functionName)
    f.restype = returnType

def occaFunction(functionName):
    global libocca

    return getattr(libocca, functionName)

def occaCast(arg):
    global libocca

    if arg.__class__ is memory:
        return arg.handle

    cType = str(arg.__class__.__name__)[2:]

    if cType[0] == 'u':
        cType = "occa" + cType[:2].swapcase() + cType[2:]
    else:
        cType = "occa" + cType[:1].swapcase() + cType[1:]

    return cVoidP(occaFunction(cType)(arg))

def sizeof(t):
    return np.dtype(t).itemsize

def nameof(t):
    return np.dtype(t).name
#=======================================

#----[ Background Device ]--------------
#  |---[ Device ]-----------------------
#  |====================================

#  |---[ Kernel ]-----------------------
def buildKernel(filename, functionName, info = None):
    global libocca
    loadLibrary()

    return kernel(cVoidP(libocca.occaBuildKernel(filename,
                                                 functionName,
                                                 info)))
#  |====================================

#---[ Memory ]--------------------------
def managedAlloc(type_, bytes_, source = None):
    global libocca
    global typeToNpyType
    loadLibrary()

    ptr = cVoidP(libocca.occaManagedAlloc(cSizeT(bytes_),
                                          source))

    cdef int cBytes = bytes_
    cdef np.npy_intp shape[1]

    shape[0] = <np.npy_intp> cBytes

    return np.PyArray_SimpleNewFromData(1,                           \
                                        shape,                       \
                                        typeToNpyType[nameof(type_)],\
                                        <void*> ptr)

#  |====================================
#=======================================

#---[ Device ]--------------------------
class device:
    def __init__(self):
        loadLibrary()

        self.handle      = None
        self.isAllocated = False

    def free(self):
        global libocca

        if self.isAllocated:
            libocca.occaDeviceFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()
#=======================================

#---[ Kernel ]--------------------------
class kernel:
    def __init__(self):
        loadLibrary()

        self.handle      = None
        self.isAllocated = False

    def __init__(self, handle_):
        self.handle      = handle_
        self.isAllocated = True

    def free(self):
        global libocca

        if self.isAllocated:
            libocca.occaKernelFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()

    def __call__(self, args):
        global libocca

        argList = cVoidP(libocca.occaCreateArgumentList())

        for i in xrange(len(args)):
            libocca.occaArgumentListAddArg(argList,
                                           cInt(i),
                                           occaCast(args[i]))

        libocca.occaKernelRun_(self.handle, argList)

        libocca.occaArgumentListFree(argList)
#=======================================

#---[ Memory ]--------------------------
class memory:
    def __init__(self):
        loadLibrary()

        self.handle      = None
        self.isAllocated = False

    def free(self):
        global libocca

        if self.isAllocated:
            libocca.occaMemoryFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()
#=======================================

# entries = 5

# addVectors = buildKernel('addVectors.okl', 'addVectors')
# addVectors([cInt(entries)])

# managedAlloc(5*sizeof(float))

print np.dtype(np.float32).name