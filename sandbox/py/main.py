import _C_occa
import numpy as np
import weakref

#---[ Setup ]---------------------------
def sizeof(npType):
    return np.dtype(npType).itemsize

def typeof(npType):
    return np.dtype(npType).num
#=======================================

#----[ Background Device ]--------------
#  |---[ Device ]-----------------------
#  |====================================

#  |---[ Kernel ]-----------------------
def buildKernel(filename, functionName, info = 0):
    kHandle = _C_occa.buildKernel(filename, functionName, info)
    return kernel(kHandle)
#  |====================================

#  |---[ Memory ]-----------------------
def managedAlloc(entries, type_):
    return _C_occa.managedAlloc(entries, sizeof(type_), typeof(type_))
#  |====================================
#=======================================

#---[ Device ]--------------------------
class device:
    def __init__(self):
        self.handle      = 0
        self.isAllocated = False

    def free(self):
        import _C_occa
        if self.isAllocated:
            _C_occa.deviceFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()
#=======================================

#---[ Kernel ]--------------------------
class kernel:
    def __init__(self):
        self.handle      = 0
        self.isAllocated = False

    def __init__(self, handle_):
        self.handle      = handle_
        self.isAllocated = True

    def free(self):
        import _C_occa
        if self.isAllocated:
            _C_occa.kernelFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()

    def __call__(self, args):
        argList = _C_occa.createArgumentList()

        for i in xrange(len(args)):
            _C_occa.argumentListAddArg(argList, i, 0) # <>

        _C_occa.kernelRun(self.handle, argList)

        _C_occa.argumenetListFree(argList)
#=======================================

#---[ Memory ]--------------------------
class memory:
    def __init__(self):
        self.handle      = 0
        self.isAllocated = False

    def free(self):
        import _C_occa
        if self.isAllocated:
            _C_occa.memoryFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()
#=======================================

k = buildKernel("addVectors.okl", "addVectors")
