from ctypes import *
from array import *

libocca = CDLL('libocca.so', RTLD_GLOBAL)

class device:
    def mode(self):
        return libocca.occaDeviceMode(self.cDevice)

    def __init__(self, mode, platformID, deviceID):
        self.cDevice = libocca.occaGetDevice(mode, platformID, deviceID)

    def setup(self, mode, platformID, deviceID):
        self.cDevice = libocca.occaGetDevice(mode, platformID, deviceID)

    def setOmpCompiler(self, compiler):
        libocca.occaDeviceSetOmpCompiler(self.cDevice, compiler)

    def setOmpCompilerFlags(self, compilerFlags):
        libocca.occaDeviceSetOmpCompilerFlags(self.cDevice, compilerFlags)

    def setCudaCompiler(self, compiler):
        libocca.occaDeviceSetCudaCompiler(self.cDevice, compiler)

    def setCudaCompilerFlags(self, compilerFlags):
        libocca.occaDeviceSetCudaCompilerFlags(self.cDevice, compilerFlags)

    def malloc(self, byteCount, source):
        return memory(libocca.occaDevicMalloc(self.cDevice,
                                              byteCount,
                                              source))

    def genStream(self):
        return libocca.occaGenStream(self.cDevice)

    def getStream(self):
        return libocca.occaGetStream(self.cDevice)

    def setStream(self):
        return libocca.occaSetStream(self.cDevice)

    def free(self):
        return libocca.occaDeviceFree(self.cDevice)

class kernelInfo:
    def addDefine(self):
        pass

class kernel:
    def mode(self):
        return libocca.occaKernelMode(self.cKernel)

    def __init__(self, cKernel):
        self.cKernel = cKernel

    def preferredDimSize(self):
        return libocca.occaKernelPreferredDimSize(self.cKernel)

    def setWorkingDims(self, dims, itemsPerGroup, groups):


        # occaDims = ?
        return libocca.occaKernelSetWorkingDims(self.cKernel)


    # (self) Operator

    def timeTaken(self):
        return libocca.occaKernelTimeTaken(self.cKernel)

    def free(self):
        libocca.occaKernelFree(self.cKernel)

class memory:
    def mode(self):
        return libocca.occaMemoryMode(self.cMemory)

    def __init__(self, cMemory):
        self.cMemory = cMemory

    def copyTo(self, dest, byteCount = 0, offset = 0):
        if dest.__class__ is array:
            pass
        elif dest.__class__ is memory:
            pass
        else:
            print "Error"

    def copyFrom(self, src, byteCount = 0, offset = 0):
        pass

    def asyncCopyTo(self, dest, byteCount = 0, offset = 0):
        pass

    def asyncCopyFrom(self, src, byteCount = 0, offset = 0):
        pass

    def swap(self, m):
        self.cMemory, m.cMemory = m.cMemory, self.cMemory

    def free(self):
        libocca.occaMemoryFree(self.cMemory)

a = array('f', [0,1])
d = device("OpenMP", 0, 0)

if a.__class__ is array:
    print 1

if d.__class__ is device:
    print 1
