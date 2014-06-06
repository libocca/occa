from ctypes import *

libocca = CDLL('libocca.so', RTLD_GLOBAL)

class device:
    def mode(self):
        libocca.occaDeviceMode(self.device)

    def setup(self, mode, platformID, deviceID):
        self.device = libocca.occaGetDevice(mode, platformID, deviceID)

    def setOmpCompiler(self, compiler):
        libocca.occaDeviceSetOmpCompiler(self.device, compiler)

    def setOmpCompilerFlags(self, compilerFlags):
        libocca.occaDeviceSetOmpCompilerFlags(self.device, compilerFlags)

    def setCudaCompiler(self, compiler):
        libocca.occaDeviceSetCudaCompiler(self.device, compiler)

    def setCudaCompilerFlags(self, compilerFlags):
        libocca.occaDeviceSetCudaCompilerFlags(self.device, compilerFlags)

    def malloc(self, byteCount, source):
        return memory(libocca.occaDevicMalloc(self.device,
                                              byteCount,
                                              source))

    def genStream(self):
        return libocca.occaGenStream(self.device)

    def getStream(self):
        return libocca.occaGetStream(self.device)

    def setStream(self):
        return libocca.occaSetStream(self.device)

    def free(self):
        return libocca.occaDeviceFree(self.device)

class kernelInfo:
    def addDefine(self):
        pass

class kernel:

    def mode(self):
        pass

    def preferredDimSize(self):
        pass

    def setWorkingDims(self):
        pass

    # (self) Operator

    def timeTaken(self):
        pass

    def free(self):
        pass

class memory:

    def mode(self):
        pass

    def copyTo(self):
        pass

    def copyFrom(self):
        pass

    def asyncCopyTo(self):
        pass

    def asyncCopyFrom(self):
        pass

    def swap(self):
        pass

    def free(self):
        pass

dev = device()

dev.setup("OpenMP", 0, 0)
