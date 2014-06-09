import sys, traceback
from ctypes import *

libocca = CDLL('libocca.so', RTLD_GLOBAL)

occaBool   = c_bool
occaChar   = c_char
occaInt    = c_int
occaLong   = c_long
occaFloat  = c_float
occaDouble = c_double

class device:
    # Ok
    def mode(self):
        cMode = libocca.occaDeviceMode(self.cDevice)
        return c_char_p(cMode).value

    # Ok
    def __init__(self, mode, platformID, deviceID):
        self.cDevice = libocca.occaGetDevice(mode, platformID, deviceID)

    # Ok
    def setup(self, mode, platformID, deviceID):
        self.cDevice = libocca.occaGetDevice(mode, platformID, deviceID)

    # Ok
    def setCompiler(self, compiler):
        libocca.occaDeviceSetCompiler(self.cDevice, compiler)

    # Ok
    def setCompilerFlags(self, compilerFlags):
        libocca.occaDeviceSetCompilerFlags(self.cDevice, compilerFlags)

    # Ok
    def malloc(self, entryType, entries):
        if type(entries) is list:
            cByteCount = sizeof(entryType)*len(entries)
            cSource    = (entryType * 3)(*entries)
        elif isinstance(entries, (int,long)):
            cByteCount = sizeof(entryType)*entries
            cSource    = None
        else:
            print "Wrong arguments"
            traceback.print_exc(file=sys.stdout)
            sys.exit()

        return memory(libocca.occaDeviceMalloc(self.cDevice,
                                               cByteCount,
                                               cSource))

    def genStream(self):
        return libocca.occaGenStream(self.cDevice)

    def getStream(self):
        return libocca.occaGetStream(self.cDevice)

    def setStream(self):
        return libocca.occaSetStream(self.cDevice)

    # Ok
    def free(self):
        return libocca.occaDeviceFree(self.cDevice)

class kernelInfo:
    def addDefine(self):
        pass

class kernel:
    def mode(self):
        cMode = libocca.occaKernelMode(self.cKernel)
        return c_char_p(cMode).value

    def __init__(self, cKernel):
        self.cKernel = cKernel

    def preferredDimSize(self):
        return libocca.occaKernelPreferredDimSize(self.cKernel)

    def setWorkingDims(self, dims, itemsPerGroup, groups):


        # occaDims = ?
        return libocca.occaKernelSetWorkingDims(self.cKernel)

    def __call__(self, args):
        argList = libocca.occaGenArgumentList()

        for arg in args:
            if arg.__class__ is memory:
                libocca.occaArgumentlistAddArg(argList, arg)
            else:
                print "Not implemented yet"

        libocca.occaKernelRun_(self.cKernel, argList)

    def timeTaken(self):
        return libocca.occaKernelTimeTaken(self.cKernel)

    def free(self):
        libocca.occaKernelFree(self.cKernel)

class memory:
    def mode(self):
        cMode = libocca.occaMemoryMode(self.cMemory)
        return c_char_p(cMode).value

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

d = device("OpenMP", 0, 0)

print d.mode()

d.setCompiler("clang++")
d.setCompilerFlags("blah blah")

d.malloc(c_float, [1,2,3])

d.free()
