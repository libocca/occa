import sys, traceback, gc
from ctypes import *

libocca = CDLL('libocca.so')

"""
---[ C types ]----------------
    c_bool
    c_char
    c_int
    c_long
    c_float
    c_double
//============================
"""

class device:
    # Ok
    def mode(self):
        cMode = self.lib.occaDeviceMode(self.cDevice)
        return c_char_p(cMode).value

    # Ok
    def __init__(self, mode, platformID, deviceID):
        self.lib = libocca

        self.isAllocated = True
        self.cDevice = self.lib.occaGetDevice(mode, platformID, deviceID)

    # Ok
    def setup(self, mode, platformID, deviceID):
        self.isAllocated = True
        self.cDevice = self.lib.occaGetDevice(mode, platformID, deviceID)

    # Ok
    def setCompiler(self, compiler):
        self.lib.occaDeviceSetCompiler(self.cDevice, compiler)

    # Ok
    def setCompilerFlags(self, compilerFlags):
        self.lib.occaDeviceSetCompilerFlags(self.cDevice, compilerFlags)

    def buildKernelFromSource(self, filename, functionName, info = None):
        return kernel(self.lib.occaBuildKernelFromSource(self.cDevice,
                                                         filename,
                                                         functionName,
                                                         info))

    def buildKernelFromBinary(self, filename, functionName):
        return kernel(self.lib.occaBuildKernelFromBinary(self.cDevice,
                                                         filename,
                                                         functionName))

    # Ok
    def malloc(self, entryType, entries):
        if type(entries) is list:
            cByteCount = sizeof(entryType)*len(entries)
            cSource    = (entryType * len(entries))(*entries)
        elif isinstance(entries, (int,long)):
            cByteCount = sizeof(entryType)*entries
            cSource    = None
        else:
            print "Entries should be a list"
            traceback.print_exc(file=sys.stdout)
            sys.exit()

        return memory(self.lib.occaDeviceMalloc(self.cDevice,
                                               cByteCount,
                                               cSource))

    def genStream(self):
        return self.lib.occaGenStream(self.cDevice)

    def getStream(self):
        return self.lib.occaGetStream(self.cDevice)

    def setStream(self):
        return self.lib.occaSetStream(self.cDevice)

    # Ok
    def free(self):
        if self.isAllocated:
            self.lib.occaDeviceFree(self.cDevice)
            self.isAllocated = False

    # Ok
    def __del__(self):
        self.free()

class kernelInfo:
    def __init__(self):
        self.lib = libocca

        self.cKernelInfo = self.lib.occaGenKernelInfo()

    def addDefine(self, macro, value):
        self.lib.occaKernelInfoAddDefine(self.cKernelInfo,
                                         macro,
                                         self.lib.occaString(str(value)));

    def __del__(self):
        self.lib.occaKernelInfoFree(self.cKernelInfo);

class kernel:
    def mode(self):
        cMode = self.lib.occaKernelMode(self.cKernel)
        return c_char_p(cMode).value

    def __init__(self, cKernel):
        self.lib = libocca

        self.cKernel = cKernel

    # Ok
    def preferredDimSize(self):
        return self.lib.occaKernelPreferredDimSize(self.cKernel)

    def setWorkingDims(self, dims, itemsPerGroup, groups):
        itemsPerGroup_ = [(itemsPerGroup[i] if (i < len(itemsPerGroup)) else 1) for i in xrange(3)]
        groups_        = [(groups[i]        if (i < len(groups))        else 1) for i in xrange(3)]

        cItemsPerGroup = (c_size_t * 3)(*itemsPerGroup_)
        cGroups        = (c_size_t * 3)(*groups_)

        self.lib.occaKernelSetWorkingDims(self.cKernel,
                                          dims,
                                          cItemsPerGroup,
                                          cGroups)

    def __call__(self, args):
        argList = self.lib.occaGenArgumentList()

        for arg in args:
            if arg.__class__ is memory:
                self.lib.occaArgumentlistAddArg(argList, arg.cMemory)
            else:
                self.lib.occaArgumentlistAddArg(argList, arg)

        self.lib.occaKernelRun_(self.cKernel, argList)

    # Ok
    def timeTaken(self):
        return self.lib.occaKernelTimeTaken(self.cKernel)

    # Ok
    def free(self):
        self.lib.occaKernelFree(self.cKernel)

    # Ok
    def __del__(self):
        self.free()

class memory:
    # Ok
    def mode(self):
        cMode = self.lib.occaMemoryMode(self.cMemory)
        return c_char_p(cMode).value

    # Ok
    def __init__(self, cMemory):
        self.lib = libocca

        self.isAllocated = True
        self.cMemory = cMemory

    # Ok
    def copyTo(self, entryType, dest, entries = 0, offset = 0):
        copyingToMem = (dest.__class__ is memory)

        if (entries == 0) and not copyingToMem:
            cEntries = len(dest)
        else:
            cEntries = entries

        cByteCount = sizeof(entryType) * cEntries

        if type(dest) is list:
            cDest = (entryType * cEntries)()

            self.lib.occaCopyMemToPtr(cDest,
                                     self.cMemory,
                                     cByteCount,
                                     offset)

            for e in xrange(cEntries):
                dest[e + offset] = cDest[e]
        elif copyingToMem:
            self.lib.occaCopyMemToMem(dest.cMemory,
                                     self.cMemory,
                                     cByteCount,
                                     offset)
        else:
            print "Wrong arguments"
            traceback.print_exc(file=sys.stdout)
            sys.exit()

    # Ok
    def copyFrom(self, entryType, src, entries = 0, offset = 0):
        copyingFromMem = (src.__class__ is memory)

        if (entries == 0) and not copyingFromMem:
            cEntries = len(src)
        else:
            cEntries = entries

        cByteCount = sizeof(entryType) * cEntries

        if type(src) is list:
            cSrc = (entryType * cEntries)(*src)

            self.lib.occaCopyPtrToMem(self.cMemory,
                                     cSrc,
                                     cByteCount,
                                     offset)
        elif copyingFromMem:
            self.lib.occaCopyMemToMem(self.cMemory,
                                     src.cMemory,
                                     cByteCount,
                                     offset)
        else:
            print "Wrong arguments"
            traceback.print_exc(file=sys.stdout)
            sys.exit()

    # [-] Add async later
    def asyncCopyTo(self, entryType, dest, byteCount = 0, offset = 0):
        self.copyTo(entryType, dest, byteCount, offset)

    # [-] Add async later
    def asyncCopyFrom(self, entryType, src, byteCount = 0, offset = 0):
        self.copyFrom(entryType, src, byteCount, offset)

    # Ok
    def swap(self, m):
        self.cMemory, m.cMemory = m.cMemory, self.cMemory

    # Ok
    def free(self):
        if self.isAllocated:
            self.lib.occaMemoryFree(self.cMemory)
            self.isAllocated = False

    # Ok
    def __del__(self):
        self.free()

d = device("OpenCL", 0, 0)

print d.mode()

d.setCompiler("clang++")
d.setCompilerFlags("")

addVectors = d.buildKernelFromSource("addVectors.occa",
                                     "addVectors")

o_a  = d.malloc(c_float, [1,1,1])
o_b  = d.malloc(c_float, [1,2,3])
o_ab = d.malloc(c_float, 3)

ab = [0]*3

o_ab.copyTo(c_float, ab)

print ab
