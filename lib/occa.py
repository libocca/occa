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
    def __init__(self, infos):
        self.lib = libocca

        self.isAllocated = True

        getDevice = getattr(self.lib, "occaGetDevice")
        getDevice.restype = c_void_p

        self.cDevice = c_void_p(self.lib.occaGetDevice(infos))

    # Ok
    def setup(self, infos):
        self.isAllocated = True
        self.cDevice = self.lib.occaGetDevice(infos)

    # Ok
    def setCompiler(self, compiler):
        self.lib.occaDeviceSetCompiler(self.cDevice, compiler)

    # Ok
    def setCompilerFlags(self, compilerFlags):
        self.lib.occaDeviceSetCompilerFlags(self.cDevice, compilerFlags)

    # Ok
    def buildKernelFromSource(self, filename, functionName, info = None):
        occaBuildKernelFromSource = getattr(self.lib, "occaBuildKernelFromSource")
        occaBuildKernelFromSource.restype = c_void_p

        return kernel(c_void_p(self.lib.occaBuildKernelFromSource(self.cDevice,
                                                                  filename,
                                                                  functionName,
                                                                  info)))

    # Ok
    def buildKernelFromBinary(self, filename, functionName):
        occaBuildKernelFromBinary = getattr(self.lib, "occaBuildKernelFromBinary")
        occaBuildKernelFromBinary.restype = c_void_p

        return c_void_p(self.lib.occaBuildKernelFromBinary(self.cDevice,
                                                           filename,
                                                           functionName))

    # Ok
    def malloc(self, entries, entryType):
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

        occaDeviceMalloc = getattr(self.lib, "occaDeviceMalloc")
        occaDeviceMalloc.restype = c_void_p

        return memory(c_void_p(self.lib.occaDeviceMalloc(self.cDevice,
                                                         c_size_t(cByteCount),
                                                         cSource)))

    def createStream(self):
        return self.lib.occaCreateStream(self.cDevice)

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

# Ok
class occaDim(Structure):
    _fields_ = [('x', c_size_t),
                ('y', c_size_t),
                ('z', c_size_t)]

class kernelInfo:
    def __init__(self):
        self.lib = libocca

        self.cKernelInfo = self.lib.occaCreateKernelInfo()

    def addDefine(self, macro, value):
        self.lib.occaKernelInfoAddDefine(self.cKernelInfo,
                                         macro,
                                         self.lib.occaString(str(value)))

    def __del__(self):
        self.lib.occaKernelInfoFree(self.cKernelInfo)

class kernel:
    # Ok
    def mode(self):
        cMode = self.lib.occaKernelMode(self.cKernel)
        return c_char_p(cMode).value

    # Ok
    def __init__(self, cKernel):
        self.lib = libocca

        self.cKernel = cKernel

    # Ok
    def preferredDimSize(self):
        return self.lib.occaKernelPreferredDimSize(self.cKernel)

    # Ok
    def setWorkingDims(self, dims, itemsPerGroup, groups):
        if type(itemsPerGroup) is list:
            ipg = [(itemsPerGroup[i] if (i < len(itemsPerGroup)) else 1) for i in xrange(3)]
        else:
            ipg = [itemsPerGroup, 1, 1]

        if type(groups) is list:
            g = [(groups[i] if (i < len(groups)) else 1) for i in xrange(3)]
        else:
            g = [groups, 1, 1]

        cItemsPerGroup = occaDim(ipg[0], ipg[1], ipg[2])
        cGroups        = occaDim(g[0]  , g[1]  , g[2])

        self.lib.occaKernelSetWorkingDims(self.cKernel,
                                          c_size_t(dims),
                                          cItemsPerGroup,
                                          cGroups)

    # Ok
    def __call__(self, args):
        # X = getattr(self.lib, "X")
        # X.restype = c_void_p

        occaCreateArgumentList = getattr(self.lib, "occaCreateArgumentList")
        occaCreateArgumentList.restype = c_void_p

        argList = c_void_p(self.lib.occaCreateArgumentList())

        for i in xrange(len(args)):
            arg = args[i]

            if arg.__class__ is memory:
                self.lib.occaArgumentListAddArg(argList,
                                                c_int(i),
                                                arg.cMemory)
            else:
                cType = str(arg.__class__.__name__)[2:]

                if cType[0] == 'u':
                    cType = "occa" + cType[:2].swapcase() + cType[2:]
                else:
                    cType = "occa" + cType[:1].swapcase() + cType[1:]

                occaCast = getattr(self.lib, cType)
                occaCast.restype = c_void_p

                self.lib.occaArgumentListAddArg(argList,
                                                c_int(i),
                                                c_void_p(getattr(self.lib, cType)(arg)))

        self.lib.occaKernelRun_(self.cKernel, argList)

        self.lib.occaArgumentListFree(argList)

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
    def copyTo(self, dest, entryType = c_byte, entries = 0, offset = 0):
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
    def copyFrom(self, src, entryType = c_byte, entries = 0, offset = 0):
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
    def asyncCopyTo(self, dest, entryType = c_byte, byteCount = 0, offset = 0):
        self.copyTo(dest, entryType, byteCount, offset)

    # [-] Add async later
    def asyncCopyFrom(self, src, entryType = c_byte, byteCount = 0, offset = 0):
        self.copyFrom(src, entryType, byteCount, offset)

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
