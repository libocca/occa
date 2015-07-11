import _C_occa
import numpy as np
import sys

#---[ Setup ]---------------------------
def varIsOfClass(v, class_):
    return v.__class__ is class_

def varIsNotOfClass(v, class_):
    return (not varIsOfClass(v, class_))

def isAString(v):
    return isinstance(v, basestring)

def isNotAString(v):
    return (not isAString(v))

def isAnInteger(v):
    return (isinstance(v, int) or \
            issubclass(np.dtype(v).type, np.integer))

def isNotAnInteger(v):
    return (not isAnInteger(v))

def isAMemoryType(v):
    return (varIsOfClass(v, memory) or varIsOfClass(v, np.ndarray))

def isNotAMemoryType(v):
    return (not isAMemoryType(v))

def isANumpyType(v):
    try:
        np.dtype(v)
    except NameError:
        return False

    return True

def isNotANumpyType(v):
    return (not isANumpyType(v))

def sizeof(npType):
    return np.dtype(npType).itemsize

def typeof(npType):
    return np.dtype(npType).num

def nameof(npType):
    return np.dtype(npType).name
#=======================================

#---[ Globals & Flags ]-----------------
def setVerboseCompilation(value):
    _C_occa.setVerboseCompilation(value)
#=======================================

#----[ Background Device ]--------------
#  |---[ Device ]-----------------------
def setDevice(arg):
    #---[ Arg Testing ]-------
    try:
        if isNotAString(arg) and \
           varIsNotOfClass(arg, device):

            raise ValueError('Argument to [occa.setDevice] must be a occa.device or string')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    if isAString(arg):
        _C_occa.setDeviceFromInfo(arg)
    elif varIsOfClass(arg, device):
        _C_occa.setDevice(arg.handle)

def getCurrentDevice():
    return device(_C_occa.getCurrentDevice())

def setCompiler(compiler):
    #---[ Arg Testing ]-------
    try:
        if isNotAString(arg):
            raise ValueError('Argument to [occa.setCompiler] must be a string')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    _C_occa.setCompiler(compiler)

def setCompilerEnvScript(compilerEnvScript):
    #---[ Arg Testing ]-------
    try:
        if isNotAString(arg):
            raise ValueError('Argument to [occa.setCompilerEnvScript] must be a string')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    _C_occa.setCompilerEnvScript(compilerEnvScript)

def setCompilerFlags(compilerFlags):
    #---[ Arg Testing ]-------
    try:
        if isNotAString(arg):
            raise ValueError('Argument to [occa.setCompilerFlags] must be a string')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    _C_occa.setCompilerFlags(compilerFlags)

def getCompiler():
    return _C_occa.getCompiler()

def getCompilerEnvScript():
    return _C_occa.getCompilerEnvScript()

def getCompilerFlags():
    return _C_occa.getCompilerFlags()

def flush():
    _C_occa.flush()

def finish():
    _C_occa.finish()

def createStream():
    return stream(_C_occa.createStream())

def getStream():
    return stream(_C_occa.getStream())

def setStream(stream_):
    #---[ Arg Testing ]-------
    try:
        if varIsNotOfClass(stream_, stream):
            raise ValueError('Argument to [occa.setStream] must be a occa.stream')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    _C_occa.setStream(stream_)

def wrapStream(handle):
    #---[ Arg Testing ]-------
    try:
        if varIsNotAnInteger(handle):
            raise ValueError('Argument to [occa.wrapStream] must be a [void*] handle')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    return stream(_C_occa.wrapStream(handle))
#  |====================================

#  |---[ Kernel ]-----------------------
def buildKernel(str_, functionName, kInfo = 0):
    #---[ Arg Testing ]-------
    try:
        if isNotAString(str_):
            raise ValueError('1st argument to [occa.buildKernel] must be a string')
        elif isNotAString(functionName):
            raise ValueError('2nd argument to [occa.buildKernel] must be a string')
        elif (kInfo != 0) and (varNotIsOfClass(kInfo, kernelInfo)):
            raise ValueError('3rd argument to [occa.buildKernel] (if given) must be an [occa.kernelInfo]')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    kInfo_ = (0 if (kInfo == 0) else kInfo.handle)
    return kernel(_C_occa.buildKernel(str_, functionName, kInfo_))

def buildKernelFromSource(filename, functionName, kInfo = 0):
    #---[ Arg Testing ]-------
    try:
        if isNotAString(filename):
            raise ValueError('1st argument to [occa.buildKernelFromSource] must be a string')
        elif isNotAString(functionName):
            raise ValueError('2nd argument to [occa.buildKernelFromSource] must be a string')
        elif (kInfo != 0) and (varNotIsOfClass(kInfo, kernelInfo)):
            raise ValueError('3rd argument to [occa.buildKernelFromSource] (if given) must be an [occa.kernelInfo]')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    kInfo_ = (0 if (kInfo == 0) else kInfo.handle)
    return kernel(_C_occa.buildKernelFromSource(filename, functionName, kInfo_))

def buildKernelFromString(source, functionName, kInfo = 0, language = "OKL"):
    #---[ Arg Testing ]-------
    try:
        if isNotAString(source):
            raise ValueError('1st argument to [occa.buildKernelFromString] must be a string')
        elif isNotAString(functionName):
            raise ValueError('2nd argument to [occa.buildKernelFromString] must be a string')
        elif (kInfo != 0) and (varNotIsOfClass(kInfo, kernelInfo)):
            raise ValueError('3rd argument to [occa.buildKernelFromString] (if given) must be an [occa.kernelInfo]')
        elif isNotAString(language):
            raise ValueError('4th argument to [occa.buildKernelFromString] (if given) must be a string')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    kInfo_ = (0 if (kInfo == 0) else kInfo.handle)
    return kernel(_C_occa.buildKernelFromString(source, functionName, kInfo, language))

def buildKernelFromBinary(binary, functionName):
    #---[ Arg Testing ]-------
    try:
        if isNotAString(binary):
            raise ValueError('1st argument to [occa.buildKernelFromBinary] must be a string')
        elif isNotAString(functionName):
            raise ValueError('2nd argument to [occa.buildKernelFromBinary] must be a string')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    return kernel(_C_occa.buildKernelFromBinary(binary, functionName))

def buildKernelFromLoopy(filename, functionName, kInfo = 0):
    #---[ Arg Testing ]-------
    try:
        if isNotAString(filename):
            raise ValueError('1st argument to [occa.buildKernelFromSource] must be a string')
        elif isNotAString(functionName):
            raise ValueError('2nd argument to [occa.buildKernelFromSource] must be a string')
        elif (kInfo != 0) and (varNotIsOfClass(kInfo, kernelInfo)):
            raise ValueError('3rd argument to [occa.buildKernelFromSource] (if given) must be an [occa.kernelInfo]')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    kInfo_ = (0 if (kInfo == 0) else kInfo.handle)
    return kernel(_C_occa.buildKernelFromLoopy(filename, functionName, kInfo_))

def buildKernelFromFloopy(filename, functionName, kInfo = 0):
    #---[ Arg Testing ]-------
    try:
        if isNotAString(filename):
            raise ValueError('1st argument to [occa.buildKernelFromSource] must be a string')
        elif isNotAString(functionName):
            raise ValueError('2nd argument to [occa.buildKernelFromSource] must be a string')
        elif (kInfo != 0) and (varNotIsOfClass(kInfo, kernelInfo)):
            raise ValueError('3rd argument to [occa.buildKernelFromSource] (if given) must be an [occa.kernelInfo]')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    kInfo_ = (0 if (kInfo == 0) else kInfo.handle)
    return kernel(_C_occa.buildKernelFromFloopy(filename, functionName, kInfo_))
#  |====================================

#  |---[ Memory ]-----------------------
def memcpy(dest, src, bytes_, offset1 = 0, offset2 = 0):
    #---[ Arg Testing ]-------
    try:
        if isNotAMemoryType(dest):
            raise ValueError('1st argument to [occa.memcpy] must be a [numpy.ndarray] or [occa.memory]')
        elif isNotAMemoryType(src):
            raise ValueError('2nd argument to [occa.memcpy] must be a [numpy.ndarray] or [occa.memory]')
        elif isNotAnInteger(bytes_):
            raise ValueError('3rd argument to [occa.memcpy] (if given) must be an integer')
        elif isNotAnInteger(offset1):
            raise ValueError('4th argument to [occa.memcpy] (if given) must be an integer')
        elif isNotAnInteger(offset2):
            raise ValueError('5th argument to [occa.memcpy] (if given) must be an integer')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    if varIsOfClass(dest, memory):
        if varIsOfClass(src, memory):
            _C_occa.copyMemToMem(dest.handle, src.handle, bytes_, offset1, offset2)
        else:
            _C_occa.copyPtrToMem(dest.handle, src, bytes_, offset1)
    else:
        if varIsOfClass(src, memory):
            _C_occa.copyMemToPtr(dest, src.handle, bytes_, offset1)
        else:
            _C_occa.memcpy(dest, src, bytes_)

def asyncMemcpy(dest, src, bytes_, offset1 = 0, offset2 = 0):
    #---[ Arg Testing ]-------
    try:
        if isNotAMemoryType(dest):
            raise ValueError('1st argument to [occa.memcpy] must be a [numpy.ndarray] or [occa.memory]')
        elif isNotAMemoryType(src):
            raise ValueError('2nd argument to [occa.memcpy] must be a [numpy.ndarray] or [occa.memory]')
        elif isNotAnInteger(bytes_):
            raise ValueError('3rd argument to [occa.memcpy] (if given) must be an integer')
        elif isNotAnInteger(offset1):
            raise ValueError('4th argument to [occa.memcpy] (if given) must be an integer')
        elif isNotAnInteger(offset2):
            raise ValueError('5th argument to [occa.memcpy] (if given) must be an integer')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    if varIsOfClass(dest, memory):
        if varIsOfClass(src, memory):
            _C_occa.asyncCopyMemToMem(dest.handle, src.handle, bytes_, offset1, offset2)
        else:
            _C_occa.asyncCopyPtrToMem(dest.handle, src, bytes_, offset1)
    else:
        if varIsOfClass(src, memory):
            _C_occa.asyncCopyMemToPtr(dest, src.handle, bytes_, offset1)
        else:
            _C_occa.asyncMemcpy(dest, src, bytes_)

def wrapMemory(handle, type_, entries):
    #---[ Arg Testing ]-------
    try:
        if varIsNotOfClass(handle, np.ndarray):
            raise ValueError('1st argument to [occa.wrapMemory] must be a numpy.ndarray')
        elif isNotANumpyType(type_):
            raise ValueError('2nd argument to [occa.wrapMemory] must be a numpy.dtype')
        elif isNotAnInteger(entries):
            raise ValueError('3rd argument to [occa.wrapMemory] must be an integer')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    return memory(_C_occa.wrapMemory(handle, entries, sizeof(type_)))

def wrapManagedMemory(handle, type_, entries):
    #---[ Arg Testing ]-------
    try:
        if varIsNotOfClass(handle, np.ndarray):
            raise ValueError('1st argument to [occa.wrapMappedMemory] must be a numpy.ndarray')
        elif isNotANumpyType(type_):
            raise ValueError('2nd argument to [occa.wrapMappedMemory] must be a numpy.dtype')
        elif isNotAnInteger(entries):
            raise ValueError('3rd argument to [occa.wrapMappedMemory] must be an integer')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    return _C_occa.wrapManagedMemory(handle, entries, sizeof(type_), typeof(type_))

def malloc(type_, entries):
    #---[ Arg Testing ]-------
    try:
        if isNotANumpyType(type_):
            raise ValueError('1st argument to [occa.malloc] must be a numpy.dtype')
        elif isNotAnInteger(entries):
            raise ValueError('2nd argument to [occa.malloc] must be an integer')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    return memory(_C_occa.malloc(entries, sizeof(type_)))

def managedAlloc(type_, entries):
    #---[ Arg Testing ]-------
    try:
        if isNotANumpyType(type_):
            raise ValueError('1st argument to [occa.managedAlloc] must be a numpy.dtype')
        elif isNotAnInteger(entries):
            raise ValueError('2nd argument to [occa.managedAlloc] must be an integer')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    return _C_occa.managedAlloc(entries, sizeof(type_), typeof(type_))

def mappedAlloc(type_, entries):
    #---[ Arg Testing ]-------
    try:
        if isNotANumpyType(type_):
            raise ValueError('1st argument to [occa.mappedAlloc] must be a numpy.dtype')
        elif isNotAnInteger(entries):
            raise ValueError('2nd argument to [occa.mappedAlloc] must be an integer')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    return memory(_C_occa.mappedAlloc(entries, sizeof(type_)))

def managedMappedAlloc(type_, entries):
    #---[ Arg Testing ]-------
    try:
        if isNotANumpyType(type_):
            raise ValueError('1st argument to [occa.managedMappedAlloc] must be a numpy.dtype')
        elif isNotAnInteger(entries):
            raise ValueError('2nd argument to [occa.managedMappedAlloc] must be an integer')

    except ValueError as e:
        print(e)
        sys.exit()
    #=========================

    return _C_occa.managedMappedAlloc(entries, sizeof(type_), typeof(type_))
#  |====================================
#=======================================

#---[ Device ]--------------------------
def printAvailableDevices():
    _C_occa.printAvailableDevices()

class device:
    def __init__(self, arg = 0):
        #---[ Arg Testing ]-------
        try:
            if isNotAnInteger(arg) and \
               isNotAString(arg):

                raise ValueError('1st argument to [occa.device.__init__] (if given) must be a string')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        if isAString(arg):
            self.handle = _C_occa.createDevice(arg)
        else:
            self.handle = 0

        self.isAllocated = True

    def free(self):
        import _C_occa

        if self.isAllocated:
            _C_occa.deviceFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()

    def mode(self):
        return _C_occa.deviceMode(self.handle)

    def setCompiler(self, compiler):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(compiler):
                raise ValueError('Argument to [occa.device.setCompiler] must be a string')
        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        _C_occa.deviceSetCompiler(self.handle, compiler)

    def setCompilerEnvScript(self, compilerEnvScript):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(compilerEnvScript):
                raise ValueError('Argument to [occa.device.setCompilerEnvScript] must be a string')
        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        _C_occa.deviceSetCompiler(self.handle, compilerEnvScript)

    def setCompilerFlags(self, compilerFlags):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(compilerFlags):
                raise ValueError('Argument to [occa.device.setCompilerFlags] must be a string')
        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        _C_occa.deviceSetCompilerFlags(self.handle, compilerFlags)

    def getCompiler(self):
        return _C_occa.deviceGetCompiler(self.handle)

    def getCompilerEnvScript(self):
        return _C_occa.deviceGetCompiler(self.handle)

    def getCompilerFlags(self):
        return _C_occa.deviceGetCompilerFlags(self.handle)

    def bytesAllocated(self):
        return _C_occa.bytesAllocated(self.handle)

    def buildKernel(self, str_, functionName, kInfo = 0):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(str_):
                raise ValueError('1st argument to [occa.device.buildKernel] must be a string')
            elif isNotAString(functionName):
                raise ValueError('2nd argument to [occa.device.buildKernel] must be a string')
            elif (kInfo != 0) and (varNotIsOfClass(kInfo, kernelInfo)):
                raise ValueError('3rd argument to [occa.device.buildKernel] (if given) must be an [occa.kernelInfo]')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        kInfo_ = (0 if (kInfo == 0) else kInfo.handle)
        return kernel(_C_occa.deviceBuildKernel(self.handle, str_, functionName, kInfo_))

    def buildKernelFromSource(self, filename, functionName, kInfo = 0):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(filename):
                raise ValueError('1st argument to [occa.device.buildKernelFromSource] must be a string')
            elif isNotAString(functionName):
                raise ValueError('2nd argument to [occa.device.buildKernelFromSource] must be a string')
            elif (kInfo != 0) and (varNotIsOfClass(kInfo, kernelInfo)):
                raise ValueError('3rd argument to [occa.device.buildKernelFromSource] (if given) must be an [occa.kernelInfo]')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        kInfo_ = (0 if (kInfo == 0) else kInfo.handle)
        return kernel(_C_occa.deviceBuildKernelFromSource(self.handle, filename, functionName, kInfo_))

    def buildKernelFromString(self, source, functionName, kInfo = 0):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(source):
                raise ValueError('1st argument to [occa.device.buildKernelFromString] must be a string')
            elif isNotAString(functionName):
                raise ValueError('2nd argument to [occa.device.buildKernelFromString] must be a string')
            elif (kInfo != 0) and (varNotIsOfClass(kInfo, kernelInfo)):
                raise ValueError('3rd argument to [occa.device.buildKernelFromString] (if given) must be an [occa.kernelInfo]')
            elif isNotAString(language):
                raise ValueError('4th argument to [occa.device.buildKernelFromString] (if given) must be a string')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        kInfo_ = (0 if (kInfo == 0) else kInfo.handle)
        return kernel(_C_occa.deviceBuildKernelFromString(self.handle, source, functionName, kInfo_))

    def buildKernelFromBinary(self, binary, functionName):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(binary):
                raise ValueError('1st argument to [occa.device.buildKernelFromBinary] must be a string')
            elif isNotAString(functionName):
                raise ValueError('2nd argument to [occa.device.buildKernelFromBinary] must be a string')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        return kernel(_C_occa.deviceBuildKernelFromBinary(self.handle, binary, functionName))

    def buildKernelFromLoopy(self, filename, functionName, kInfo = 0):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(filename):
                raise ValueError('1st argument to [occa.device.buildKernelFromSource] must be a string')
            elif isNotAString(functionName):
                raise ValueError('2nd argument to [occa.device.buildKernelFromSource] must be a string')
            elif (kInfo != 0) and (varNotIsOfClass(kInfo, kernelInfo)):
                raise ValueError('3rd argument to [occa.device.buildKernelFromSource] (if given) must be an [occa.kernelInfo]')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        kInfo_ = (0 if (kInfo == 0) else kInfo.handle)
        return kernel(_C_occa.deviceBuildKernelFromLoopy(self.handle, filename, functionName, kInfo_))

    def buildKernelFromFloopy(self, filename, functionName, kInfo = 0):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(filename):
                raise ValueError('1st argument to [occa.device.buildKernelFromSource] must be a string')
            elif isNotAString(functionName):
                raise ValueError('2nd argument to [occa.device.buildKernelFromSource] must be a string')
            elif (kInfo != 0) and (varNotIsOfClass(kInfo, kernelInfo)):
                raise ValueError('3rd argument to [occa.device.buildKernelFromSource] (if given) must be an [occa.kernelInfo]')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        kInfo_ = (0 if (kInfo == 0) else kInfo.handle)
        return kernel(_C_occa.deviceBuildKernelFromFloopy(self.handle, filename, functionName, kInfo_))

    def malloc(self, type_, entries):
        #---[ Arg Testing ]-------
        try:
            if isNotANumpyType(type_):
                raise ValueError('1st argument to [occa.device.malloc] must be a numpy.dtype')
            elif isNotAnInteger(entries):
                raise ValueError('2nd argument to [occa.device.malloc] must be an integer')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        return memory(_C_occa.deviceMalloc(self.handle, entries, sizeof(type_)))

    def managedAlloc(self, type_, entries):
        #---[ Arg Testing ]-------
        try:
            if isNotANumpyType(type_):
                raise ValueError('1st argument to [occa.device.managedAlloc] must be a numpy.dtype')
            elif isNotAnInteger(entries):
                raise ValueError('2nd argument to [occa.device.managedAlloc] must be an integer')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        return _C_occa.deviceManagedAlloc(self.handle, entries, sizeof(type_), typeof(type_))

    def mappedAlloc(self, type_, entries):
        #---[ Arg Testing ]-------
        try:
            if isNotANumpyType(type_):
                raise ValueError('1st argument to [occa.device.mappedAlloc] must be a numpy.dtype')
            elif isNotAnInteger(entries):
                raise ValueError('2nd argument to [occa.device.mappedAlloc] must be an integer')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        return memory(_C_occa.deviceMappedAlloc(self.handle, entries, sizeof(type_)))

    def managedMappedAlloc(self, type_, entries):
        #---[ Arg Testing ]-------
        try:
            if isNotANumpyType(type_):
                raise ValueError('1st argument to [occa.device.managedMappedAlloc] must be a numpy.dtype')
            elif isNotAnInteger(entries):
                raise ValueError('2nd argument to [occa.device.managedMappedAlloc] must be an integer')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        return _C_occa.deviceManagedMappedAlloc(self.handle, entries, sizeof(type_), typeof(type_))

    def flush(self):
        return _C_occa.deviceFlush(self.handle)

    def finish(self):
        return _C_occa.deviceFinish(self.handle)

    def createStream(self):
        return stream(_C_occa.deviceCreateStream(self.handle))

    def getStream(self):
        return stream(_C_occa.deviceGetStream(self.handle))

    def setStream(self, stream):
        #---[ Arg Testing ]-------
        try:
            if varIsNotOfClass(stream_, stream):
                raise ValueError('Argument to [occa.setStream] must be a occa.stream')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        return stream(_C_occa.deviceSetStream(self.handle, stream))

    def wrapStream(self, handle):
        #---[ Arg Testing ]-------
        try:
            if varIsNotAnInteger(handle):
                raise ValueError('Argument to [occa.wrapStream] must be a [void*] handle')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        return stream(_C_occa.deviceWrapStream(self.handle, handle))

class stream:
    def __init__(self, handle_ = None):
        #---[ Arg Testing ]-------
        try:
            if handle_ is not None and \
               isNotAnInteger(handle_):

                raise ValueError('1st argument to [occa.stream.__init__] (if given) must be a stream handle')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        if handle_ is not None:
            self.handle      = handle_
            self.isAllocated = True
        else:
            self.handle      = 0
            self.isAllocated = False

    def free(self):
        import _C_occa

        if self.isAllocated:
            _C_occa.streamFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()
#=======================================

#---[ Kernel ]--------------------------
class kernel:
    def __init__(self, handle_ = None):
        #---[ Arg Testing ]-------
        try:
            if handle_ is not None and \
               isNotAnInteger(handle_):

                raise ValueError('1st argument to [occa.kernel.__init__] (if given) must be a kernel handle')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        if handle_ is not None:
            self.handle      = handle_
            self.isAllocated = True
        else:
            self.handle      = 0
            self.isAllocated = False

    def free(self):
        import _C_occa

        if self.isAllocated:
            _C_occa.kernelFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()

    def __call__(self, *args):
        argList = _C_occa.createArgumentList()

        for i in xrange(len(args)):
            arg = args[i]

            if varIsOfClass(arg, np.ndarray):
                argType = _C_occa.ptr(arg.ctypes.data)
                _C_occa.argumentListAddArg(argList, i, argType)
            elif varIsOfClass(arg, memory):
                _C_occa.argumentListAddArg(argList, i, arg.handle)
            else:
                #---[ Arg Testing ]-------
                try:
                    if isNotANumpyType(arg):
                        raise ValueError('Argument to an [occa.kernel()] launch must be either:\n'\
                                         '    - OCCA-allocated numpy.ndarray\n'        \
                                         '    - occa.memory object\n'                  \
                                         '    - numpy.dtype (like numpy.int32, numpy.float32, ...)')

                except ValueError as e:
                    print(e)
                    sys.exit()
                #=========================

                argType = getattr(_C_occa, nameof(arg))(arg)
                _C_occa.argumentListAddArg(argList, i, argType)

        _C_occa.kernelRun(self.handle, argList)

        _C_occa.argumentListFree(argList)

    def mode(self):
        return _C_occa.kernelMode(self.handle)

    def name(self):
        return _C_occa.kernelName(self.handle)

    def getDevice(self):
        return device(_C_occa.kernelGetDevice(self.handle))

class kernelInfo:
    def __init__(self, handle_ = None):
        #---[ Arg Testing ]-------
        try:
            if handle_ is not None and \
               isNotAnInteger(handle_):

                raise ValueError('1st argument to [occa.kernelInfo.__init__] (if given) must be a kernelInfo handle')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        if handle_ is not None:
            self.handle      = handle_
        else:
            self.handle      = _C_occa.createKernelInfo()

        self.isAllocated = True

    def free(self):
        import _C_occa

        if self.isAllocated:
            _C_occa.kernelInfoFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()

    def addDefine(self, macro, value):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(macro):
                raise ValueError('1st argument to [occa.kernelInfo.addDefine] must be a string')
            elif not hasattr(value, '__str__'):
                raise ValueError('2nd argument to [occa.kernelInfo.addDefine] must be have an [__str__] method')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        _C_occa.kernelInfoAddDefine(self.handle, macro, value.__str__())

    def addInclude(self, filename):
        #---[ Arg Testing ]-------
        try:
            if isNotAString(filename):
                raise ValueError('1st argument to [occa.kernelInfo.addInclude] must be a string')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        _C_occa.kernelInfoAddInclude(self.handle, filename)
#=======================================

#---[ Memory ]--------------------------
class memory:
    def __init__(self, handle_ = None):
        #---[ Arg Testing ]-------
        try:
            if handle_ is not None and \
               isNotAnInteger(handle_):

                raise ValueError('1st argument to [occa.memory.__init__] (if given) must be a memory handle')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        if handle_ is not None:
            self.handle      = handle_
            self.isAllocated = True
        else:
            self.handle      = 0
            self.isAllocated = False

    def free(self):
        import _C_occa

        if self.isAllocated:
            _C_occa.memoryFree(self.handle)
            self.isAllocated = False

    def __del__(self):
        self.free()

    def mode(self):
        return _C_occa.memoryMode(self.handle)

    def getMemoryHandle(self):
        return _C_occa.memoryGetMemoryHandle(self.handle)

    def getMappedPointer(self):
        return _C_occa.memoryGetMappedPointer(self.handle)

    def getTextureHandle(self):
        return _C_occa.memoryGetTextureHandle(self.handle)

    def copyFrom(self, src, bytes_ = 0, offset1 = 0, offset2 = 0):
        #---[ Arg Testing ]-------
        try:
            if isNotAMemoryType(src):
                raise ValueError('1st argument to [occa.memory.copyFrom] must be a [numpy.ndarray] or [occa.memory]')
            elif isNotAnInteger(bytes_):
                raise ValueError('2nd argument to [occa.memory.copyFrom] (if given) must be an integer')
            elif isNotAnInteger(offset1):
                raise ValueError('3rd argument to [occa.memory.copyFrom] (if given) must be an integer')
            elif isNotAnInteger(offset2):
                raise ValueError('4th argument to [occa.memory.copyFrom] (if given) must be an integer')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        memcpy(self, src, bytes_, offset1, offset2)

    def copyTo(self, dest, bytes_ = 0, offset1 = 0, offset2 = 0):
        #---[ Arg Testing ]-------
        try:
            if isNotAMemoryType(dest):
                raise ValueError('1st argument to [occa.memory.copyTo] must be a [numpy.ndarray] or [occa.memory]')
            elif isNotAnInteger(bytes_):
                raise ValueError('2nd argument to [occa.memory.copyTo] (if given) must be an integer')
            elif isNotAnInteger(offset1):
                raise ValueError('3rd argument to [occa.memory.copyTo] (if given) must be an integer')
            elif isNotAnInteger(offset2):
                raise ValueError('4th argument to [occa.memory.copyTo] (if given) must be an integer')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        memcpy(dest, self, bytes_, offset1, offset2)

    def asyncCopyFrom(self, src, bytes_ = 0, offset1 = 0, offset2 = 0):
        #---[ Arg Testing ]-------
        try:
            if isNotAMemoryType(src):
                raise ValueError('1st argument to [occa.memory.asyncCopyFrom] must be a [numpy.ndarray] or [occa.memory]')
            elif isNotAnInteger(bytes_):
                raise ValueError('2nd argument to [occa.memory.asyncCopyFrom] (if given) must be an integer')
            elif isNotAnInteger(offset1):
                raise ValueError('3rd argument to [occa.memory.asyncCopyFrom] (if given) must be an integer')
            elif isNotAnInteger(offset2):
                raise ValueError('4th argument to [occa.memory.asyncCopyFrom] (if given) must be an integer')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        asyncMemcpy(self, src, bytes_, offset1, offset2)

    def asyncCopyTo(self, dest, bytes_ = 0, offset1 = 0, offset2 = 0):
        #---[ Arg Testing ]-------
        try:
            if isNotAMemoryType(dest):
                raise ValueError('1st argument to [occa.memory.asyncCopyTo] must be a [numpy.ndarray] or [occa.memory]')
            elif isNotAnInteger(bytes_):
                raise ValueError('2nd argument to [occa.memory.asyncCopyTo] (if given) must be an integer')
            elif isNotAnInteger(offset1):
                raise ValueError('3rd argument to [occa.memory.asyncCopyTo] (if given) must be an integer')
            elif isNotAnInteger(offset2):
                raise ValueError('4th argument to [occa.memory.asyncCopyTo] (if given) must be an integer')

        except ValueError as e:
            print(e)
            sys.exit()
        #=========================

        asyncMemcpy(dest, self, bytes_, offset1, offset2)
#=======================================
