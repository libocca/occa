import _C_occa
import numpy as np

#---[ Setup ]---------------------------
def sizeof(npType):
    return np.dtype(npType).itemsize

def typeof(npType):
    return np.dtype(npType).num
#=======================================

#---[ Globals & Flags ]-----------------
def setVerboseCompilation(value):
    _C_occa.setVerboseCompilation(value)
#=======================================

#----[ Background Device ]--------------
#  |---[ Device ]-----------------------
def setDevice(device):
    _C_occa.setDevice(device)

def setDeviceFromInfo(infos):
    _C_occa.setDeviceFromInfo(infos)

def getCurrentDevice():
    return device(_C_occa.getCurrentDevice())

def setCompiler(compiler):
    _C_occa.setCompiler(compiler)

def setCompilerEnvScript(compilerEnvScript):
    _C_occa.setCompilerEnvScript(compilerEnvScript)

def setCompilerFlags(compilerFlags):
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

def setStream(stream):
    _C_occa.setStream(stream)

def wrapStream(handle):
    return _C_occa.wrapStream(handle)
#  |====================================

#  |---[ Kernel ]-----------------------
def buildKernel(str_, functionName, kInfo = 0):
    return kernel(_C_occa.buildKernel(filename, functionName, info))

def buildKernelFromSource(filename, functionName, kInfo = 0):
    return kernel(_C_occa.buildKernelFromSource(filename, functionName, info))

def buildKernelFromString(source, functionName, kInfo = 0, language = "OKL"):
    return kernel(_C_occa.buildKernelFromString(source, functionName, kInfo, language))

def buildKernelFromBinary(binary, functionName):
    return kernel(_C_occa.buildKernelFromBinary(filename, functionName))

def buildKernelFromLoopy(filename, functionName, kInfo = 0):
    return kernel(_C_occa.buildKernelFromLoopy(filename, functionName, info))

def buildKernelFromFloopy():
    return kernel(_C_occa.buildKernelFromFloopy(filename, functionName, info))
#  |====================================

#  |---[ Memory ]-----------------------
def wrapMemory(handle, entries, type_):
    return memory(_C_occa.wrapMemory(handle, entries, sizeof(type_)))

def wrapManagedMemory(handle, entries, type_):
    return _C_occa.wrapManagedMemory(handle, entries, sizeof(type_), typeof(type_))

def malloc(entries, type_):
    return memory(_C_occa.malloc(entries, sizeof(type_)))

def managedAlloc():
    return _C_occa.managedAlloc(entries, sizeof(type_), typeof(type_))

def malloc(entries, type_):
    return memory(_C_occa.mappedAlloc(entries, sizeof(type_)))

def managedAlloc():
    return _C_occa.managedMappedAlloc(entries, sizeof(type_), typeof(type_))
#  |====================================
#=======================================

#---[ Device ]--------------------------
def printAvailableDevices():
    _C_occa.printAvailableDevices()

class device:
    def deviceInfoAppend():
        ;

    def deviceInfoFree():
        ;

    def createDevice():
        ;

    def createDeviceFromInfo():
        ;

    def deviceMode():
        ;

    def deviceSetCompiler():
        ;

    def deviceSetCompilerEnvScript():
        ;

    def deviceSetCompilerFlags():
        ;

    def deviceSetCompiler():
        ;

    def deviceSetCompilerEnvScript():
        ;

    def deviceSetCompilerFlags():
        ;

    def deviceBytesAllocated():
        ;

    def deviceBuildKernel():
        ;

    def deviceBuildKernelFromSource():
        ;

    def deviceBuildKernelFromString():
        ;

    def deviceBuildKernelFromBinary():
        ;

    def deviceBuildKernelFromLoopy():
        ;

    def deviceBuildKernelFromFloopy():
        ;

    def deviceMalloc():
        ;

    def deviceManagedAlloc():
        ;

    def deviceMappedAlloc():
        ;

    def deviceManagedMappedAlloc():
        ;

    def deviceFlush():
        ;

    def deviceFinish():
        ;

    def deviceCreateStream():
        ;

    def deviceGetStream():
        ;

    def deviceSetStream():
        ;

    def deviceWrapStream():
        ;

    def deviceStreamFree():
        ;

    def deviceFree():
        ;
#=======================================

#---[ Kernel ]--------------------------
class kernel:
    def kernelMode():
        ;

    def kernelName():
        ;

    def kernelGetDevice():
        ;

    def createArgumentList():
        ;

    def argumentListClear():
        ;

    def argumentListFree():
        ;

    def argumentListAddArg():
        ;

    def kernelRun():
        ;

    def createKernelInfo():
        ;

    def kernelInfoAddDefine():
        ;

    def kernelInfoAddInclude():
        ;

    def kernelFree():
        ;
#=======================================

#---[ Memory ]--------------------------
class memory:
    def memoryMode():
        ;

    def memoryGetMemoryHandle():
        ;

    def memoryGetMappedPointer():
        ;

    def memoryGetTextureHandle():
        ;

    def memcpy():
        ;
#=======================================