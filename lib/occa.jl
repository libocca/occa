module occa

#---[ Setup ]-----------------
macro libocca()
    return bytestring(ENV["OCCA_DIR"], "/lib/libocca.so")
end

#---[ Types ]-----------------
type device
    cDevice::Ptr{Void}
end

function device(infos::String)
    cDevice = ccall((:occaGetDevice, @libocca()),
                    Ptr{Void},
                    (Ptr{Uint8},),
                    bytestring(infos));

   return device(cDevice);
end

type stream
    cStream::Ptr{Void}

end

function stream(ptr::Ptr{Void})
    return stream(cStream);
end

type kernel
    cKernel::Ptr{Void}
end
function kernel(cKernel)
    return kernel(cKernel);
end


type kernelInfo
    cKernelInfo::Ptr{Void}
end

function kernelInfo(ptr::Ptr{Void})
    return kernlInfo(cKernelInfo);
end


type memory
    cMemory::Ptr{Void}
    cTypes
end

function memory(cMemory, cTypes)
    return memory(cMemory,cTypes);
end


#---[ Device ]----------------
function finalizer(d::device)
    ccall((:occaDeviceFree, @libocca()),
          Void,
          (Ptr{Void},),
          d.cDevice)
end

function mode(d::device)
    cMode = ccall((:occaDeviceMode, @libocca()),
                  Ptr{Uint8},
                  (Ptr{Void},), d.cDevice)

    return bytestring(cMode)
end

function setCompiler(d::device,
                     compiler::String)
    ccall((:occaDeviceSetCompiler, @libocca()),
          Void,
          (Ptr{Void}, Ptr{Uint8},),
          d.cDevice, bytestring(compiler))
end

function setCompilerFlags(d::device,
                          compilerFlags::String)
    ccall((:occaDeviceSetCompilerFlags, @libocca()),
          Void,
          (Ptr{Void}, Ptr{Uint8},),
          d.cDevice, bytestring(compilerFlags))
end

function buildKernelFromSource(d::device,
                               filename::String,
                               functionName::String,
                               info = C_NULL)
    if info == C_NULL
        cKernel = ccall((:occaBuildKernelFromSource, @libocca()),
                        Ptr{Void},
                        (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}, Ptr{Void},),
                        d.cDevice,
                        bytestring(filename),
                        bytestring(functionName),
                        C_NULL)
    else
        cKernel = ccall((:occaBuildKernelFromSource, @libocca()),
                        Ptr{Void},
                        (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}, Ptr{Void},),
                        d.cDevice,
                        bytestring(filename),
                        bytestring(functionName),
                        info.cKernelInfo)
    end

    return kernel(cKernel)
end

function buildKernelFromBinary(d::device,
                               filename::String,
                               functionName::String)
    cKernel = ccall((:occaBuildKernelFromBinary, @libocca()),
                    Ptr{Void},
                    (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8},),
                    d.cDevice,
                    bytestring(filename),
                    bytestring(functionName))

    return kernel(cKernel)
end

function malloc(d::device, source::Array)
    cTypes = typeof(source[1])
    bytes  = length(source) * sizeof(cTypes)

    convert(Uint, bytes)

    cMemory = ccall((:occaDeviceMalloc, @libocca()),
                    Ptr{Void},
                    (Ptr{Void}, Uint, Ptr{Void},),
                    d.cDevice, bytes, pointer(source))

    return memory(cMemory, cTypes)
end

function malloc(d::device, entriesAndType)
    if length(entriesAndType) != 2
        error("malloc second argument must be a tuple of (bytes, type) or Array")
    end

    cTypes = entriesAndType[2]

    bytes  = entriesAndType[1] * sizeof(cTypes)

    convert(Uint, bytes)

    cMemory = ccall((:occaDeviceMalloc, @libocca()),
                    Ptr{Void},
                    (Ptr{Void}, Uint, Ptr{Void},),
                    d.cDevice, bytes, C_NULL)

    return memory(cMemory, cTypes)
end

function createStream(d::device)
    cStream = ccall((:occaGenStream, @libocca()),
                    Ptr{Void},
                    (Ptr{Void},),
                    d.cDevice)

    return stream(cStream)
end

function getStream(d::device)
    cStream = ccall((:occaGetStream, @libocca()),
                    Ptr{Void},
                    (Ptr{Void},),
                    d.cDevice)

    return stream(cStream)
end

function setStream(d::device, s::stream)
    ccall((:occaSetStream, @libocca()),
          Void,
          (Ptr{Void}, Ptr{Void},),
          d.cDevice, s.cStream)
end

#---[ Kernel ]----------------
function finalizer(k::kernel)
    ccall((:occaKernelFree, @libocca()),
          Void,
          (Ptr{Void},),
          k.cKernel)
end

function mode(k::kernel)
    cMode = ccall((:occaKernelMode, @libocca()),
                  Ptr{Uint8},
                  (Ptr{Void},),
                  k.cKernel)

    return bytestring(cMode)
end

function getPreferredDimSize(k::kernel)
    return ccall((:occaKernelPreferredDimSize, @libocca()),
                 Int32,
                 (Ptr{Void},),
                 k.cKernel)
end

function setWorkingDims(k::kernel,
                        dims, items, groups)
    convert(Int32, dims)

    items_  = ones(Uint, 3)
    groups_ = ones(Uint, 3)

    for i = 1:dims
        items_[i]  = items[i]
        groups_[i] = groups[i]
    end

    ccall((:occaKernelSetAllWorkingDims, @libocca()),
          Void,
          (Ptr{Void},
           Int32,
           Uint, Uint, Uint,
           Uint, Uint, Uint,),
          k.cKernel,
          dims,
          items_[1] , items_[2] , items_[3],
          groups_[1], groups_[2], groups_[3])
end

argType(arg::Int8)  = ccall((:occaChar , @libocca()), Ptr{Void}, (Int8,) , arg)
argType(arg::Uint8) = ccall((:occaUChar, @libocca()), Ptr{Void}, (Uint8,), arg)

argType(arg::Int16)  = ccall((:occaShort , @libocca()), Ptr{Void}, (Int16,) , arg)
argType(arg::Uint16) = ccall((:occaUShort, @libocca()), Ptr{Void}, (Uint16,), arg)

argType(arg::Int32)  = ccall((:occaInt , @libocca()), Ptr{Void}, (Int32,) , arg)
argType(arg::Uint32) = ccall((:occaUInt, @libocca()), Ptr{Void}, (Uint32,), arg)

argType(arg::Int64)  = ccall((:occaLong , @libocca()), Ptr{Void}, (Int64,) , arg)
argType(arg::Uint64) = ccall((:occaULong, @libocca()), Ptr{Void}, (Uint64,), arg)

argType(arg::Float32) = ccall((:occaFloat , @libocca()), Ptr{Void}, (Float32,) , arg)
argType(arg::Float64) = ccall((:occaDouble, @libocca()), Ptr{Void}, (Float64,) , arg)

function runKernel(k::kernel, args...)
    argList = ccall((:occaGenArgumentList, @libocca()),
                    Ptr{Void}, ())

    pos = convert(Int32, 0)

    for arg in args
        if isa(arg, memory)
            ccall((:occaArgumentListAddArg, @libocca()),
                  Void,
                  (Ptr{Void}, Int32, Ptr{Void},),
                  argList, pos, arg.cMemory)
        else
            if length(arg) != 2
                error("Kernel argument should be in the form of (value, type)")
            end

            arg_ = arg[1]
            convert(arg[2], arg_)

            cArg = argType(arg_)

            ccall((:occaArgumentListAddArg, @libocca()),
                  Void,
                  (Ptr{Void}, Int32, Ptr{Void},),
                  argList, pos, cArg)
        end

        pos += 1
    end

    ccall((:occaKernelRun_, @libocca()),
          Void,
          (Ptr{Void}, Ptr{Void},),
          k.cKernel, argList)

    ccall((:occaArgumentListFree, @libocca()),
          Void,
          (Ptr{Void},),
          argList)
end

function timeTaken(k::kernel)
    return ccall((:occaKernelTimeTaken, @libocca()),
                 Float64,
                 (Ptr{Void},),
                 k.cKernel)
end

function addDefine(info::kernelInfo, macro_::String, value::String)
    occaValue = ccall((:occaString, @libocca()),
                      Ptr{Void},
                      (Ptr{Uint8},),
                      bytestring(value))

    ccall((:occaKernelInfoAddDefine, @libocca()),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Ptr{Void},),
          info.cKernelInfo, bytestring(macro_), occaValue)
end

function finalizer(info::kernelInfo)
    ccall((:occaKernelInfoFree, @libocca()),
          Void,
          (Ptr{Void},),
          info.cKernelInfo)
end

#---[ Memory ]----------------
function finalizer(m::memory)
    ccall((:occaMemoryFree, @libocca()),
          Void,
          (Ptr{Void},),
          m.cMemory)
end

function mode(m::memory)
    cMode = ccall((:occaMemoryMode, @libocca()),
                  Ptr{Uint8},
                  (Ptr{Void},),
                  m.cMemory)

    return bytestring(cMode)
end

function memcpy(destTuple, srcTuple, bytes::Number = 0)
    if isa(destTuple, memory)
        dest = destTuple.cMemory

        destOffset = 0
        convert(Uint, destOffset)

        destIsAMemory = true
    elseif isa(destTuple, Array)
        dest = pointer(destTuple)

        destOffset = 0
        convert(Uint, destOffset)

        destIsAMemory = false
    else
        dest = destTuple[1]

        if isa(dest, memory)
            dest = dest.cMemory
            destIsAMemory = true
        else
            dest = pointer(destTuple[1])
            destIsAMemory = false
        end

        destOffset = destTuple[2]
        convert(Uint, destOffset)
    end

    if isa(srcTuple, memory)
        src = srcTuple.cMemory

        srcOffset = 0
        convert(Uint, srcOffset)

        srcIsAMemory = true
    elseif isa(srcTuple, Array)
        src = pointer(srcTuple)

        srcOffset = 0
        convert(Uint, srcOffset)

        srcIsAMemory = false
    else
        src = srcTuple[1]

        if isa(src, memory)
            src = src.cMemory
            srcIsAMemory = true
        else
            src = pointer(srcTuple[1])
            srcIsAMemory = false
        end

        srcOffset = srcTuple[2]
        convert(Uint, srcOffset)
    end

    convert(Uint, bytes)

    if destIsAMemory
        if srcIsAMemory
            ccall((:occaCopyMemToMem, @libocca()),
                  Void,
                  (Ptr{Void}, Ptr{Void}, Uint, Uint, Uint,),
                  dest, src, bytes, destOffset, srcOffset)
        else
            ccall((:occaCopyPtrToMem, @libocca()),
                  Void,
                  (Ptr{Void}, Ptr{Void}, Uint, Uint,),
                  dest, src, bytes, destOffset)
        end
    else
        if srcIsAMemory
            ccall((:occaCopyMemToPtr, @libocca()),
                  Void,
                  (Ptr{Void}, Ptr{Void}, Uint, Uint,),
                  dest, src, bytes, srcOffset)
        else
            error("One of the arguments should be an OCCA memory type")
        end
    end
end

function swap(a::memory, b::memory)
    tmp       = a.cMemory
    a.cMemory = b.cMemory
    b.cMemory = tmp
end

end
