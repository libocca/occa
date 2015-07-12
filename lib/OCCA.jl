module OCCA

macro libocca()
    return bytestring(ENV["OCCA_DIR"], "/lib/libocca.so")
end

macro libCall(name)
    return :($name, @libocca())
end

#|---[ Globals & Flags ]-----------------
function set_verbose_compilation(yn::Bool)
    ccall(@libCall(:occaSetVerboseCompilation),
          Void,
          ({Uint8},),
          yn)
end
#|=======================================

#|----[ Background Device ]--------------
#  |---[ Device ]-----------------------
type Device
    handle::Ptr{Void}

    function Device()
        ret = new()

        ret.handle = C_NULL

        return ret
    end

    function Device(handle_::Ptr{Void})
        ret = new()

        ret.handle = handle_

        return ret
    end

    function Device(infos::String)
        ret = Device()

        ret.handle = ccall(@libCall(:occaCreateDevice),
                           Ptr{Void},
                           (Ptr{Uint8},),
                           bytestring(infos))

        return ret
    end

    function Device(d::Device)
        ret = new()

        ret.handle = d.handle

        return ret
    end
end

type Stream
    handle::Ptr{Void}

    function Stream()
        ret = new()

        ret.handle = C_NULL

        return ret
    end

    function Stream(handle_::Ptr{Void})
        ret = new()

        ret.handle = handle_

        return ret
    end

    function Stream(s::Stream)
        ret = new()

        ret.handle = s.handle

        return ret
    end
end

function set(; compiler = "", flags = "", env_script = "")
    if 0 < length(compiler)
        ccall(@libCall(:occaSetCompiler),
              Void,
              (Ptr{Uint8},),
              bytestring(compiler))
    end

    if 0 < length(flags)
        ccall(@libCall(:occaSetCompilerFlags),
              Void,
              (Ptr{Uint8},),
              bytestring(flags))
    end

    if 0 < length(env_script)
        ccall(@libCall(:occaSetCompilerEnvScript),
              Void,
              (Ptr{Uint8},),
              bytestring(env_script))
    end
end

function get_compiler()
    str = ccall(@libCall(:occaGetCompiler),
                Ptr{Uint8}, ())

    return bytestring(str)
end

function get_compiler_flags()
    str = ccall(@libCall(:occaGetCompilerFlags),
                Ptr{Uint8}, ())

    return bytestring(str)
end

function get_compiler_env_script()
    str = ccall(@libCall(:occaGetCompilerEnvScript),
                Ptr{Uint8}, ())

    return bytestring(str)
end

function flush()
    ccall(@libCall(:occaFlush),
          Void, ())
end

function finish()
    ccall(@libCall(:occaFinish),
          Void, ())
end

function create_stream()
    return Stream(ccall(@libCall(:occaCreateStream),
                        Ptr{Void}, ()))
end

function get_stream()
    return Stream(ccall(@libCall(:occaGetStream),
                        Ptr{Void}, ()))
end

function set_stream(s::Stream)
    return Stream(ccall(@libCall(:occaSetStream),
                        Void,
                        (Ptr{Void},),
                        s.handle))
end

function wrap_stream(sHandle::Ptr{Void})
    return Stream(ccall(@libCall(:occaWrapStream),
                        Ptr{Void},
                        (Ptr{Void},),
                        sHandle))
end
#  |====================================

#  |---[ Memory ]-----------------------
type Memory
    handle::Ptr{Void}

    function Memory()
        ret = new()

        ret.handle = C_NULL

        return ret
    end

    function Memory(handle_::Ptr{Void})
        ret = new()

        ret.handle = handle_

        return ret
    end

    function Memory(m::Memory)
        ret = new()

        ret.handle = m.handle

        return ret
    end
end

function wrap_memory(arr::Array; managed = false)
    entries = length(arr)

    if entries == 0
        if !managed
            return Memory()
        else
            return []
        end
    end

    bytes = entries * sizeof(arr[1])

    if !managed
        ptr = ccall(@libCall(:occaWrapMemory),
                    Ptr{Void},
                    (Uint64, Ptr{Void},),
                    pointer(arr), bytes)
    else
        ptr = ccall(@libCall(:occaWrapManagedMemory),
                    Ptr{Void},
                    (Uint64, Ptr{Void},),
                    pointer(arr), bytes)
    end

    if !managed
        return Memory(ptr)
    else
        return pointer_to_array(convert(Ptr{typeof(arr[1])}, ptr),
                                entries)
    end
end

function malloc(arr::Array; managed = false, mapped = false)
    entries = length(arr)

    if entries == 0
        if !managed
            return Memory()
        else
            return []
        end
    end

    return malloc(typeof(arr[1]), length(arr), pointer(arr), managed = managed, mapped = mapped)
end

function managed_malloc(t::Type, entries, source = C_NULL)

    return malloc(t, entries, source, managed = true, mapped = false)
end

function managed_malloc(arr::Array)
    return malloc(arr, managed = true, mapped = false)
end

function managed_mapped_malloc(t::Type, entries, source = C_NULL)

    return malloc(t, entries, source, managed = true, mapped = true)
end

function managed_mapped_malloc(arr::Array)
    return malloc(arr, managed = true, mapped = true)
end

function malloc(t::Type, entries, source = C_NULL; managed = false, mapped = false)
    bytes = entries * sizeof(t)

    if !managed
        if !mapped
            ptr = ccall(@libCall(:occaMalloc),
                        Ptr{Void},
                        (Uint64, Ptr{Void},),
                        bytes, source)
        else
            ptr = ccall(@libCall(:occaMappedAlloc),
                        Ptr{Void},
                        (Uint64, Ptr{Void},),
                        bytes, source)
        end
    else
        if !mapped
            ptr = ccall(@libCall(:occaManagedAlloc),
                        Ptr{Void},
                        (Uint64, Ptr{Void},),
                        bytes, source)
        else
            ptr = ccall(@libCall(:occaManagedMappedAlloc),
                        Ptr{Void},
                        (Uint64, Ptr{Void},),
                        bytes, source)
        end
    end

    if !managed
        return Memory(ptr)
    else
        return pointer_to_array(convert(Ptr{t}, ptr),
                                entries)
    end
end
#  |====================================

#  |---[ Kernel ]-----------------------
type Kernel
    handle::Ptr{Void}

    function Kernel()
        ret = new()

        ret.handle = C_NULL

        return ret
    end

    function Kernel(handle_::Ptr{Void})
        ret = new()

        ret.handle = handle_

        return ret
    end

    function Kernel(k::Kernel)
        ret = new()

        ret.handle = k.handle

        return ret
    end
end

occaType(v::Int8)    = ccall(@libCall(:occaChar)  , Ptr{Void}, (Int8,)     , v)
occaType(v::Uint8)   = ccall(@libCall(:occaUChar) , Ptr{Void}, (Uint8,)    , v)

occaType(v::Int16)   = ccall(@libCall(:occaShort) , Ptr{Void}, (Int16,)    , v)
occaType(v::Uint16)  = ccall(@libCall(:occaUShort), Ptr{Void}, (Uint16,)   , v)

occaType(v::Int32)   = ccall(@libCall(:occaInt)   , Ptr{Void}, (Int32,)    , v)
occaType(v::Uint32)  = ccall(@libCall(:occaUInt)  , Ptr{Void}, (Uint32,)   , v)

occaType(v::Int64)   = ccall(@libCall(:occaLong)  , Ptr{Void}, (Int64,)    , v)
occaType(v::Uint64)  = ccall(@libCall(:occaULong) , Ptr{Void}, (Uint64,)   , v)

occaType(v::Float32) = ccall(@libCall(:occaFloat) , Ptr{Void}, (Float32,)  , v)
occaType(v::Float64) = ccall(@libCall(:occaDouble), Ptr{Void}, (Float64,)  , v)

occaType(v::Array)   = ccall(@libCall(:occaPtr)   , Ptr{Void}, (Ptr{Void},), pointer(v))

occaType(v::Memory)  = v.handle

type KernelInfo
    handle::Ptr{Void}

    function KernelInfo()
        ret = new()

        ret.handle = C_NULL

        return ret
    end

    function KernelInfo(handle_::Ptr{Void})
        ret = new()

        ret.handle = handle_

        return ret
    end

    function KernelInfo(kInfo::KernelInfo)
        ret = new()

        ret.handle = kInfo.handle

        return ret
    end
end

function build_kernel(filename::String, functionName::String, kInfo::KernelInfo; from = :auto)
    return build_kernel(filename, functionName, kInfo.handle, from=from)
end

function build_kernel(filename::String, functionName::String; from = :auto)
    return build_kernel(filename, functionName, C_NULL, from=from)
end

function build_kernel(filename::String, functionName::String, kInfo::Ptr{Void}; from = :auto)
    if from == :auto
        return Kernel(ccall(@libCall(:occaBuildKernel),
                            Ptr{Void},
                            (Ptr{Uint8}, Ptr{Uint8}, Ptr{Void},),
                            bytestring(filename),
                            bytestring(functionName),
                            kInfo))
    elseif from == :source
        return Kernel(ccall(@libCall(:occaBuildKernelFromSource),
                            Ptr{Void},
                            (Ptr{Uint8}, Ptr{Uint8}, Ptr{Void},),
                            bytestring(filename),
                            bytestring(functionName),
                            kInfo))
    elseif from == :string
        return Kernel(ccall(@libCall(:occaBuildKernelFromString),
                            Ptr{Void},
                            (Ptr{Uint8}, Ptr{Uint8}, Ptr{Void},),
                            bytestring(filename),
                            bytestring(functionName),
                            kInfo))
    elseif from == :binary
        return Kernel(ccall(@libCall(:occaBuildKernelFromBinary),
                            Ptr{Void},
                            (Ptr{Uint8}, Ptr{Uint8},),
                            bytestring(filename),
                            bytestring(functionName)))
    end
end
#  |====================================
#|=======================================

#|---[ Device ]--------------------------
function free!(d::Device)
    if d.handle == C_NULL
        return
    end

    ccall(@libCall(:occaDeviceFree),
          Void,
          (Ptr{Void},),
          d.handle)

    d.handle = C_NULL
end

function mode(d::Device)
    strMode = ccall(@libCall(:occaDeviceMode),
                    Ptr{Uint8},
                    (Ptr{Void},),
                    d.handle)

    return bytestring(strMode)
end

function set!(d::Device; compiler = "", flags = "", env_script = "")
    if 0 < length(compiler)
        ccall(@libCall(:occaDeviceSetCompiler),
              Void,
              (Ptr{Void}, Ptr{Uint8},),
              d.handle, bytestring(compiler))
    end

    if 0 < length(flags)
        ccall(@libCall(:occaDeviceSetCompilerFlags),
              Void,
              (Ptr{Void}, Ptr{Uint8},),
              d.handle, bytestring(flags))
    end

    if 0 < length(env_script)
        ccall(@libCall(:occaDeviceSetCompilerEnvScript),
              Void,
              (Ptr{Void}, Ptr{Uint8},),
              d.handle, bytestring(env_script))
    end
end

function get_compiler(d::Device)
    str = ccall(@libCall(:occaDeviceGetCompiler),
                Ptr{Uint8},
                (Ptr{Void},),
                d.handle)

    return bytestring(str)
end

function get_compiler_flags(d::Device)
    str = ccall(@libCall(:occaDeviceGetCompilerFlags),
                Ptr{Uint8},
                (Ptr{Void},),
                d.handle)

    return bytestring(str)
end

function get_compiler_env_script(d::Device)
    str = ccall(@libCall(:occaDeviceGetCompilerEnvScript),
                Ptr{Uint8},
                (Ptr{Void},),
                d.handle)

    return bytestring(str)
end

function bytes_allocated(d::Device)
    return ccall(@libCall(:occaDeviceBytesAllocated),
                 Uint64,
                 (Ptr{Void},),
                 d.handle)
end

function build_kernel(d::Device, filename::String, functionName::String, kInfo::KernelInfo; from = :auto)
    return build_kernel(d, filename, functionName, kInfo.handle, from=from)
end

function build_kernel(d::Device, filename::String, functionName::String; from = :auto)
    return build_kernel(d, filename, functionName, C_NULL, from=from)
end

function build_kernel(d::Device, filename::String, functionName::String, kInfo::Ptr{Void}; from = :auto)
    if from == :auto
        return Kernel(ccall(@libCall(:occaDeviceBuildKernel),
                            Ptr{Void},
                            (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}, Ptr{Void},),
                            d.handle,
                            bytestring(filename),
                            bytestring(functionName),
                            kInfo))
    elseif from == :source
        return Kernel(ccall(@libCall(:occaDeviceBuildKernelFromSource),
                            Ptr{Void},
                            (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}, Ptr{Void},),
                            d.handle,
                            bytestring(filename),
                            bytestring(functionName),
                            kInfo))
    elseif from == :string
        return Kernel(ccall(@libCall(:occaDeviceBuildKernelFromString),
                            Ptr{Void},
                            (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}, Ptr{Void},),
                            d.handle,
                            bytestring(filename),
                            bytestring(functionName),
                            kInfo))
    elseif from == :binary
        return Kernel(ccall(@libCall(:occaDeviceBuildKernelFromBinary),
                            Ptr{Void},
                            (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8},),
                            d.handle,
                            bytestring(filename),
                            bytestring(functionName)))
    end
end

function malloc(d::Device, arr::Array; managed = false, mapped = false)
    entries = length(arr)

    if entries == 0
        if !managed
            return Memory()
        else
            return []
        end
    end

    return malloc(d, typeof(arr[1]), length(arr), pointer(arr), managed = managed, mapped = mapped)
end

function managed_malloc(d::Device, t::Type, entries, source = C_NULL)

    return malloc(d, t, entries, source, managed = true, mapped = false)
end

function managed_malloc(d::Device, arr::Array)
    return malloc(d, arr, managed = true, mapped = false)
end

function managed_mapped_malloc(d::Device, t::Type, entries, source = C_NULL)

    return malloc(d, t, entries, source, managed = true, mapped = true)
end

function managed_mapped_malloc(d::Device, arr::Array)
    return malloc(d, arr, managed = true, mapped = true)
end

function malloc(d::Device, t::Type, entries, source = C_NULL; managed = false, mapped = false)
    bytes = entries * sizeof(t)

    if !managed
        if !mapped
            ptr = ccall(@libCall(:occaDeviceMalloc),
                        Ptr{Void},
                        (Ptr{Void}, Uint64, Ptr{Void},),
                        d.handle, bytes, source)
        else
            ptr = ccall(@libCall(:occaDeviceMappedAlloc),
                        Ptr{Void},
                        (Ptr{Void}, Uint64, Ptr{Void},),
                        d.handle, bytes, source)
        end
    else
        if !mapped
            ptr = ccall(@libCall(:occaDeviceManagedAlloc),
                        Ptr{Void},
                        (Ptr{Void}, Uint64, Ptr{Void},),
                        d.handle, bytes, source)
        else
            ptr = ccall(@libCall(:occaDeviceManagedMappedAlloc),
                        Ptr{Void},
                        (Ptr{Void}, Uint64, Ptr{Void},),
                        d.handle, bytes, source)
        end
    end

    if !managed
        return Memory(ptr)
    else
        return pointer_to_array(convert(Ptr{t}, ptr),
                                entries)
    end
end

function flush(d::Device)
    ccall(@libCall(:occaDeviceFlush),
          Void,
          (Ptr{Void},),
          d.handle)
end

function finish(d::Device)
    ccall(@libCall(:occaDeviceFinish),
          Void,
          (Ptr{Void},),
          d.handle)
end

function create_stream(d::Device)
    return Stream(ccall(@libCall(:occaDeviceCreateStream),
                        Ptr{Void},
                        (Ptr{Void},),
                        d.handle))
end

function get_stream(d::Device)
    return Stream(ccall(@libCall(:occaDeviceGetStream),
                        Ptr{Void},
                        (Ptr{Void},),
                        d.handle))
end

function set_stream!(d::Device, s::Stream)
    return Stream(ccall(@libCall(:occaDeviceSetStream),
                        Void,
                        (Ptr{Void}, Ptr{Void},),
                        d.handle, s.handle))
end

function wrap_stream(d::Device, sHandle::Ptr{Void})
    return Stream(ccall(@libCall(:occaDeviceWrapStream),
                        Ptr{Void},
                        (Ptr{Void}, Ptr{Void},),
                        d.handle, sHandle))
end

function free!(s::Stream)
    if s.handle == C_NULL
        return
    end

    ccall(@libCall(:occaStreamFree),
          Void,
          (Ptr{Void},),
          s.handle)

    s.handle = C_NULL
end
#|=======================================

#|---[ Memory ]--------------------------
function free!(m::Memory)
    if m.handle == C_NULL
        return
    end

    ccall(@libCall(:occaMemoryFree),
          Void,
          (Ptr{Void},),
          m.handle)

    m.handle = C_NULL
end

function mode(m::Memory)
    strMode = ccall(@libCall(:occaMemoryMode),
                    Ptr{Uint8},
                    (Ptr{Void},),
                    m.handle)

    return bytestring(strMode)
end

function get_memory_handle(m::Memory)
    return ccall(@libCall(:occaMemoryGetMemoryHandle),
                 Ptr{Void},
                 (Ptr{Void},),
                 m.handle)
end

function get_mapped_pointer(m::Memory)
    return ccall(@libCall(:occaMemoryGetMappedPointer),
                 Ptr{Void},
                 (Ptr{Void},),
                 m.handle)
end

function get_texture_handle(m::Memory)
    return ccall(@libCall(:occaMemoryGetTextureHandle),
                 Ptr{Void},
                 (Ptr{Void},),
                 m.handle)
end

function memcpy!(dest::Memory, src::Memory, bytes; src_off = 0, dest_off = 0, async = false)
    dest_ = dest.handle
    src_  = src.handle

    if !async
        return ccall(@libCall(:occaCopyMemToMem),
                     Void,
                     (Ptr{Void}, Ptr{Void}, Uint64, Uint64, Uint64),
                     dest, src, bytes, src_off, dest_off)
    else
        return ccall(@libCall(:occaAsyncCopyMemToMem),
                     Void,
                     (Ptr{Void}, Ptr{Void}, Uint64, Uint64, Uint64),
                     dest, src, bytes, src_off, dest_off)
    end
end

function memcpy!(dest::Array, src::Memory, bytes; src_off = 0, dest_off = 0, async = false)
    dest_ = pointer(dest)
    src_  = src.handle

    if !async
        return ccall(@libCall(:occaCopyMemToPtr),
                     Void,
                     (Ptr{Void}, Ptr{Void}, Uint64, Uint64, Uint64),
                     dest, src, bytes, src_off, dest_off)
    else
        return ccall(@libCall(:occaAsyncCopyMemToPtr),
                     Void,
                     (Ptr{Void}, Ptr{Void}, Uint64, Uint64, Uint64),
                     dest, src, bytes, src_off, dest_off)
    end
end

function memcpy!(dest::Memory, src::Array, bytes; src_off = 0, dest_off = 0, async = false)
    dest_ = dest.handle
    src_  = pointer(src)

    if !async
        return ccall(@libCall(:occaCopyPtrToMem),
                     Void,
                     (Ptr{Void}, Ptr{Void}, Uint64, Uint64, Uint64),
                     dest, src, bytes, src_off, dest_off)
    else
        return ccall(@libCall(:occaAsyncCopyPtrToMem),
                     Void,
                     (Ptr{Void}, Ptr{Void}, Uint64, Uint64, Uint64),
                     dest, src, bytes, src_off, dest_off)
    end
end

function memcpy!(dest::Array, src::Array, bytes; src_off = 0, dest_off = 0, async = false)
    dest_ = pointer(dest.handle)
    src_  = pointer(src)

    if !async
        return ccall(@libCall(:occaCopyPtrToPtr),
                     Void,
                     (Ptr{Void}, Ptr{Void}, Uint64, Uint64, Uint64),
                     dest, src, bytes, src_off, dest_off)
    else
        return ccall(@libCall(:occaAsyncCopyPtrToPtr),
                     Void,
                     (Ptr{Void}, Ptr{Void}, Uint64, Uint64, Uint64),
                     dest, src, bytes, src_off, dest_off)
    end
end
#|=======================================

#|---[ Kernel ]--------------------------
function mode(k::Kernel)
    strMode = ccall(@libCall(:occaKernelMode),
                    Ptr{Uint8},
                    (Ptr{Void},),
                    k.handle)

    return bytestring(strMode)
end

function name(k::Kernel)
    strName = ccall(@libCall(:occaKernelName),
                    Ptr{Uint8},
                    (Ptr{Void},),
                    k.handle)

    return bytestring(strName)
end

function get_device(k::Kernel)
    return Device(ccall(@libCall(:occaKernelGetDevice),
                        Ptr{Void},
                        (Ptr{Void},),
                        k.handle))
end

function free!(k::Kernel)
    if k.handle == C_NULL
        return
    end

    ccall(@libCall(:occaKernelFree),
          Void,
          (Ptr{Void},),
          k.handle)

    k.handle = C_NULL
end

function call(k::Kernel, args...)
    argList = ccall(@libCall(:occaCreateArgumentList),
                    Ptr{Void}, ())

    argCount = length(args)

    for i in 1:argCount
        ccall(@libCall(:occaArgumentListAddArg),
              Void,
              (Ptr{Void}, Int32, Ptr{Void},),
              argList, convert(Int32, i - 1), occaType(args[i]))
    end

    ccall(@libCall(:occaKernelRun_),
          Void,
          (Ptr{Void}, Ptr{Void},),
          k.handle, argList)

    ccall(@libCall(:occaArgumentListFree),
          Void,
          (Ptr{Void},),
          argList)
end

function add_define!(kInfo::KernelInfo, macro_::String, value::String)
    occaValue = ccall(@libCall(:occaString),
                      Ptr{Void},
                      (Ptr{Uint8},),
                      bytestring(value))

    ccall(@libCall(:occaKernelInfoAddDefine),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Ptr{Void},),
          kInfo.handle, bytestring(macro_), occaValue)
end

function add_define!(kInfo::KernelInfo, macro_::String, value)
    add_define(kInfo, macro_, string(value))
end

function add_include!(kInfo::KernelInfo, include_::String)
    ccall(@libCall(:occaKernelInfoAddInclude),
          Void,
          (Ptr{Void}, Ptr{Uint8},),
          kInfo.handle, include_)
end

function free!(kInfo::KernelInfo)
    if kInfo.handle == C_NULL
        return
    end

    ccall(@libCall(:occaKernelInfoFree),
          Void,
          (Ptr{Void},),
          kInfo.handle)

    kInfo.handle = C_NULL
end
#|=======================================

end