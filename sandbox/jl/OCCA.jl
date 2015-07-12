module OCCA

macro libocca()
    return bytestring(ENV["OCCA_DIR"], "/lib/libocca.so")
end

macro libCall(name)
    return :($name, @libocca())
end

#|---[ Setup ]---------------------------

#|=======================================

#|---[ Globals & Flags ]-----------------
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

function malloc(d::Device, t::Type, entries; source = C_NULL, mapped = false)
    bytes = entries * sizeof(t)
    convert(Uint64, bytes)

    if !mapped
        return Memory(ccall(@libCall(:occaDeviceMalloc),
                            Ptr{Void},
                            (Ptr{Void}, Uint64, Ptr{Void},),
                            d.handle, bytes, source))
    else
        return Memory(ccall(@libCall(:occaDeviceMappedMalloc),
                            Ptr{Void},
                            (Ptr{Void}, Uint64, Ptr{Void},),
                            d.handle, bytes, source))
    end
end

function malloc(d::Device, arr::Array; mapped = false)
    entries = length(arr)

    if entries == 0
        return Memory()
    end

    return malloc(d, typeof(arr[0]), entries, source = pointer(arr), mapped = mapped)
end

function managedAlloc(d::Device, t::Type, entries; mapped = false)
    bytes = entries * sizeof(t)
    convert(Uint64, bytes)

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

    convert(Ptr{t}, ptr)

    return pointer_to_array(ptr, entries)
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

function memcpy(dest, src, bytes; src_off = 0, dest_off = 0)
    if (typeof(dest) == Memory)
        if (typeof(src) == Memory)
            memcpyCall = :occaCopyMemToMem
        else
            memcpyCall = :occaCopyPtrToMem
        end
    else
        if (typeof(src) == Memory)
            memcpyCall = :occaCopyMemToPtr
        else
            memcpyCall = :occaCopyPtrToPtr
        end
    end

    return ccall(@libCall(memcpyCall),
                 Void,
                 (Ptr{Void}, Ptr{Void}, Uint64, Uint64, Uint64),
                 dest.handle, src.handle, bytes, src_off, dest_off)
end

function async_memcpy(dest, src, bytes; src_off = 0, dest_off = 0)
    if (typeof(dest) == Memory)
        if (typeof(src) == Memory)
            memcpyCall = :occaAsyncCopyMemToMem
        else
            memcpyCall = :occaAsyncCopyPtrToMem
        end
    else
        if (typeof(src) == Memory)
            memcpyCall = :occaAsyncCopyMemToPtr
        else
            memcpyCall = :occaAsyncCopyPtrToPtr
        end
    end

    return ccall(@libCall(memcpyCall),
                 Void,
                 (Ptr{Void}, Ptr{Void}, Uint64, Uint64, Uint64),
                 dest.handle, src.handle, bytes, src_off, dest_off)
end
#|=======================================

end