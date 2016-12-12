/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include "occa/defines.hpp"

#if OCCA_CUDA_ENABLED

#include "occa/CUDA.hpp"

namespace occa {
  namespace cuda {
    device::device() {
      strMode = "CUDA";

      data = NULL;

      uvaEnabled_ = false;

      bytesAllocated = 0;

      getEnvironmentVariables();
    }

    device::device(const device &d) {
      *this = d;
    }

    device& device::operator = (const device &d) {
      modelID_ = d.modelID_;
      id_      = d.id_;

      data = d.data;

      uvaEnabled_    = d.uvaEnabled_;
      uvaMap         = d.uvaMap;
      uvaStaleMemory = d.uvaStaleMemory;

      compiler      = d.compiler;
      compilerFlags = d.compilerFlags;

      bytesAllocated = d.bytesAllocated;

      return *this;
    }

    device::~device() {}

    void* device::getContextHandle() {
      OCCA_EXTRACT_DATA(CUDA, Device);

      return (void*) data_.context;
    }

    void device::setup(argInfoMap &aim) {
      cuda::init();
      properties = aim;

      data = new CUDADeviceData_t;

      OCCA_EXTRACT_DATA(CUDA, Device);

      data_.p2pEnabled = false;

      OCCA_CHECK(aim.has("deviceID"),
                 "[CUDA] device not given [deviceID]");

      const int deviceID = aim.iGet("deviceID");

      OCCA_CUDA_CHECK("Device: Creating Device",
                      cuDeviceGet(&data_.device, deviceID));

      OCCA_CUDA_CHECK("Device: Creating Context",
                      cuCtxCreate(&data_.context, CU_CTX_SCHED_AUTO, data_.device));
    }

    void device::addOccaHeadersToInfo(kernelInfo &info_) {
      info_.mode = CUDA;
    }

    std::string device::getInfoSalt(const kernelInfo &info_) {
      std::stringstream salt;

      salt << "CUDA"
           << info_.salt()
           << parserVersion
           << compilerEnvScript
           << compiler
           << compilerFlags;

      return salt.str();
    }

    deviceIdentifier device::getIdentifier() const {
      deviceIdentifier dID;

      dID.mode_ = CUDA;

      const size_t archPos = compilerFlags.find("-arch=sm_");

      if (archPos == std::string::npos) {
        OCCA_EXTRACT_DATA(CUDA, Device);

        std::stringstream archSM_;

        int major, minor;
        OCCA_CUDA_CHECK("Getting CUDA Device Arch",
                        cuDeviceComputeCapability(&major, &minor, data_.device) );

        archSM_ << major << minor;

        dID.flagMap["sm_arch"] = archSM_.str();
      } else {
        const char *c0 = (compilerFlags.c_str() + archPos);
        const char *c1 = c0;

        while((*c1 != '\0') && (*c1 != ' '))
          ++c1;

        dID.flagMap["sm_arch"] = std::string(c0, c1 - c0);
      }

      return dID;
    }

    void device::getEnvironmentVariables() {
      char *c_compiler = getenv("OCCA_CUDA_COMPILER");

      if (c_compiler != NULL) {
        compiler = std::string(c_compiler);
      } else {
        compiler = "nvcc";
      }
      char *c_compilerFlags = getenv("OCCA_CUDA_COMPILER_FLAGS");

      if (c_compilerFlags != NULL) {
        compilerFlags = std::string(c_compilerFlags);
      } else {
#if OCCA_DEBUG_ENABLED
        compilerFlags = "-g";
#else
        compilerFlags = "";
#endif
      }

      properties["compiler"]      = compiler;
      properties["compilerFlags"] = compilerFlags;
    }

    void device::appendAvailableDevices(std::vector<device> &dList) {
      cuda::init();

      int deviceCount = cuda::getDeviceCount();

      for (int i = 0; i < deviceCount; ++i) {
        device d;
        d.setup("CUDA", i, 0);

        dList.push_back(d);
      }
    }

    void device::setCompiler(const std::string &compiler_) {
      compiler = compiler_;
    }

    void device::setCompilerEnvScript(const std::string &compilerEnvScript_) {
      compilerEnvScript = compilerEnvScript_;
    }

    void device::setCompilerFlags(const std::string &compilerFlags_) {
      compilerFlags = compilerFlags_;
    }

    void device::flush() {}

    void device::finish() {
      OCCA_CUDA_CHECK("Device: Finish",
                      cuStreamSynchronize(*((CUstream*) currentStream)) );
    }

    bool device::fakesUva() {
      return true;
    }

    void device::waitFor(streamTag tag) {
      OCCA_CUDA_CHECK("Device: Waiting For Tag",
                      cuEventSynchronize(cuda::event(tag)));
    }

    stream_t device::createStream() {
      OCCA_EXTRACT_DATA(CUDA, Device);

      CUstream *retStream = new CUstream;

      OCCA_CUDA_CHECK("Device: Setting Context",
                      cuCtxSetCurrent(data_.context));

      OCCA_CUDA_CHECK("Device: createStream",
                      cuStreamCreate(retStream, CU_STREAM_DEFAULT));

      return retStream;
    }

    void device::freeStream(stream_t s) {
      OCCA_CUDA_CHECK("Device: freeStream",
                      cuStreamDestroy( *((CUstream*) s) ));

      delete (CUstream*) s;
    }

    stream_t device::wrapStream(void *handle_) {
      return handle_;
    }

    streamTag device::tagStream() {
      OCCA_EXTRACT_DATA(CUDA, Device);

      streamTag ret;

      OCCA_CUDA_CHECK("Device: Setting Context",
                      cuCtxSetCurrent(data_.context));

      OCCA_CUDA_CHECK("Device: Tagging Stream (Creating Tag)",
                      cuEventCreate(&cuda::event(ret), CU_EVENT_DEFAULT));

      OCCA_CUDA_CHECK("Device: Tagging Stream",
                      cuEventRecord(cuda::event(ret), 0));

      return ret;
    }

    double device::timeBetween(const streamTag &startTag, const streamTag &endTag) {
      OCCA_CUDA_CHECK("Device: Waiting for endTag",
                      cuEventSynchronize(cuda::event(endTag)));

      float msTimeTaken;
      OCCA_CUDA_CHECK("Device: Timing Between Tags",
                      cuEventElapsedTime(&msTimeTaken, cuda::event(startTag), cuda::event(endTag)));

      return (double) (1.0e-3 * (double) msTimeTaken);
    }

    kernel_v* device::buildKernel(const std::string &filename,
                                  const std::string &functionName,
                                  const kernelInfo &info_) {
      OCCA_EXTRACT_DATA(CUDA, Device);

      OCCA_CUDA_CHECK("Device: Setting Context",
                      cuCtxSetCurrent(data_.context));

      kernel_v *k = new kernel_t<CUDA>;

      k->dHandle = this;
      k->data    = new CUDAKernelData_t;

      CUDAKernelData_t &kData_ = *((CUDAKernelData_t*) k->data);

      kData_.device  = data_.device;
      kData_.context = data_.context;

      k->build(filename, functionName, info_);

      return k;
    }

    kernel_v* device::buildKernelFromBinary(const std::string &filename,
                                            const std::string &functionName) {
      OCCA_EXTRACT_DATA(CUDA, Device);

      kernel_v *k = new kernel_t<CUDA>;

      k->dHandle = this;
      k->data    = new CUDAKernelData_t;

      CUDAKernelData_t &kData_ = *((CUDAKernelData_t*) k->data);

      kData_.device  = data_.device;
      kData_.context = data_.context;

      k->buildFromBinary(filename, functionName);
      return k;
    }

    void device::cacheKernelInLibrary(const std::string &filename,
                                      const std::string &functionName,
                                      const kernelInfo &info_) {
#if 0
      //---[ Creating shared library ]----
      kernel tmpK = occa::device(this).buildKernel(filename, functionName, info_);
      tmpK.free();

      kernelInfo info = info_;

      addOccaHeadersToInfo(info);

      std::string cachedBinary = getCachedName(filename, getInfoSalt(info));
      std::string contents     = io::read(cachedBinary);
      //==================================

      library::infoID_t infoID;

      infoID.modelID    = modelID_;
      infoID.kernelName = functionName;

      library::infoHeader_t &header = library::headerMap[infoID];

      header.fileID = -1;
      header.mode   = CUDA;

      const std::string flatDevID = getIdentifier().flattenFlagMap();

      header.flagsOffset = library::addToScratchPad(flatDevID);
      header.flagsBytes  = flatDevID.size();

      header.contentOffset = library::addToScratchPad(contents);
      header.contentBytes  = contents.size();

      header.kernelNameOffset = library::addToScratchPad(functionName);
      header.kernelNameBytes  = functionName.size();
#endif
    }

    kernel_v* device::loadKernelFromLibrary(const char *cache,
                                            const std::string &functionName) {
#if 0
      OCCA_EXTRACT_DATA(CUDA, Device);

      kernel_v *k = new kernel_t<CUDA>;

      k->dHandle = this;
      k->data    = new CUDAKernelData_t;

      CUDAKernelData_t &kData_ = *((CUDAKernelData_t*) k->data);

      kData_.device  = data_.device;
      kData_.context = data_.context;

      k->loadFromLibrary(cache, functionName);
      return k;
#endif
      return NULL;
    }

    memory_v* device::wrapMemory(void *handle_,
                                 const udim_t bytes) {
      memory_v *mem = new memory_t<CUDA>;

      mem->dHandle = this;
      mem->size    = bytes;
      mem->handle  = new CUdeviceptr;
      ::memcpy(mem->handle, &handle_, sizeof(CUdeviceptr));

      return mem;
    }

    memory_v* device::wrapTexture(void *handle_,
                                  const int dim, const occa::dim &dims,
                                  occa::formatType type, const int permissions) {
      memory_v *mem = new memory_t<CUDA>;

      mem->dHandle = this;
      mem->size    = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();
      mem->handle  = handle_;

      mem->memInfo |= memFlag::isATexture;

      mem->textureInfo.dim = dim;

      mem->textureInfo.w = dims.x;
      mem->textureInfo.h = dims.y;
      mem->textureInfo.d = dims.z;

      mem->textureInfo.bytesInEntry = type.bytes();

      return mem;
    }

    memory_v* device::malloc(const udim_t bytes,
                             void *src) {
      OCCA_EXTRACT_DATA(CUDA, Device);

      memory_v *mem = new memory_t<CUDA>;

      mem->dHandle = this;
      mem->handle  = new CUdeviceptr;
      mem->size    = bytes;

      OCCA_CUDA_CHECK("Device: Setting Context",
                      cuCtxSetCurrent(data_.context));

      OCCA_CUDA_CHECK("Device: malloc",
                      cuMemAlloc((CUdeviceptr*) mem->handle, bytes));

      if (src != NULL)
        mem->copyFrom(src, bytes, 0);

      return mem;
    }

    memory_v* device::textureAlloc(const int dim, const occa::dim &dims,
                                   void *src,
                                   occa::formatType type, const int permissions) {
      OCCA_EXTRACT_DATA(CUDA, Device);

      memory_v *mem = new memory_t<CUDA>;

      mem->dHandle = this;
      mem->handle  = new CUDATextureData_t;
      mem->size    = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

      mem->memInfo |= memFlag::isATexture;

      mem->textureInfo.dim = dim;

      mem->textureInfo.w = dims.x;
      mem->textureInfo.h = dims.y;
      mem->textureInfo.d = dims.z;

      mem->textureInfo.bytesInEntry = type.bytes();

      CUarray &array        = ((CUDATextureData_t*) mem->handle)->array;
      CUsurfObject &surface = ((CUDATextureData_t*) mem->handle)->surface;

      CUDA_ARRAY_DESCRIPTOR arrayDesc;
      CUDA_RESOURCE_DESC surfDesc;

      memset(&arrayDesc, 0, sizeof(arrayDesc));
      memset(&surfDesc , 0, sizeof(surfDesc));

      arrayDesc.Width       = dims.x;
      arrayDesc.Height      = (dim == 1) ? 0 : dims.y;
      arrayDesc.Format      = *((CUarray_format*) type.format<CUDA>());
      arrayDesc.NumChannels = type.count();

      OCCA_CUDA_CHECK("Device: Setting Context",
                      cuCtxSetCurrent(data_.context));

      OCCA_CUDA_CHECK("Device: Creating Array",
                      cuArrayCreate(&array, (CUDA_ARRAY_DESCRIPTOR*) &arrayDesc) );

      surfDesc.res.array.hArray = array;
      surfDesc.resType = CU_RESOURCE_TYPE_ARRAY;

      OCCA_CUDA_CHECK("Device: Creating Surface Object",
                      cuSurfObjectCreate(&surface, &surfDesc) );

      mem->textureInfo.arg = new int;
      *((int*) mem->textureInfo.arg) = CUDA_ADDRESS_CLAMP;

      mem->copyFrom(src);

      /*
        if (dims == 3) {
        CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
        memset(&arrayDesc, 0, sizeof(arrayDesc);

        arrayDesc.Width  = size.x;
        arrayDesc.Height = size.y;
        arrayDesc.Depth  = size.z;

        arrayDesc.Format      = type.format<CUDA>();
        arrayDesc.NumChannels = type.count();

        cuArray3DCreate(&arr, (CUDA_ARRAY3D_DESCRIPTOR*) &arrayDesc);
        }
      */

      return mem;
    }

    memory_v* device::mappedAlloc(const udim_t bytes,
                                  void *src) {
      OCCA_EXTRACT_DATA(CUDA, Device);

      memory_v *mem = new memory_t<CUDA>;

      mem->dHandle  = this;
      mem->handle   = new CUdeviceptr;
      mem->size     = bytes;

      mem->memInfo |= memFlag::isMapped;

      OCCA_CUDA_CHECK("Device: Setting Context",
                      cuCtxSetCurrent(data_.context));

      OCCA_CUDA_CHECK("Device: malloc host",
                      cuMemAllocHost((void**) &(mem->mappedPtr), bytes));

      OCCA_CUDA_CHECK("Device: get device pointer from host",
                      cuMemHostGetDevicePointer((CUdeviceptr*) mem->handle,
                                                mem->mappedPtr,
                                                0));

      if (src != NULL)
        ::memcpy(mem->mappedPtr, src, bytes);

      return mem;
    }

    udim_t device::memorySize() {
      OCCA_EXTRACT_DATA(CUDA, Device);

      return cuda::getDeviceMemorySize(data_.device);
    }

    void device::free() {
      OCCA_EXTRACT_DATA(CUDA, Device);

      OCCA_CUDA_CHECK("Device: Freeing Context",
                      cuCtxDestroy(data_.context) );

      delete (CUDADeviceData_t*) data;
    }

    int device::simdWidth() {
      if (simdWidth_)
        return simdWidth_;

      OCCA_EXTRACT_DATA(CUDA, Device);

      OCCA_CUDA_CHECK("Device: Get Warp Size",
                      cuDeviceGetAttribute(&simdWidth_,
                                           CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                           data_.device) );

      return simdWidth_;
    }
  }
}

#endif
