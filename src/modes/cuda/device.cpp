/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include "occa/modes/cuda/device.hpp"
#include "occa/modes/cuda/kernel.hpp"
#include "occa/modes/cuda/memory.hpp"
#include "occa/modes/cuda/utils.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/sys.hpp"
#include "occa/base.hpp"

namespace occa {
  namespace cuda {
    device::device(const occa::properties &properties_) :
      occa::device_v(properties_) {

      OCCA_ERROR("[CUDA] device not given [deviceID]",
                 properties.has("deviceID"));

      const int deviceID = properties.get<int>("deviceID");

      OCCA_CUDA_ERROR("Device: Creating Device",
                      cuDeviceGet(&handle, deviceID));

      OCCA_CUDA_ERROR("Device: Creating Context",
                      cuCtxCreate(&context, CU_CTX_SCHED_AUTO, handle));

      p2pEnabled = false;

      std::string compiler = properties["compiler"].getString();
      std::string compilerFlags = properties["compilerFlags"].getString();

      if (!compiler.size()) {
        if (env::var("OCCA_CUDA_COMPILER").size()) {
          compiler = env::var("OCCA_CUDA_COMPILER");
        } else {
          compiler = "nvcc";
        }
      }

      if (!compilerFlags.size()) {
        if (env::var("OCCA_CUDA_COMPILER_FLAGS").size()) {
          compilerFlags = env::var("OCCA_CUDA_COMPILER_FLAGS");
        } else {
#if OCCA_DEBUG_ENABLED
          compilerFlags = "-g";
#else
          compilerFlags = "";
#endif
        }
      }

      properties["compiler"]      = compiler;
      properties["compilerFlags"] = compilerFlags;

      OCCA_CUDA_ERROR("Device: Getting CUDA Device Arch",
                      cuDeviceComputeCapability(&archMajorVersion,
                                                &archMinorVersion,
                                                handle) );
    }

    device::~device() {}

    void device::free() {
      OCCA_CUDA_ERROR("Device: Freeing Context",
                      cuCtxDestroy(context) );
    }

    void* device::getHandle(const occa::properties &props) {
      if (props.get<std::string>("type", "") == "context") {
        return (void*) context;
      }
      return (void*) (uintptr_t) handle;
    }

    void device::finish() {
      OCCA_CUDA_ERROR("Device: Finish",
                      cuStreamSynchronize(*((CUstream*) currentStream)) );
    }

    bool device::hasSeparateMemorySpace() {
      return true;
    }

    hash_t device::hash() {
      if (!hash_.initialized) {
        hash_ ^= occa::hash(properties);
        hash_ ^= occa::hash(archMajorVersion);
        hash_ ^= occa::hash(archMinorVersion);
      }
      return hash_;
    }

    //  |---[ Stream ]----------------
    stream_t device::createStream() {
      CUstream *retStream = new CUstream;

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(context));
      OCCA_CUDA_ERROR("Device: createStream",
                      cuStreamCreate(retStream, CU_STREAM_DEFAULT));

      return retStream;
    }

    void device::freeStream(stream_t s) {
      OCCA_CUDA_ERROR("Device: freeStream",
                      cuStreamDestroy( *((CUstream*) s) ));
      delete (CUstream*) s;
    }

    streamTag device::tagStream() {
      streamTag ret;

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(context));
      OCCA_CUDA_ERROR("Device: Tagging Stream (Creating Tag)",
                      cuEventCreate(&cuda::event(ret), CU_EVENT_DEFAULT));
      OCCA_CUDA_ERROR("Device: Tagging Stream",
                      cuEventRecord(cuda::event(ret), 0));

      return ret;
    }

    void device::waitFor(streamTag tag) {
      OCCA_CUDA_ERROR("Device: Waiting For Tag",
                      cuEventSynchronize(cuda::event(tag)));
    }

    double device::timeBetween(const streamTag &startTag, const streamTag &endTag) {
      OCCA_CUDA_ERROR("Device: Waiting for endTag",
                      cuEventSynchronize(cuda::event(endTag)));

      float msTimeTaken;
      OCCA_CUDA_ERROR("Device: Timing Between Tags",
                      cuEventElapsedTime(&msTimeTaken, cuda::event(startTag), cuda::event(endTag)));

      return (double) (1.0e-3 * (double) msTimeTaken);
    }

    stream_t device::wrapStream(void *handle_) {
      return handle_;
    }
    //  |===============================

    //  |---[ Kernel ]------------------
    kernel_v* device::buildKernel(const std::string &filename,
                                  const std::string &kernelName,
                                  const occa::properties &props) {
      cuda::kernel *k = new cuda::kernel(props);

      k->dHandle = this;
      k->context = context;

      k->build(filename, kernelName, props);

      return k;
    }

    kernel_v* device::buildKernelFromBinary(const std::string &filename,
                                            const std::string &kernelName,
                                            const occa::properties &props) {
      cuda::kernel *k = new cuda::kernel();

      k->dHandle = this;
      k->context = context;

      k->buildFromBinary(filename, kernelName, props);

      return k;
    }
    //  |===============================

    //  |---[ Memory ]------------------
    memory_v* device::malloc(const udim_t bytes,
                             void *src,
                             const occa::properties &props) {

      if (props.get<bool>("mapped")) {
        return mappedAlloc(bytes, src);
      }

      cuda::memory *mem = new cuda::memory();
      mem->dHandle = this;
      mem->handle  = new CUdeviceptr;
      mem->size    = bytes;

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(context));

      OCCA_CUDA_ERROR("Device: malloc",
                      cuMemAlloc((CUdeviceptr*) mem->handle, bytes));

      if (src != NULL) {
        mem->copyFrom(src, bytes, 0);
      }
      return mem;
    }

    memory_v* device::mappedAlloc(const udim_t bytes,
                                  void *src) {

      cuda::memory *mem = new cuda::memory();
      mem->dHandle = this;
      mem->handle  = new CUdeviceptr;
      mem->size    = bytes;

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(context));
      OCCA_CUDA_ERROR("Device: malloc host",
                      cuMemAllocHost((void**) &(mem->mappedPtr), bytes));
      OCCA_CUDA_ERROR("Device: get device pointer from host",
                      cuMemHostGetDevicePointer((CUdeviceptr*) mem->handle,
                                                mem->mappedPtr,
                                                0));

      if (src != NULL) {
        ::memcpy(mem->mappedPtr, src, bytes);
      }
      return mem;
    }

    memory_v* device::wrapMemory(void *handle_,
                                 const udim_t bytes,
                                 const occa::properties &props) {
      cuda::memory *mem = new cuda::memory();
      mem->dHandle = this;
      mem->handle  = new CUdeviceptr;
      mem->size    = bytes;
      ::memcpy(mem->handle, &handle_, sizeof(CUdeviceptr));
      return mem;
    }

    udim_t device::memorySize() {
      return cuda::getDeviceMemorySize(handle);
    }
    //  |===============================
  }
}

#endif
