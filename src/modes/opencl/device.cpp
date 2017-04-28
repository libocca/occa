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

#if OCCA_OPENCL_ENABLED

#include "occa/modes/opencl/device.hpp"
#include "occa/modes/opencl/kernel.hpp"
#include "occa/modes/opencl/memory.hpp"
#include "occa/modes/opencl/utils.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/sys.hpp"
#include "occa/base.hpp"

namespace occa {
  namespace opencl {
    device::device(const occa::properties &properties_) :
      occa::device_v(properties_) {

      cl_int error;
      OCCA_ERROR("[OpenCL] device not given a [platformID] integer",
                 properties.has("platformID") &&
                 properties["platformID"].isNumber());


      OCCA_ERROR("[OpenCL] device not given a [deviceID] integer",
                 properties.has("deviceID") &&
                 properties["deviceID"].isNumber());

      platformID = properties.get<int>("platformID");
      deviceID   = properties.get<int>("deviceID");

      clPlatformID = opencl::platformID(platformID);
      clDeviceID   = opencl::deviceID(platformID, deviceID);

      clContext = clCreateContext(NULL, 1, &clDeviceID, NULL, NULL, &error);
      OCCA_OPENCL_ERROR("Device: Creating Context", error);

      std::string compilerFlags;

      if (properties.has("compilerFlags")) {
        compilerFlags = properties["compilerFlags"].string();
      } else if (env::var("OCCA_OPENCL_COMPILER_FLAGS").size()) {
        compilerFlags = env::var("OCCA_OPENCL_COMPILER_FLAGS");
      } else {
#if OCCA_DEBUG_ENABLED
        compilerFlags = "-cl-opt-disable";
#else
        compilerFlags = "";
#endif
      }

      properties["compilerFlags"] = compilerFlags;
    }

    device::~device() {}

    void device::free() {
      if (clContext) {
        OCCA_OPENCL_ERROR("Device: Freeing Context",
                          clReleaseContext(clContext) );
        clContext = NULL;
      }
    }

    void* device::getHandle(const occa::properties &props) const {
      if (props["type"] == "context") {
        return (void*) clContext;
      }
      return NULL;
    }

    void device::finish() const {
      OCCA_OPENCL_ERROR("Device: Finish",
                        clFinish(*((cl_command_queue*) currentStream)));
    }

    bool device::hasSeparateMemorySpace() const {
      return true;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        hash_ = occa::hash(properties);
      }
      return hash_;
    }

    //  |---[ Stream ]----------------
    stream_t device::createStream() const {
      cl_int error;

      cl_command_queue *retStream = new cl_command_queue;

      *retStream = clCreateCommandQueue(clContext, clDeviceID, CL_QUEUE_PROFILING_ENABLE, &error);
      OCCA_OPENCL_ERROR("Device: createStream", error);

      return retStream;
    }

    void device::freeStream(stream_t s) const {
      OCCA_OPENCL_ERROR("Device: freeStream",
                        clReleaseCommandQueue( *((cl_command_queue*) s) ));

      delete (cl_command_queue*) s;
    }

    streamTag device::tagStream() const {
      cl_command_queue &stream = *((cl_command_queue*) currentStream);

      streamTag ret;

#ifdef CL_VERSION_1_2
      OCCA_OPENCL_ERROR("Device: Tagging Stream",
                        clEnqueueMarkerWithWaitList(stream, 0, NULL, &event(ret)));
#else
      OCCA_OPENCL_ERROR("Device: Tagging Stream",
                        clEnqueueMarker(stream, &event(ret)));
#endif

      return ret;
    }

    void device::waitFor(streamTag tag) const {
      OCCA_OPENCL_ERROR("Device: Waiting For Tag",
                        clWaitForEvents(1, &event(tag)));
    }

    double device::timeBetween(const streamTag &startTag, const streamTag &endTag) const {
      cl_ulong start, end;

      finish();

      OCCA_OPENCL_ERROR ("Device: Time Between Tags (Start)",
                         clGetEventProfilingInfo(event(startTag),
                                                 CL_PROFILING_COMMAND_END,
                                                 sizeof(cl_ulong),
                                                 &start, NULL) );

      OCCA_OPENCL_ERROR ("Device: Time Between Tags (End)",
                         clGetEventProfilingInfo(event(endTag),
                                                 CL_PROFILING_COMMAND_START,
                                                 sizeof(cl_ulong),
                                                 &end, NULL) );

      OCCA_OPENCL_ERROR("Device: Time Between Tags (Freeing start tag)",
                        clReleaseEvent(event(startTag)));

      OCCA_OPENCL_ERROR("Device: Time Between Tags (Freeing end tag)",
                        clReleaseEvent(event(endTag)));

      return (double) (1.0e-9 * (double)(end - start));
    }

    stream_t device::wrapStream(void *handle_, const occa::properties &props) const {
      return handle_;
    }
    //  |===============================

    //  |---[ Kernel ]------------------
    kernel_v* device::buildKernel(const std::string &filename,
                                  const std::string &kernelName,
                                  const occa::properties &props) {
      opencl::kernel *k = new opencl::kernel(props);

      k->dHandle = this;

      k->platformID = platformID;
      k->deviceID   = deviceID;

      k->clPlatformID = clPlatformID;
      k->clDeviceID   = clDeviceID;
      k->clContext    = clContext;

      k->build(filename, kernelName, props);

      return k;
    }

    kernel_v* device::buildKernelFromBinary(const std::string &filename,
                                            const std::string &kernelName,
                                            const occa::properties &props) {
      opencl::kernel *k = new opencl::kernel(props);

      k->dHandle = this;

      k->platformID = platformID;
      k->deviceID   = deviceID;

      k->clPlatformID = clPlatformID;
      k->clDeviceID   = clDeviceID;
      k->clContext    = clContext;

      k->buildFromBinary(filename, kernelName, props);

      return k;
    }
    //  |===============================

    //  |---[ Memory ]------------------
    memory_v* device::malloc(const udim_t bytes,
                             const void *src,
                             const occa::properties &props) {

      if (props.get<bool>("mapped")) {
        return mappedAlloc(bytes, src, props);
      }

      cl_int error;

      opencl::memory *mem = new opencl::memory(props);
      mem->dHandle = this;
      mem->handle  = new cl_mem;
      mem->size    = bytes;

      if (src == NULL) {
        *((cl_mem*) mem->handle) = clCreateBuffer(clContext,
                                                  CL_MEM_READ_WRITE,
                                                  bytes, NULL, &error);
        OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);
      } else {
        *((cl_mem*) mem->handle) = clCreateBuffer(clContext,
                                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                  bytes, const_cast<void*>(src), &error);
        OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);

        finish();
      }

      return mem;
    }

    memory_v* device::mappedAlloc(const udim_t bytes,
                                  const void *src,
                                  const occa::properties &props) {

      cl_int error;

      cl_command_queue &stream = *((cl_command_queue*) currentStream);
      opencl::memory *mem = new opencl::memory(props);
      mem->dHandle  = this;
      mem->handle   = new cl_mem;
      mem->size     = bytes;

      // Alloc pinned host buffer
      *((cl_mem*) mem->handle) = clCreateBuffer(clContext,
                                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                                bytes,
                                                NULL, &error);

      OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);

      if (src != NULL){
        mem->copyFrom(src, mem->size);
      }

      // Map memory to read/write
      mem->mappedPtr = clEnqueueMapBuffer(stream,
                                          *((cl_mem*) mem->handle),
                                          CL_TRUE,
                                          CL_MAP_READ | CL_MAP_WRITE,
                                          0, bytes,
                                          0, NULL, NULL,
                                          &error);

      OCCA_OPENCL_ERROR("Device: clEnqueueMapBuffer", error);

      // Sync memory mapping
      finish();

      return mem;
    }

    memory_v* device::wrapMemory(void *handle_,
                                 const udim_t bytes,
                                 const occa::properties &props) {
      opencl::memory *mem = new opencl::memory(props);
      mem->dHandle  = this;
      mem->handle   = new cl_mem;
      mem->size     = bytes;
      ::memcpy(mem->handle, handle_, sizeof(cl_mem));
      return mem;
    }

    udim_t device::memorySize() const {
      return opencl::getDeviceMemorySize(clDeviceID);
    }
    //  |===============================
  }
}

#endif
