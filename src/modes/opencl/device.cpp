/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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

#include <occa/defines.hpp>

#if OCCA_OPENCL_ENABLED

#include <occa/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/opencl/device.hpp>
#include <occa/modes/opencl/kernel.hpp>
#include <occa/modes/opencl/memory.hpp>
#include <occa/modes/opencl/utils.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/modes/opencl.hpp>

namespace occa {
  namespace opencl {
    device::device(const occa::properties &properties_) :
      occa::device_v(properties_) {

      if (!properties.has("wrapped")) {
        cl_int error;
        OCCA_ERROR("[OpenCL] device not given a [platformID] integer",
                   properties.has("platformID") &&
                   properties["platformID"].isNumber());


        OCCA_ERROR("[OpenCL] device not given a [deviceID] integer",
                   properties.has("deviceID") &&
                   properties["deviceID"].isNumber());

        platformID = properties.get<int>("platformID");
        deviceID   = properties.get<int>("deviceID");

        clDevice = opencl::deviceID(platformID, deviceID);

        clContext = clCreateContext(NULL, 1, &clDevice, NULL, NULL, &error);
        OCCA_OPENCL_ERROR("Device: Creating Context", error);
      }

      std::string compilerFlags;

      if (properties.has("kernel/compilerFlags")) {
        compilerFlags = (std::string) properties["kernel/compilerFlags"];
      } else if (env::var("OCCA_OPENCL_COMPILER_FLAGS").size()) {
        compilerFlags = env::var("OCCA_OPENCL_COMPILER_FLAGS");
      } else {
        compilerFlags = "-cl-opt-disable";
      }

      properties["kernel/compilerFlags"] = compilerFlags;
    }

    device::~device() {}

    void device::free() {
      if (clContext) {
        OCCA_OPENCL_ERROR("Device: Freeing Context",
                          clReleaseContext(clContext) );
        clContext = NULL;
      }
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
        std::stringstream ss;
        ss << "platform: " << platformID << ' '
           << "device: " << deviceID;
        hash_ = occa::hash(ss.str());
      }
      return hash_;
    }

    //---[ Stream ]---------------------
    stream_t device::createStream() const {
      cl_int error;

      cl_command_queue *retStream = new cl_command_queue;

      *retStream = clCreateCommandQueue(clContext,
                                        clDevice,
                                        CL_QUEUE_PROFILING_ENABLE,
                                        &error);
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
    //==================================

    //---[ Kernel ]---------------------
    bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const std::string &hostOutputFile,
                           const occa::properties &kernelProps,
                           lang::kernelMetadataMap &hostMetadata,
                           lang::kernelMetadataMap &deviceMetadata) {
      lang::okl::openclParser parser(kernelProps);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      if (!parser.succeeded()) {
        if (!kernelProps.get("silent", false)) {
          OCCA_FORCE_ERROR("Unable to transform OKL kernel");
        }
        return false;
      }

      if (!sys::fileExists(outputFile)) {
        hash_t hash = occa::hash(outputFile);
        io::lock_t lock(hash, "opencl-parser-device");
        if (lock.isMine()) {
          parser.writeToFile(outputFile);
        }
      }

      if (!sys::fileExists(hostOutputFile)) {
        hash_t hash = occa::hash(hostOutputFile);
        io::lock_t lock(hash, "opencl-parser-host");
        if (lock.isMine()) {
          parser.hostParser.writeToFile(hostOutputFile);
        }
      }

      parser.hostParser.setMetadata(hostMetadata);
      parser.setMetadata(deviceMetadata);

      return true;
    }

    kernel_v* device::buildKernel(const std::string &filename,
                                  const std::string &kernelName,
                                  const hash_t kernelHash,
                                  const occa::properties &kernelProps) {

      occa::properties allProps = properties["kernel"] + kernelProps;

      const std::string hashDir = io::hashDir(filename, kernelHash);
      const std::string binaryFilename = hashDir + kc::binaryFile;
      bool foundBinary = true;
      bool usingOKL = allProps.get("okl", true);

      info_t clInfo;
      clInfo.clDevice  = clDevice;
      clInfo.clContext = clContext;

      io::lock_t lock(kernelHash, "opencl-kernel");
      if (lock.isMine()) {
        if (sys::fileExists(binaryFilename)) {
          lock.release();
        } else {
          foundBinary = false;
        }
      }

      const bool verbose = allProps.get("verbose", false);
      if (foundBinary) {
        if (verbose) {
           std::cout << "Loading cached ["
                     << kernelName
                     << "] from ["
                     << io::shortname(filename)
                     << "] in [" << io::shortname(binaryFilename) << "]\n";
        }
        if (usingOKL) {
          lang::kernelMetadataMap hostMetadata = (
            lang::getBuildFileMetadata(hashDir + kc::hostBuildFile)
          );
          lang::kernelMetadataMap deviceMetadata = (
            lang::getBuildFileMetadata(hashDir + kc::buildFile)
          );
          return buildOKLKernelFromBinary(clInfo,
                                          hashDir,
                                          kernelName,
                                          hostMetadata,
                                          deviceMetadata,
                                          allProps,
                                          lock);
        } else {
          return buildKernelFromBinary(binaryFilename,
                                       kernelName,
                                       kernelProps);
        }
      }

      // Cache raw origin
      std::string sourceFilename = (
        io::cacheFile(filename,
                      kc::rawSourceFile,
                      kernelHash,
                      assembleHeader(allProps))
      );

      kernel_v *launcherKernel = NULL;
      lang::kernelMetadataMap hostMetadata, deviceMetadata;
      if (usingOKL) {
        const std::string outputFile = hashDir + kc::sourceFile;
        const std::string hostOutputFile = hashDir + kc::hostSourceFile;
        bool valid = parseFile(sourceFilename,
                               outputFile,
                               hostOutputFile,
                               allProps,
                               hostMetadata,
                               deviceMetadata);
        if (!valid) {
          return NULL;
        }
        sourceFilename = outputFile;

        launcherKernel = buildLauncherKernel(hashDir,
                                             kernelName,
                                             hostMetadata[kernelName]);

        // No OKL means no build file is generated,
        //   so we need to build it
        host()
          .getDHandle()
          ->writeKernelBuildFile(hashDir + kc::hostBuildFile,
                                 kernelHash,
                                 occa::properties(),
                                 hostMetadata);

        writeKernelBuildFile(hashDir + kc::buildFile,
                             kernelHash,
                             kernelProps,
                             deviceMetadata);
      }

      // Build OpenCL program
      std::string source = io::read(sourceFilename);

      opencl::buildProgramFromSource(clInfo,
                                     source,
                                     kernelName,
                                     allProps["compilerFlags"],
                                     sourceFilename,
                                     allProps,
                                     lock);

      opencl::saveProgramBinary(clInfo.clProgram,
                                binaryFilename,
                                lock);

      // Regular OpenCL Kernel
      if (!launcherKernel) {
        opencl::buildKernelFromProgram(clInfo,
                                       kernelName,
                                       lock);
        return new kernel(this,
                          kernelName,
                          sourceFilename,
                          clDevice,
                          clInfo.clKernel,
                          kernelProps);
      }

      return buildOKLKernelFromBinary(clInfo,
                                      hashDir,
                                      kernelName,
                                      hostMetadata,
                                      deviceMetadata,
                                      kernelProps,
                                      lock);
    }

    kernel_v* device::buildOKLKernelFromBinary(info_t &clInfo,
                                               const std::string &hashDir,
                                               const std::string &kernelName,
                                               lang::kernelMetadataMap &hostMetadata,
                                               lang::kernelMetadataMap &deviceMetadata,
                                               const occa::properties &kernelProps,
                                               io::lock_t lock) {

      const std::string sourceFilename = hashDir + kc::sourceFile;
      const std::string binaryFilename = hashDir + kc::binaryFile;

      if (!clInfo.clProgram) {
        opencl::buildProgramFromBinary(clInfo,
                                       io::read(binaryFilename),
                                       kernelName,
                                       properties["compilerFlags"],
                                       lock);
      }

      // Create wrapper kernel and set launcherKernel
      kernel &k = *(new kernel(this,
                               kernelName,
                               sourceFilename,
                               kernelProps));

      k.launcherKernel = buildLauncherKernel(hashDir,
                                             kernelName,
                                             hostMetadata[kernelName]);
      if (!k.launcherKernel) {
        delete &k;
        return NULL;
      }

      // Find clKernels
      typedef std::map<int, lang::kernelMetadata> kernelOrderMap;
      kernelOrderMap clKernelMetadata;

      const std::string prefix = "_occa_" + kernelName + "_";

      lang::kernelMetadataMap::iterator it = deviceMetadata.begin();
      while (it != deviceMetadata.end()) {
        const std::string &name = it->first;
        lang::kernelMetadata &metadata = it->second;
        ++it;
        if (!startsWith(name, prefix)) {
          continue;
        }
        std::string suffix = name.substr(prefix.size());
        const char *c = suffix.c_str();
        primitive number = primitive::load(c, false);
        // Make sure we reached the end ['\0']
        //   and have a number
        if (*c || number.isNaN()) {
          continue;
        }
        clKernelMetadata[number] = metadata;
      }

      kernelOrderMap::iterator oit = clKernelMetadata.begin();
      while (oit != clKernelMetadata.end()) {
        lang::kernelMetadata &metadata = oit->second;
        opencl::buildKernelFromProgram(clInfo,
                                       metadata.name,
                                       lock);

        kernel *clKernel = new kernel(this,
                                      metadata.name,
                                      sourceFilename,
                                      clDevice,
                                      clInfo.clKernel,
                                      kernelProps);
        clKernel->dontUseRefs();
        k.clKernels.push_back(clKernel);

        ++oit;
      }

      return &k;
    }

    kernel_v* device::buildLauncherKernel(const std::string &hashDir,
                                          const std::string &kernelName,
                                          lang::kernelMetadata &hostMetadata) {
      const std::string hostOutputFile = hashDir + kc::hostSourceFile;

      occa::kernel hostKernel = host().buildKernel(hostOutputFile,
                                                   kernelName,
                                                   "okl: false");

      // Launcher and clKernels use the same refs as the wrapper kernel
      kernel_v *launcherKernel = hostKernel.getKHandle();
      if (!launcherKernel) {
        return NULL;
      }

      launcherKernel->dontUseRefs();

      launcherKernel->metadata = hostMetadata;

      return launcherKernel;
    }

    kernel_v* device::buildKernelFromBinary(const std::string &filename,
                                            const std::string &kernelName,
                                            const occa::properties &kernelProps) {

      std::string source = io::read(filename);

      info_t clInfo;
      clInfo.clDevice  = clDevice;
      clInfo.clContext = clContext;

      opencl::buildProgramFromBinary(clInfo,
                                     source,
                                     kernelName,
                                     properties["compilerFlags"]);

      opencl::buildKernelFromProgram(clInfo,
                                     kernelName);

      return new kernel(this,
                        filename,
                        kernelName,
                        clDevice,
                        clInfo.clKernel,
                        kernelProps);
    }
    //==================================

    //---[ Memory ]---------------------
    memory_v* device::malloc(const udim_t bytes,
                             const void *src,
                             const occa::properties &props) {

      if (props.get("opencl/mapped", false)) {
        return mappedAlloc(bytes, src, props);
      }

      cl_int error;

      opencl::memory *mem = new opencl::memory(props);
      mem->dHandle = this;
      mem->size    = bytes;

      if (src == NULL) {
        mem->clMem = clCreateBuffer(clContext,
                                    CL_MEM_READ_WRITE,
                                    bytes, NULL, &error);
        OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);
      } else {
        mem->clMem = clCreateBuffer(clContext,
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
      mem->dHandle = this;
      mem->size    = bytes;

      // Alloc pinned host buffer
      mem->clMem = clCreateBuffer(clContext,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  bytes,
                                  NULL, &error);

      OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);

      if (src != NULL){
        mem->copyFrom(src, mem->size);
      }

      // Map memory to read/write
      mem->mappedPtr = clEnqueueMapBuffer(stream,
                                          mem->clMem,
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

    udim_t device::memorySize() const {
      return opencl::getDeviceMemorySize(clDevice);
    }
    //==================================
  }
}

#endif
