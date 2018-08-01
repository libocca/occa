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

#if OCCA_HIP_ENABLED

#include <occa/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/misc.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/hip/device.hpp>
#include <occa/modes/hip/kernel.hpp>
#include <occa/modes/hip/memory.hpp>
#include <occa/modes/hip/utils.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/modes/hip.hpp>

namespace occa {
  namespace hip {
    device::device(const occa::properties &properties_) :
      occa::device_v(properties_) {

      if (!properties.has("wrapped")) {
        OCCA_ERROR("[HIP] device not given a [device_id] integer",
                   properties.has("device_id") &&
                   properties["device_id"].isNumber());

        const int deviceID = properties.get<int>("device_id");

        OCCA_HIP_ERROR("Device: Creating Device",
                       hipDeviceGet(&hipDevice, deviceID));

        OCCA_HIP_ERROR("Device: Creating Context",
                       hipCtxCreate(&hipContext, 0, hipDevice));
      }

      p2pEnabled = false;

      std::string compiler = properties["kernel/compiler"];
      std::string compilerFlags = properties["kernel/compilerFlags"];

      if (!compiler.size()) {
        if (env::var("OCCA_HIP_COMPILER").size()) {
          compiler = env::var("OCCA_HIP_COMPILER");
        } else {
          compiler = "hipcc";
        }
      }

      if (!compilerFlags.size()) {
        compilerFlags = env::var("OCCA_HIP_COMPILER_FLAGS");
      }

      properties["kernel/compiler"]      = compiler;
      properties["kernel/compilerFlags"] = compilerFlags;

      OCCA_HIP_ERROR("Device: Getting HIP Device Arch",
                     hipDeviceComputeCapability(&archMajorVersion,
                                                &archMinorVersion,
                                                hipDevice) );

      archMajorVersion = properties.get("hip/arch/major", archMajorVersion);
      archMinorVersion = properties.get("hip/arch/minor", archMinorVersion);

      properties["kernel/verbose"] = properties.get("verbose", false);
    }

    device::~device() {}

    void device::free() {
      if (hipContext) {
        OCCA_HIP_ERROR("Device: Freeing Context",
                       hipCtxDestroy(hipContext) );
        hipContext = NULL;
      }
    }

    void device::finish() const {
      OCCA_HIP_ERROR("Device: Finish",
                     hipStreamSynchronize(*((hipStream_t*) currentStream)) );
      hipDeviceSynchronize();
    }

    bool device::hasSeparateMemorySpace() const {
      return true;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        std::stringstream ss;
        ss << "major: " << archMajorVersion << ' '
           << "minor: " << archMinorVersion;
        hash_ = occa::hash(ss.str());
      }
      return hash_;
    }

    //---[ Stream ]---------------------
    stream_t device::createStream() const {
      hipStream_t *retStream = new hipStream_t;

      OCCA_HIP_ERROR("Device: Setting Context",
                     hipCtxSetCurrent(hipContext));
      OCCA_HIP_ERROR("Device: createStream",
                     hipStreamCreate(retStream));

      return retStream;
    }

    void device::freeStream(stream_t s) const {
      OCCA_HIP_ERROR("Device: freeStream",
                     hipStreamDestroy( *((hipStream_t*) s) ));
      delete (hipStream_t*) s;
    }

    streamTag device::tagStream() const {
      streamTag ret;

      OCCA_HIP_ERROR("Device: Setting Context",
                     hipCtxSetCurrent(hipContext));
      OCCA_HIP_ERROR("Device: Tagging Stream (Creating Tag)",
                     hipEventCreate(&hip::event(ret)));
      OCCA_HIP_ERROR("Device: Tagging Stream",
                     hipEventRecord(hip::event(ret), 0));

      return ret;
    }

    void device::waitFor(streamTag tag) const {
      OCCA_HIP_ERROR("Device: Waiting For Tag",
                     hipEventSynchronize(hip::event(tag)));
    }

    double device::timeBetween(const streamTag &startTag,
                               const streamTag &endTag) const {
      OCCA_HIP_ERROR("Device: Waiting for endTag",
                     hipEventSynchronize(hip::event(endTag)));

      float msTimeTaken;
      OCCA_HIP_ERROR("Device: Timing Between Tags",
                     hipEventElapsedTime(&msTimeTaken, hip::event(startTag), hip::event(endTag)));

      return (double) (1.0e-3 * (double) msTimeTaken);
    }
    //==================================

    //---[ Kernel ]---------------------
    bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const std::string &hostOutputFile,
                           const occa::properties &kernelProps,
                           lang::kernelMetadataMap &hostMetadata,
                           lang::kernelMetadataMap &deviceMetadata) {
      lang::okl::hipParser parser(kernelProps);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      if (!parser.succeeded()) {
        if (!kernelProps.get("silent", false)) {
          OCCA_FORCE_ERROR("Unable to transform OKL kernel");
        }
        return false;
      }

      if (!io::isFile(outputFile)) {
        hash_t hash = occa::hash(outputFile);
        io::lock_t lock(hash, "hip-parser-device");
        if (lock.isMine()) {
          parser.writeToFile(outputFile);
        }
      }

      if (!io::isFile(hostOutputFile)) {
        hash_t hash = occa::hash(hostOutputFile);
        io::lock_t lock(hash, "hip-parser-host");
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
      const std::string binaryFilename = hashDir + kc::binaryFile+".adipose";
      bool foundBinary = true;
      bool usingOKL = allProps.get("okl", true);

      io::lock_t lock(kernelHash, "hip-kernel");
      if (lock.isMine()) {
        if (io::isFile(binaryFilename)) {
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
          return buildOKLKernelFromBinary(hashDir,
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
          .getModeDevice()
          ->writeKernelBuildFile(hashDir + kc::hostBuildFile,
                                 kernelHash,
                                 occa::properties(),
                                 hostMetadata);

        writeKernelBuildFile(hashDir + kc::buildFile,
                             kernelHash,
                             allProps,
                             deviceMetadata);
      }

      compileKernel(hashDir,
                    kernelName,
                    allProps,
                    lock);

      // Regular HIP Kernel
      if (!launcherKernel) {
        hipModule_t hipModule;
        hipFunction_t hipFunction;
        hipError_t error;

        error = hipModuleLoad(&hipModule, binaryFilename.c_str());
        if (error) {
          lock.release();
          OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Module",
                         error);
        }
        error = hipModuleGetFunction(&hipFunction,
                                     hipModule,
                                     kernelName.c_str());
        if (error) {
          lock.release();
          OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Function",
                         error);
        }
        return new kernel(this,
                          kernelName,
                          sourceFilename,
                          hipModule,
                          hipFunction,
                          allProps);
      }

      return buildOKLKernelFromBinary(hashDir,
                                      kernelName,
                                      hostMetadata,
                                      deviceMetadata,
                                      allProps,
                                      lock);
    }

    void device::compileKernel(const std::string &hashDir,
                               const std::string &kernelName,
                               occa::properties &kernelProps,
                               io::lock_t &lock) {

      const bool verbose = kernelProps.get("verbose", false);

      std::string sourceFilename = hashDir + kc::sourceFile;
      std::string binaryFilename = hashDir + kc::binaryFile;
      const std::string ptxBinaryFilename = hashDir + "ptx_binary.o";

      std::stringstream command;

      //---[ Compiling Command ]--------
      command.str("");
      command << kernelProps["compiler"]
              << " --genco "
              << " "       << sourceFilename
              << " -o "    << binaryFilename
              << ' ' << kernelProps["compilerFlags"]
#if (OCCA_OS == OCCA_WINDOWS_OS)
              << " -D OCCA_OS=OCCA_WINDOWS_OS -D _MSC_VER=1800"
#endif
        ;

      if (!verbose) {
        command << " > /dev/null 2>&1";
      }
      const std::string &sCommand = command.str();
      if (verbose) {
        std::cout << sCommand << '\n';
      }

      const int compileError = system(sCommand.c_str());

      lock.release();
      if (compileError) {
        OCCA_FORCE_ERROR("Error compiling [" << kernelName << "],"
                         " Command: [" << sCommand << ']');
      }
      //================================
    }

    kernel_v* device::buildOKLKernelFromBinary(const std::string &hashDir,
                                               const std::string &kernelName,
                                               lang::kernelMetadataMap &hostMetadata,
                                               lang::kernelMetadataMap &deviceMetadata,
                                               const occa::properties &kernelProps,
                                               const io::lock_t &lock) {

      const std::string sourceFilename = hashDir + kc::sourceFile;
      const std::string binaryFilename = hashDir + kc::binaryFile +".adipose";

      hipModule_t hipModule;
      hipError_t error;

      error = hipModuleLoad(&hipModule, binaryFilename.c_str());
      if (error) {
        lock.release();
        OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Module",
                       error);
      }

      // Create wrapper kernel and set launcherKernel
      kernel &k = *(new kernel(this,
                               kernelName,
                               sourceFilename,
                               kernelProps));

      k.launcherKernel = buildLauncherKernel(hashDir,
                                             kernelName,
                                             hostMetadata[kernelName]);

      // Find clKernels
      typedef std::map<int, lang::kernelMetadata> kernelOrderMap;
      kernelOrderMap hipKernelMetadata;

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
        hipKernelMetadata[number] = metadata;
      }

      kernelOrderMap::iterator oit = hipKernelMetadata.begin();
      while (oit != hipKernelMetadata.end()) {
        lang::kernelMetadata &metadata = oit->second;

        hipFunction_t hipFunction;
        error = hipModuleGetFunction(&hipFunction,
                                     hipModule,
                                     metadata.name.c_str());
        if (error) {
          lock.release();
          OCCA_HIP_ERROR("Kernel [" + metadata.name + "]: Loading Function",
                         error);
        }

        kernel *hipKernel = new kernel(this,
                                       metadata.name,
                                       sourceFilename,
                                       hipModule,
                                       hipFunction,
                                       kernelProps);
        hipKernel->dontUseRefs();
        k.hipKernels.push_back(hipKernel);

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
      kernel_v *launcherKernel = hostKernel.getModeKernel();
      launcherKernel->dontUseRefs();

      launcherKernel->metadata = hostMetadata;

      return launcherKernel;
    }

    kernel_v* device::buildKernelFromBinary(const std::string &filename,
                                            const std::string &kernelName,
                                            const occa::properties &kernelProps) {

      occa::properties allProps = properties["kernel"] + kernelProps;

      hipModule_t hipModule;
      hipFunction_t hipFunction;

      OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Module",
                     hipModuleLoad(&hipModule, filename.c_str()));

      OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Function",
                     hipModuleGetFunction(&hipFunction, hipModule, kernelName.c_str()));

      return new kernel(this,
                        kernelName,
                        filename,
                        hipModule,
                        hipFunction,
                        allProps);
    }
    //==================================

    //---[ Memory ]---------------------
    memory_v* device::malloc(const udim_t bytes,
                             const void *src,
                             const occa::properties &props) {

      if (props.get("hip/mapped", false)) {
        return mappedAlloc(bytes, src, props);
      } else if (props.get("hip/managed", false)) {
        return managedAlloc(bytes, src, props);
      }

      hip::memory &mem = *(new hip::memory(props));
      mem.modeDevice = this;
      mem.size = bytes;

      OCCA_HIP_ERROR("Device: Setting Context",
                     hipCtxSetCurrent(hipContext));

      OCCA_HIP_ERROR("Device: malloc",
                     hipMalloc(&(mem.hipPtr), bytes));

      if (src != NULL) {
        mem.copyFrom(src, bytes, 0);
      }
      return &mem;
    }

    memory_v* device::mappedAlloc(const udim_t bytes,
                                  const void *src,
                                  const occa::properties &props) {

      hip::memory &mem = *(new hip::memory(props));
      mem.modeDevice = this;
      mem.size = bytes;

      OCCA_HIP_ERROR("Device: Setting Context",
                     hipCtxSetCurrent(hipContext));
      OCCA_HIP_ERROR("Device: malloc host",
                     hipHostMalloc((void**) &(mem.mappedPtr), bytes));
      OCCA_HIP_ERROR("Device: get device pointer from host",
                     hipHostGetDevicePointer(&(mem.hipPtr),
                                             mem.mappedPtr,
                                             0));

      if (src != NULL) {
        ::memcpy(mem.mappedPtr, src, bytes);
      }
      return &mem;
    }

    memory_v* device::managedAlloc(const udim_t bytes,
                                   const void *src,
                                   const occa::properties &props) {
      hip::memory &mem = *(new hip::memory(props));

      OCCA_FORCE_ERROR("HIP version ["
                       << hip::getVersion()
                       << "] does not support unified memory allocation");

      return &mem;
    }

    udim_t device::memorySize() const {
      return hip::getDeviceMemorySize(hipDevice);
    }
    //==================================
  }
}

#endif
