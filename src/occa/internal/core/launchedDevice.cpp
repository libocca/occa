#include <map>

#include <occa/core/base.hpp>
#include <occa/types/primitive.hpp>
#include <occa/internal/core/launchedDevice.hpp>
#include <occa/internal/core/launchedKernel.hpp>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/kernel.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  launchedModeDevice_t::launchedModeDevice_t(const occa::json &properties_) :
    modeDevice_t(properties_) {
    needsLauncherKernel = true;
  }

  bool launchedModeDevice_t::parseFile(const std::string &filename,
                                       const std::string &outputFile,
                                       const std::string &launcherOutputFile,
                                       const occa::json &kernelProps,
                                       lang::sourceMetadata_t &launcherMetadata,
                                       lang::sourceMetadata_t &deviceMetadata) {
    lang::okl::withLauncher &parser = *(createParser(kernelProps));
    parser.parseFile(filename);

    // Verify if parsing succeeded
    if (!parser.succeeded()) {
      if (!kernelProps.get("silent", false)) {
        OCCA_FORCE_ERROR("Unable to transform OKL kernel [" << filename << "]");
      }
      delete &parser;
      return false;
    }

    io::stageFiles(
      { outputFile, launcherOutputFile },
      true,
      [&](const strVector &tempFilenames) -> bool {
        const std::string &tempOutputFilename = tempFilenames[0];
        const std::string &tempLauncherOutputFilename = tempFilenames[1];

        parser.writeToFile(tempOutputFilename);
        parser.launcherParser.writeToFile(tempLauncherOutputFilename);

        return true;
      }
    );

    parser.launcherParser.setSourceMetadata(launcherMetadata);
    parser.setSourceMetadata(deviceMetadata);

    delete &parser;
    return true;
  }

  modeKernel_t* launchedModeDevice_t::buildKernel(const std::string &filename,
                                                  const std::string &kernelName,
                                                  const hash_t kernelHash,
                                                  const occa::json &kernelProps) {
    bool usingOkl = kernelProps.get("okl/enabled", true);

    launchedModeKernel_t *kernel = (launchedModeKernel_t*) (
      buildKernel(filename,
                  kernelName,
                  kernelHash,
                  usingOkl,
                  kernelProps)
    );

    if (usingOkl) {
      std::vector<modeKernel_t*> &deviceKernels = kernel->deviceKernels;
      const int kernelCount = (int) deviceKernels.size();
      for (int i = 0; i < kernelCount; ++i) {
        modeKernel_t *deviceKernel = deviceKernels[i];

        // The launchedKernel handles deleting the launcher + device kernels
        removeKernelRef(deviceKernel);
        deviceKernel->dontUseRefs();

        // Some backends inject additional arguments
        deviceKernel->properties["type_validation"] = false;
      }
    }

    return kernel;
  }

  modeKernel_t* launchedModeDevice_t::buildKernel(const std::string &filename,
                                                  const std::string &kernelName,
                                                  const hash_t kernelHash,
                                                  const bool usingOkl,
                                                  const occa::json &kernelProps) {
    const std::string hashDir = io::hashDir(filename, kernelHash);
    std::string sourceFilename = hashDir + kc::sourceFile;
    const std::string binaryFilename = hashDir + kc::binaryFile;

    // Check if binary exists and is finished
    const bool foundBinary = io::isFile(binaryFilename);

    const bool verbose = kernelProps.get("verbose", false);
    if (foundBinary) {
      if (verbose) {
        io::stdout << "Loading cached ["
                   << kernelName
                   << "] from ["
                   << filename
                   << "] in [" << binaryFilename << "]\n";
      }

      if (usingOkl) {
        lang::sourceMetadata_t launcherMetadata = (
          lang::sourceMetadata_t::fromBuildFile(hashDir + kc::launcherBuildFile)
        );
        lang::sourceMetadata_t deviceMetadata = (
          lang::sourceMetadata_t::fromBuildFile(hashDir + kc::buildFile)
        );
        return buildOKLKernelFromBinary(kernelHash,
                                        hashDir,
                                        kernelName,
                                        sourceFilename,
                                        binaryFilename,
                                        launcherMetadata,
                                        deviceMetadata,
                                        kernelProps);
      } else {
        return buildKernelFromBinary(binaryFilename,
                                     kernelName,
                                     kernelProps);
      }
    }

    lang::sourceMetadata_t launcherMetadata, deviceMetadata;
    if (usingOkl) {
      // Cache raw origin
      sourceFilename = (
        io::cacheFile(filename,
                      kc::cppRawSourceFile,
                      kernelHash,
                      assembleKernelHeader(kernelProps))
      );

      const std::string outputFile = hashDir + kc::sourceFile;
      const std::string launcherOutputFile = hashDir + kc::launcherSourceFile;
      bool valid = parseFile(sourceFilename,
                             outputFile,
                             launcherOutputFile,
                             kernelProps,
                             launcherMetadata,
                             deviceMetadata);
      if (!valid) {
        return NULL;
      }
      sourceFilename = outputFile;

      buildLauncherKernel(kernelHash,
                          hashDir,
                          kernelName,
                          launcherMetadata);

      // No OKL means no build file is generated,
      //   so we need to build it
      host()
        .getModeDevice()
        ->writeKernelBuildFile(hashDir + kc::launcherBuildFile,
                               kernelHash,
                               occa::json(),
                               launcherMetadata);

      writeKernelBuildFile(hashDir + kc::buildFile,
                           kernelHash,
                           kernelProps,
                           deviceMetadata);
    } else {
      // Cache in sourceFile to directly compile file
      sourceFilename = (
        io::cacheFile(filename,
                      kc::sourceFile,
                      kernelHash,
                      assembleKernelHeader(kernelProps))
      );
    }

    modeKernel_t *k;
    io::stageFile(
      binaryFilename,
      false,
      [&](const std::string &tempFilename) -> bool {
        k = buildKernelFromProcessedSource(
          kernelHash,
          hashDir,
          kernelName,
          sourceFilename,
          tempFilename,
          usingOkl,
          launcherMetadata,
          deviceMetadata,
          kernelProps
        );
        return true;
      }
    );

    return k;
  }

  modeKernel_t* launchedModeDevice_t::buildLauncherKernel(
    const hash_t kernelHash,
    const std::string &hashDir,
    const std::string &kernelName,
    lang::sourceMetadata_t sourceMetadata
  ) {
    const std::string launcherOutputFile = hashDir + kc::launcherSourceFile;

    serial::device *hostDevice = (serial::device*) host().getModeDevice();

    modeKernel_t *launcherKernel = hostDevice->buildLauncherKernel(
      launcherOutputFile,
      kernelName,
      kernelHash
    );
    if (!launcherKernel) {
      return NULL;
    }

    // Launcher and device kernels use the same refs as the wrapper kernel
    launcherKernel->dontUseRefs();
    launcherKernel->metadata = sourceMetadata.kernelsMetadata[kernelName];

    return launcherKernel;
  }

  orderedKernelMetadata launchedModeDevice_t::getLaunchedKernelsMetadata(
    const std::string &kernelName,
    lang::sourceMetadata_t &deviceMetadata
  ) {
    // Find device kernels
    typedef std::map<int, lang::kernelMetadata_t> kernelOrderMap;
    kernelOrderMap kernelMetadataMap;

    const std::string prefix = "_occa_" + kernelName + "_";


    lang::kernelMetadataMap &kernelsMetadata = deviceMetadata.kernelsMetadata;

    lang::kernelMetadataMap::iterator it = kernelsMetadata.begin();
    while (it != kernelsMetadata.end()) {
      const std::string &name = it->first;
      lang::kernelMetadata_t &metadata = it->second;
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
      kernelMetadataMap[number] = metadata;
    }

    // Setup vector from ordered metadata
    orderedKernelMetadata kernelMetadata;
    kernelOrderMap::iterator kIt;
    for (kIt = kernelMetadataMap.begin(); kIt != kernelMetadataMap.end(); ++kIt) {
      kernelMetadata.push_back(kIt->second);
    }
    return kernelMetadata;
  }
}
