#include <map>
#include <fstream>

#include <occa/core/base.hpp>
#include <occa/types/primitive.hpp>
#include <occa/internal/core/launchedDevice.hpp>
#include <occa/internal/core/launchedKernel.hpp>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/kernel.hpp>
#include <occa/internal/utils/string.hpp>


#include "oklt/pipeline/normalizer_and_transpiler.h"
#include "oklt/core/error.h"


namespace occa {
  launchedModeDevice_t::launchedModeDevice_t(const occa::json &properties_) :
    modeDevice_t(properties_) {
    needsLauncherKernel = true;
  }

  bool launchedModeDevice_t::transpileFile(const std::string &filename,
                     const std::string &outputFile,
                     const std::string &launcherOutputFile,
                     const occa::json &kernelProps,
                     lang::sourceMetadata_t &launcherMetadata,
                     lang::sourceMetadata_t &deviceMetadata)
  {
    static const std::map<std::string, oklt::TargetBackend> targetBackends =
        {
         // {"openmp", oklt::TargetBackend::OPENMP},
         {"cuda", oklt::TargetBackend::CUDA},
         {"hip", oklt::TargetBackend::HIP},
         {"dpcpp", oklt::TargetBackend::DPCPP},
    };

    std::string normalized_mode = lowercase(mode);
    auto targetIter = targetBackends.find(normalized_mode);
    if(targetIter == targetBackends.end()) {
        std::string errorDescription = "Can't find target backend: " + mode;
        OCCA_FORCE_ERROR(errorDescription +
                             ", unable to transform OKL kernel [" << filename << "]");
        return false;
    }

    std::string fullFilePath = io::expandFilename(filename);
    std::ifstream sourceFile(fullFilePath);
    std::string sourceCode{std::istreambuf_iterator<char>(sourceFile), {}};
    oklt::UserInput input {
        .backend = oklt::TargetBackend::CUDA,
        .astProcType = oklt::AstProcessorType::OKL_WITH_SEMA,
        .sourceCode = std::move(sourceCode),
        .sourcePath = std::filesystem::path(fullFilePath),
        .inlcudeDirectories = {},
        .defines = {}
    };
    auto result = normalizeAndTranspile(std::move(input));

    if(!result) {
        std::stringstream ss;
        ss << "Unable to transform OKL kernel [" << filename << "]" << std::endl;
        ss << "Transpilation errors occured: " << std::endl;
        for(const auto &err: result.error()) {
            ss << err.desc << std::endl;
        }
        OCCA_FORCE_ERROR(ss.str());
        return false;
    }

    auto userOutput = result.value();
    io::stageFiles(
        { outputFile, launcherOutputFile },
        true,
        [&](const strVector &tempFilenames) -> bool {
            const std::string &tempOutputFilename = tempFilenames[0];
            const std::string &tempLauncherOutputFilename = tempFilenames[1];

            std::ofstream transpiledTempOutput(tempOutputFilename);
            transpiledTempOutput << userOutput.kernel.sourceCode;

            //TODO: need to check
            std::ofstream laucherOutputFile(tempLauncherOutputFilename);
            laucherOutputFile << userOutput.launcher.sourceCode;
            return true;
        });
    //TODO: make convertors
    // launcherMetadata = userOutput.launcher.metadataJson;
    // deviceMetadata = userOutput.kernel.metadataJson;

    return true;
  }

  namespace impl_v3 {
      std::vector<std::string> buildDefines(const json &kernelProp) {
        const json &defines = kernelProp["defines"];
        if (!defines.isObject()) {
            {};
        }

        std::vector<std::string> definesStrings;
        const jsonObject &defineMap = defines.object();
        jsonObject::const_iterator it = defineMap.cbegin();
        while (it != defineMap.end()) {
            const std::string &define = it->first;
            const json value = it->second;

            //preprocessor.addSourceDefine(define, value);
            std::string defineString = define + "=" + value.toString();
            definesStrings.push_back(std::move(defineString));
            ++it;
        }
        return definesStrings;
      }

      std::vector<std::filesystem::path> buildIncludes(const json &kernelProp) {
        std::vector<std::filesystem::path> includes;
        json oklIncludePaths = kernelProp.get("okl/include_paths", json{});
        if (oklIncludePaths.isArray()) {
            jsonArray pathArray = oklIncludePaths.array();
            const int pathCount = (int) pathArray.size();
            for (int i = 0; i < pathCount; ++i) {
                json path = pathArray[i];
                if (path.isString()) {
                    includes.push_back(std::filesystem::path(path.string()));
                }
            }
        }
        return includes;
      }
  }
  bool launchedModeDevice_t::parseFile(const std::string &filename,
                                       const std::string &outputFile,
                                       const std::string &launcherOutputFile,
                                       const occa::json &kernelProps,
                                       lang::sourceMetadata_t &launcherMetadata,
                                       lang::sourceMetadata_t &deviceMetadata)
  {
    std::unique_ptr<lang::okl::withLauncher> parser(createParser(kernelProps));
    parser->parseFile(filename);


    // ============ HOT HACK FOR DEMO, use new transpiler directly ==============
    static const std::map<std::string, oklt::TargetBackend> targetBackends =
        {
         // {"openmp", oklt::TargetBackend::OPENMP},
         {"cuda", oklt::TargetBackend::CUDA},
         {"hip", oklt::TargetBackend::HIP},
         {"dpcpp", oklt::TargetBackend::DPCPP},
         };

    std::string normalized_mode = lowercase(mode);
    auto targetIter = targetBackends.find(normalized_mode);
    if(targetIter == targetBackends.end()) {
        std::string errorDescription = "Can't find target backend: " + mode;
        OCCA_FORCE_ERROR(errorDescription +
                             ", unable to transform OKL kernel [" << filename << "]");
        return false;
    }
    auto defines = impl_v3::buildDefines(kernelProps);
    auto includes = impl_v3::buildIncludes(kernelProps);

    std::string fullFilePath = io::expandFilename(filename);
    std::ifstream sourceFile(fullFilePath);
    std::string sourceCode{std::istreambuf_iterator<char>(sourceFile), {}};
    oklt::UserInput input {
        .backend = oklt::TargetBackend::CUDA,
        .astProcType = oklt::AstProcessorType::OKL_WITH_SEMA,
        .sourceCode = std::move(sourceCode),
        .sourcePath = std::filesystem::path(fullFilePath),
        .inlcudeDirectories = std::move(includes),
        .defines = std::move(defines)
    };
    auto result = normalizeAndTranspile(std::move(input));

    if(!result) {
        std::stringstream ss;
        ss << "Unable to transform OKL kernel [" << filename << "]" << std::endl;
        ss << "Transpilation errors occured: " << std::endl;
        for(const auto &err: result.error()) {
            ss << err.desc << std::endl;
        }
        OCCA_FORCE_ERROR(ss.str());
        return false;
    }
    // ============ HOT HACK FOR DEMO, use new transpiler directly ==============

    // Verify if parsing succeeded
    if (!parser->succeeded()) {
      if (!kernelProps.get("silent", false)) {
        OCCA_FORCE_ERROR("Unable to transform OKL kernel [" << filename << "]");
      }
      return false;
    }

    auto &userOutput = result.value();
    io::stageFiles(
      { outputFile, launcherOutputFile },
      true,
      [&](const strVector &tempFilenames) -> bool {
        const std::string &tempOutputFilename = tempFilenames[0];
        const std::string &tempLauncherOutputFilename = tempFilenames[1];

        // ============ HOT HACK FOR DEMO, use new transpiler directly ==============
        // parser->writeToFile(tempOutputFilename);
        std::ofstream ofs(tempOutputFilename);
        ofs << userOutput.kernel.sourceCode;
        // ============ HOT HACK FOR DEMO, use new transpiler directly ==============
        parser->launcherParser.writeToFile(tempLauncherOutputFilename);

        return true;
      }
    );

    parser->launcherParser.setSourceMetadata(launcherMetadata);
    parser->setSourceMetadata(deviceMetadata);

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
    std::string sourceFilename = hashDir + kc::cachedSourceFilename(filename);
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

      modeKernel_t* k;
      if (usingOkl) {
        lang::sourceMetadata_t launcherMetadata = (
          lang::sourceMetadata_t::fromBuildFile(hashDir + kc::launcherBuildFile)
        );
        lang::sourceMetadata_t deviceMetadata = (
          lang::sourceMetadata_t::fromBuildFile(hashDir + kc::buildFile)
        );
        k = buildOKLKernelFromBinary(kernelHash,
                                        hashDir,
                                        kernelName,
                                        sourceFilename,
                                        binaryFilename,
                                        launcherMetadata,
                                        deviceMetadata,
                                        kernelProps);
      } else {
        k = buildKernelFromBinary(binaryFilename,
                                     kernelName,
                                     kernelProps);
      }
      if (k) {
        k->sourceFilename = filename;
        k->binaryFilename = binaryFilename;
      }
      return k;
    }

    lang::sourceMetadata_t launcherMetadata, deviceMetadata;
    if (usingOkl) {
      // Cache raw origin
      sourceFilename = io::cacheFile(filename,
                                     kc::cachedRawSourceFilename(filename),
                                     kernelHash,
                                     assembleKernelHeader(kernelProps));

      int transpilerVersion = kernelProps.get("transpiler-version", 2);
      const std::string outputFile = hashDir + kc::cachedSourceFilename(filename);
      const std::string launcherOutputFile = hashDir + kc::launcherSourceFile;

      bool isValid = false;
      if(transpilerVersion > 2) {
        isValid = transpileFile(sourceFilename,
                                outputFile,
                                launcherOutputFile,
                                kernelProps,
                                launcherMetadata,
                                deviceMetadata);
      } else {
          isValid = parseFile(sourceFilename,
                                 outputFile,
                                 launcherOutputFile,
                                 kernelProps,
                                 launcherMetadata,
                                 deviceMetadata);

      }

      if (!isValid) {
          return nullptr;
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
                      kc::cachedSourceFilename(filename),
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

    if (k) {
      k->sourceFilename = filename;
      k->binaryFilename = binaryFilename;
    }
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
