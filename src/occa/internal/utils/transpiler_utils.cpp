#include <occa/internal/utils/transpiler_utils.h>
#include <occa/internal/utils/env.hpp>
#include <occa/core.hpp>
#include <occa/utils/io.hpp>
#include <occa/internal/utils/sys.hpp>

#include <occa/internal/lang/modes/serial.hpp>
#include <occa/internal/lang/modes/openmp.hpp>
#include <occa/internal/lang/modes/cuda.hpp>
#include <occa/internal/lang/modes/hip.hpp>
#include <occa/internal/lang/modes/opencl.hpp>
#include <occa/internal/lang/modes/metal.hpp>
#include <occa/internal/lang/modes/dpcpp.hpp>
#include <occa/internal/modes.hpp>


#ifdef BUILD_WITH_CLANG_BASED_TRANSPILER
#include <oklt/pipeline/normalizer_and_transpiler.h>
#include <oklt/util/io_helper.h>
#include <oklt/core/error.h>
#endif

namespace occa {
namespace transpiler {

int getTranspilerVersion(const json &options) {
    json jsonTranspileVersion = options["transpiler-version"];
    int transpilerVersion = 2;
    //INFO: have no idea why json here has array type
    if(!jsonTranspileVersion.isArray()) {
        return transpilerVersion;
    }
    json elem = jsonTranspileVersion.asArray()[0];
    if(!elem.isString()) {
        return transpilerVersion;
    }

    try {
        transpilerVersion = std::stoi(elem.string());
    } catch(const std::exception &)
    {
        return transpilerVersion;
    }
    return transpilerVersion;
}

std::string getKernelHash(const json &kernelProp) {
    auto hashStr = kernelProp.get<std::string>("hash");
    if(hashStr.empty()) {
        throw std::runtime_error("kernel proerties does not contain hash entry");
    }
    return hashStr;
}

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

        std::string defineString = define + "=" + value.toString();
        definesStrings.push_back(std::move(defineString));
        ++it;
    }
    return definesStrings;
}

std::vector<std::filesystem::path> buildIncludes(const json &kernelProp) {
    std::vector<std::filesystem::path> includes;
    json oklIncludePaths = kernelProp.get("okl/include_paths", json{});
    if (!oklIncludePaths.isArray()) {
        return {};
    }

    jsonArray pathArray = oklIncludePaths.array();
    const int pathCount = (int) pathArray.size();
    for (int i = 0; i < pathCount; ++i) {
        json path = pathArray[i];
        if (path.isString()) {
            includes.push_back(std::filesystem::path(path.string()));
        }
    }

    auto &envIncludes = env::OCCA_INCLUDE_PATH;
    for(const auto &includePath: envIncludes) {
        includes.push_back(includePath);
    }
    return includes;
}

void makeMetadata(lang::sourceMetadata_t &sourceMetadata,
                  const std::string &jsonStr)
{

    lang::kernelMetadataMap &metadataMap = sourceMetadata.kernelsMetadata;
    auto json = json::parse(jsonStr);
    auto metadataObj = json.get<occa::json>("metadata");
    if(!metadataObj.isArray()) {
        printError("Can't get the metadata");
        return;
    }
    jsonArray metaArr = metadataObj.asArray().array();
    for(const auto &elem: metaArr) {
        auto name = elem.get<std::string>("name");
        auto kernelObj  = lang::kernelMetadata_t::fromJson(elem);
        metadataMap.insert(std::make_pair(name, kernelObj));
    }
}


namespace v2 {
    bool runTranspiler(const json &options,
                       const json &arguments,
                       const json &kernelProps,
                       const std::string &originalMode,
                       const std::string &mode)
    {
        using ParserBuildFunc = std::function<std::unique_ptr<lang::parser_t>(const json &params)>;
        static const std::map<std::string, ParserBuildFunc> originalParserBackends =
            {
                {"", [](const json &params) {
                     return std::make_unique<lang::okl::serialParser>(params);
                 }},
                {"serial", [](const json &params) {
                     return std::make_unique<lang::okl::serialParser>(params);
                 }},
                {"openmp", [](const json &params) {
                     return std::make_unique<lang::okl::openmpParser>(params);
                 }},
                {"cuda", [](const json &params) {
                     return std::make_unique<lang::okl::cudaParser>(params);
                 }},
                {"hip", [](const json &params) {
                     return std::make_unique<lang::okl::hipParser>(params);
                 }},
                {"opencl", [](const json &params) {
                     return std::make_unique<lang::okl::openclParser>(params);
                 }},
                {"metal", [](const json &params) {
                     return std::make_unique<lang::okl::metalParser>(params);
                 }},
                {"dpcpp", [](const json &params) {
                     return std::make_unique<lang::okl::dpcppParser>(params);
                 }}
            };

        const bool printLauncher = options["launcher"];
        const std::string filename = arguments[0];

        if (!io::exists(filename)) {
            printError("File [" + filename + "] doesn't exist" );
            ::exit(1);
        }

        auto parserIt = originalParserBackends.find(mode);
        if(parserIt == originalParserBackends.end()) {
            printError("Unable to translate for mode [" + originalMode + "]");
            ::exit(1);
        }

        std::unique_ptr<lang::parser_t> parser = parserIt->second(kernelProps);
        parser->parseFile(filename);

        bool success = parser->succeeded();
        if (!success) {
            ::exit(1);
        }

        if (options["verbose"]) {
            json translationInfo;
            // Filename
            translationInfo["translate_info/filename"] = io::expandFilename(filename);
            // Date information
            translationInfo["translate_info/date"] = sys::date();
            translationInfo["translate_info/human_date"] = sys::humanDate();
            // Version information
            translationInfo["translate_info/occa_version"] = OCCA_VERSION_STR;
            translationInfo["translate_info/okl_version"] = OKL_VERSION_STR;
            // Kernel properties
            translationInfo["kernel_properties"] = kernelProps;

            io::stdout
                << "/* Translation Info:\n"
                << translationInfo
                << "*/\n";
        }

        if (printLauncher && ((mode == "cuda")
                              || (mode == "hip")
                              || (mode == "opencl")
                              || (mode == "dpcpp")
                              || (mode == "metal"))) {
            lang::parser_t *launcherParser = &(((occa::lang::okl::withLauncher*) parser.get())->launcherParser);
            io::stdout << launcherParser->toString();
        } else {
            io::stdout << parser->toString();
        }
        return true;
    }
}

#ifdef BUILD_WITH_CLANG_BASED_TRANSPILER
namespace v3 {
    bool runTranspiler(const json &options,
                       const json &arguments,
                       const json &kernelProps,
                       const std::string &originalMode,
                       const std::string &mode)
    {
        static const std::map<std::string, oklt::TargetBackend> targetBackends =
            {
             {"openmp", oklt::TargetBackend::OPENMP},
             {"cuda", oklt::TargetBackend::CUDA},
             {"hip", oklt::TargetBackend::HIP},
             {"dpcpp", oklt::TargetBackend::DPCPP},
             {"serial", oklt::TargetBackend::SERIAL},
             };

        const bool printLauncher = options["launcher"];
        const std::string filename = arguments[0];

        if (!io::exists(filename)) {
            printError("File [" + filename + "] doesn't exist" );
            ::exit(1);
        }

        auto transpiler = targetBackends.find(mode);
        if(transpiler == targetBackends.end()) {
            printError("Unable to translate for mode [" + originalMode + "]");
            ::exit(1);
        }

        auto defines = transpiler::buildDefines(kernelProps);
        auto includes = transpiler::buildIncludes(kernelProps);
        auto hash = transpiler::getKernelHash(kernelProps);

        std::filesystem::path sourcePath = io::expandFilename(filename);
        auto sourceCode = oklt::util::readFileAsStr(sourcePath);
        if(!sourceCode) {
            printError("Can't open file: " + sourcePath.string());
            ::exit(sourceCode.error());
        }
        oklt::UserInput input {
            .backend = transpiler->second,
            .source = std::move(sourceCode.value()),
            .headers = {},
            .sourcePath = sourcePath,
            .includeDirectories = std::move(includes),
            .defines = std::move(defines),
            .hash = std::move(hash)
        };
        auto result = normalizeAndTranspile(std::move(input));

        if(!result) {
            std::stringstream ss;
            for(const auto &err: result.error()) {
                ss << err.desc << std::endl;
            }
            printError(ss.str());
            ::exit(1);
        }

        if (options["verbose"]) {
            json translationInfo;
            // Filename
            translationInfo["translate_info/filename"] = io::expandFilename(filename);
            // Date information
            translationInfo["translate_info/date"] = sys::date();
            translationInfo["translate_info/human_date"] = sys::humanDate();
            // Version information
            translationInfo["translate_info/occa_version"] = OCCA_VERSION_STR;
            translationInfo["translate_info/okl_version"] = OKL_VERSION_STR;
            // Kernel properties
            translationInfo["kernel_properties"] = kernelProps;

            io::stdout
                << "/* Translation Info:\n"
                << translationInfo
                << "*/\n";
        }

        auto userOutput = result.value();
        bool hasLauncher = transpiler->second == oklt::TargetBackend::CUDA ||
                           transpiler->second == oklt::TargetBackend::HIP ||
                           transpiler->second == oklt::TargetBackend::DPCPP;
        if(printLauncher && hasLauncher) {
            io::stdout << userOutput.launcher.source;
        } else {
            io::stdout << userOutput.kernel.source;
        }

        return true;
    }
}
#endif

bool runTranspiler(const json &options,
                   const json &arguments,
                   const json &kernelProps,
                   const std::string &originalMode,
                   const std::string &mode)
{
    int transpilerVersion = getTranspilerVersion(options);
#ifdef BUILD_WITH_CLANG_BASED_TRANSPILER
    if(transpilerVersion > 2) {
        return v3::runTranspiler(options, arguments, kernelProps, originalMode, mode);
    }
#endif
    if (transpilerVersion > 2) {
        printError("OCCA compiler is built without BUILD_WITH_CLANG_BASED_TRANSPILER support");
        return false;
    }
    return v2::runTranspiler(options, arguments, kernelProps, originalMode, mode);
}

}
}
