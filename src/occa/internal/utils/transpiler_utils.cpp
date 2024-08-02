#include <occa/internal/utils/transpiler_utils.h>
#include <occa/internal/utils/env.hpp>
#include <occa/core.hpp>
#include <occa/utils/io.hpp>
#include <occa/internal/io/utils.hpp>
#include <occa/internal/utils/sys.hpp>

#ifdef BUILD_WITH_CLANG_BASED_TRANSPILER
#include <oklt/pipeline/normalizer_and_transpiler.h>
#include <oklt/util/io_helper.h>
#include <oklt/core/error.h>
#endif

namespace occa {
namespace transpiler {

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
    if(metadataObj.isArray()) {
        jsonArray metaArr = metadataObj.asArray().array();
        for(const auto &elem: metaArr) {
            auto name = elem.get<std::string>("name");
            auto kernelObj  = lang::kernelMetadata_t::fromJson(elem);
            metadataMap.insert(std::make_pair(name, kernelObj));
        }
    }
}

#if BUILD_WITH_CLANG_BASED_TRANSPILER
Transpiler::Transpiler(SuccessFunc success,
           FailFunc fail,
           WrongInputFile wrongInputFile,
           WrongBackend wrongBackend)
    :_success(std::move(success))
    , _fail(std::move(fail))
    , _wrongInput(std::move(wrongInputFile))
    , _wrongBackend(std::move(wrongBackend))
{}

bool Transpiler::run(const std::string &filename,
     const std::string &mode,
     const occa::json &kernelProps)
{
    static const std::map<std::string, oklt::TargetBackend> targetBackends =
        {
         {"openmp", oklt::TargetBackend::OPENMP},
         {"cuda", oklt::TargetBackend::CUDA},
         {"hip", oklt::TargetBackend::HIP},
         {"dpcpp", oklt::TargetBackend::DPCPP},
         {"serial", oklt::TargetBackend::SERIAL},
    };

    std::string normalizedMode = lowercase(mode);
    auto backend = targetBackends.find(normalizedMode);
    if(backend == targetBackends.end()) {
        _wrongBackend(mode);
        return false;
    }
    auto expandedFile = io::expandFilename(filename);
    if (!io::exists(filename)) {
        _wrongInput(filename);
        return false;
    }
    auto sourceCode = oklt::util::readFileAsStr(expandedFile);
    if(!sourceCode) {
        _wrongInput(filename);
        return false;
    }

    auto defines = transpiler::buildDefines(kernelProps);
    auto includes = transpiler::buildIncludes(kernelProps);
    auto hash = transpiler::getKernelHash(kernelProps);

    oklt::UserInput input {
        .backend = backend->second,
        .source = std::move(sourceCode.value()),
        .headers = {},
        .sourcePath = expandedFile,
        .includeDirectories = std::move(includes),
        .defines = std::move(defines),
        .hash = std::move(hash)
    };
    auto result = normalizeAndTranspile(std::move(input));
    if(!result) {
        _fail(result.error());
        return false;
    }
    bool hasLauncher = backend->second == oklt::TargetBackend::CUDA ||
                       backend->second == oklt::TargetBackend::HIP ||
                       backend->second == oklt::TargetBackend::DPCPP;
    auto userOutput = result.value();
    return _success(userOutput, hasLauncher);
}
#endif
}
}
