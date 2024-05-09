#include <occa/internal/utils/transpiler_utils.h>
#include <occa/internal/utils/env.hpp>

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

}
}
