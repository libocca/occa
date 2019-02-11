#include <occa/lang/kernelMetadata.hpp>
#include <occa/io/utils.hpp>
#include <occa/tools/properties.hpp>

namespace occa {
  namespace lang {
    argumentInfo::argumentInfo(const bool isConst_,
                               const dtype_t &dtype_) :
      isConst(isConst_),
      dtype(dtype_) {}

    argumentInfo argumentInfo::fromJson(const json &j) {
      return argumentInfo((bool) j["const"]);
    }

    json argumentInfo::toJson() const {
      json j;
      j["const"] = isConst;
      return j;
    }

    kernelMetadata::kernelMetadata() {}

    kernelMetadata& kernelMetadata::operator += (const argumentInfo &argInfo) {
      arguments.push_back(argInfo);
      return *this;
    }

    bool kernelMetadata::argIsConst(const int pos) const {
      if (pos < (int) arguments.size()) {
        return arguments[pos].isConst;
      }
      return false;
    }

    bool kernelMetadata::argMatchesDtype(const int pos,
                                         const dtype_t &dtype) const {
      if (pos < (int) arguments.size()) {
        const dtype_t &argDtype = arguments[pos].dtype;
        return dtype.canBeCastedTo(argDtype);
      }
      return false;
    }

    kernelMetadata kernelMetadata::fromJson(const json &j) {
      kernelMetadata meta;

      meta.name = (std::string) j["name"];

      const jsonArray &argInfos = j["arguments"].array();
      const int argumentCount = (int) argInfos.size();
      for (int i = 0; i < argumentCount; ++i) {
        meta.arguments.push_back(argumentInfo::fromJson(argInfos[i]));
      }

      return meta;
    }

    json kernelMetadata::toJson() const {
      json j;
      j["name"] = name;

      const int argumentCount = (int) arguments.size();
      json &argInfos = j["arguments"].asArray();
      for (int k = 0; k < argumentCount; ++k) {
        argInfos += arguments[k].toJson();
      }

      return j;
    }

    kernelMetadataMap getBuildFileMetadata(const std::string &filename) {
      kernelMetadataMap metadataMap;
      if (!io::exists(filename)) {
        return metadataMap;
      }

      properties props = properties::read(filename);
      jsonArray &metadata = props["kernel/metadata"].array();

      const int kernelCount = (int) metadata.size();
      for (int i = 0; i < kernelCount; ++i) {
        kernelMetadata kernel = kernelMetadata::fromJson(metadata[i]);
        metadataMap[kernel.name] = kernel;
      }

      return metadataMap;
    }
  }
}
