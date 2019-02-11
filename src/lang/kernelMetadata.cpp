#include <occa/lang/kernelMetadata.hpp>
#include <occa/io/utils.hpp>
#include <occa/tools/properties.hpp>

namespace occa {
  namespace lang {
    argumentInfo::argumentInfo() :
      isConst(false),
      isPtr(false),
      dtype(dtype::byte) {}

    argumentInfo::argumentInfo(const bool isConst_,
                               const bool isPtr_,
                               const dtype_t &dtype_) :
      isConst(isConst_),
      isPtr(isPtr_),
      dtype(dtype_) {}

    argumentInfo argumentInfo::fromJson(const json &j) {
      return argumentInfo((bool) j["const"],
                          (bool) j["ptr"],
                          dtype_t::fromJson(j["dtype"]));
    }

    json argumentInfo::toJson() const {
      json j;
      j["const"] = isConst;
      j["ptr"]   = isPtr;
      j["dtype"] = dtype.toJson();
      return j;
    }

    kernelMetadata::kernelMetadata() :
      initialized(false) {}

    bool kernelMetadata::isInitialized() const {
      return initialized;
    }

    kernelMetadata& kernelMetadata::operator += (const argumentInfo &argInfo) {
      initialized = true;
      arguments.push_back(argInfo);
      return *this;
    }

    kernelMetadata kernelMetadata::fromJson(const json &j) {
      kernelMetadata meta;
      meta.initialized = true;

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
