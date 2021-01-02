#include <occa/internal/lang/kernelMetadata.hpp>
#include <occa/internal/io/utils.hpp>
#include <occa/types/json.hpp>

namespace occa {
  namespace lang {
    argMetadata_t::argMetadata_t() :
      isConst(false),
      isPtr(false),
      dtype(dtype::byte) {}

    argMetadata_t::argMetadata_t(const bool isConst_,
                                 const bool isPtr_,
                                 const dtype_t &dtype_,
                                 const std::string &name_) :
      isConst(isConst_),
      isPtr(isPtr_),
      dtype(dtype_),
      name(name_) {}

    argMetadata_t argMetadata_t::fromJson(const json &j) {
      return argMetadata_t((bool) j["const"],
                           (bool) j["ptr"],
                           dtype_t::fromJson(j["dtype"]),
                           (std::string) j["name"]);
    }

    json argMetadata_t::toJson() const {
      json j;
      j["const"] = isConst;
      j["ptr"]   = isPtr;
      j["dtype"] = dtype::toJson(dtype);
      j["name"]  = name;
      return j;
    }

    kernelMetadata_t::kernelMetadata_t() :
      initialized(false) {}

    bool kernelMetadata_t::isInitialized() const {
      return initialized;
    }

    kernelMetadata_t& kernelMetadata_t::operator += (const argMetadata_t &argInfo) {
      initialized = true;
      arguments.push_back(argInfo);
      return *this;
    }

    kernelMetadata_t kernelMetadata_t::fromJson(const json &j) {
      kernelMetadata_t meta;
      meta.initialized = true;

      meta.name = (std::string) j["name"];

      const jsonArray &argInfos = j["arguments"].array();
      const int argumentCount = (int) argInfos.size();
      for (int i = 0; i < argumentCount; ++i) {
        meta.arguments.push_back(argMetadata_t::fromJson(argInfos[i]));
      }

      return meta;
    }

    json kernelMetadata_t::toJson() const {
      json j;
      j["name"] = name;

      const int argumentCount = (int) arguments.size();
      json &argInfos = j["arguments"].asArray();
      for (int k = 0; k < argumentCount; ++k) {
        argInfos += arguments[k].toJson();
      }

      return j;
    }

    sourceMetadata_t::sourceMetadata_t() {}

    json sourceMetadata_t::getKernelMetadataJson() const {
      json metadataJson(json::array_);

      lang::kernelMetadataMap::const_iterator it = kernelsMetadata.begin();
      while (it != kernelsMetadata.end()) {
        metadataJson += (it->second).toJson();
        ++it;
      }

      return metadataJson;
    }

    json sourceMetadata_t::getDependencyJson() const {
      json metadataJson;

      strHashMap::const_iterator it = dependencyHashes.begin();
      while (it != dependencyHashes.end()) {
        metadataJson.set(it->first, it->second.getFullString());
        ++it;
      }

      return metadataJson;
    }

    sourceMetadata_t sourceMetadata_t::fromBuildFile(const std::string &filename) {
      sourceMetadata_t metadata;

      if (!io::exists(filename)) {
        return metadata;
      }

      json props = json::read(filename);
      jsonArray &kernelMetadata = props["kernel/metadata"].array();
      jsonObject &dependencyHashes_ = props["kernel/dependencies"].object();

      kernelMetadataMap &metadataMap = metadata.kernelsMetadata;
      const int kernelCount = (int) kernelMetadata.size();
      for (int i = 0; i < kernelCount; ++i) {
        kernelMetadata_t kernel = kernelMetadata_t::fromJson(kernelMetadata[i]);
        metadataMap[kernel.name] = kernel;
      }

      jsonObject::iterator it = dependencyHashes_.begin();
      while (it != dependencyHashes_.end()) {
        metadata.dependencyHashes[it->first] = hash_t::fromString(it->second);
        ++it;
      }

      return metadata;
    }
  }
}
